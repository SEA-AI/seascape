"""One H.264 video per camera, its frames in the order labels.json times them."""

import math
import tempfile
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path

import bpy
import cv2
import PIL.Image

from seascape.config import RECORDED, Agc, Quality, Scenario
from seascape.outputs import agc, montage
from seascape.outputs.labels import Image, agcs, cameras

# Judgement: small enough to upload while it plays.
PREVIEW: Quality = "HIGH"


def encode(
    run: Path,
    quality: Quality = RECORDED,
    advance: Callable[[], None] = lambda: None,
) -> list[Path]:
    """Write each camera's video, `<camera>.mp4`, into the run's folder.

    Args:
        run: The render's folder, with its labels.json.
        quality: A Blender `constant_rate_factor` preset.
        advance: Runs after each camera's video.
    """
    run = run.resolve()
    settings = agcs(run)
    written = []
    for camera, frames in cameras(run).items():
        written.append(_encode(run, camera, frames, quality, settings[camera]))
        advance()
    return written


def _toned(paths: list[Path], setting: Agc, step_s: float, into: Path) -> list[Path]:
    """8-bit copies of 16-bit thermal frames, which the sequencer shows as flat grey;
    any other frame as it is.
    """
    if agc.kelvin(paths[0]) is None:
        return paths
    tone, toned = agc.Agc(setting, step_s), []
    for i, path in enumerate(paths):
        t_k = agc.kelvin(path)
        if t_k is None:
            raise ValueError(f"{path}: not a 16-bit thermal frame like {paths[0]}")
        toned.append(into / f"{i:04d}.png")
        if not cv2.imwrite(str(toned[-1]), tone(t_k)):
            raise OSError(f"cannot write {toned[-1]}")
    return toned


def _encode(
    run: Path, camera: str, frames: list[Image], quality: Quality, setting: Agc
) -> Path:
    if len(frames) < 2:
        raise ValueError(f"{camera}: one frame is a still, not a video")
    times = [frame.time_s for frame in frames]
    step = times[1] - times[0]
    if step <= 0 or not all(
        math.isclose(b - a, step, rel_tol=1e-6) for a, b in pairwise(times)
    ):
        raise ValueError(f"{camera}: a video needs a constant time step, not {times}")
    paths = [run / frame.file_name for frame in frames]
    # The sequencer encodes a missing frame as black, and says nothing.
    if missing := [str(path) for path in paths if not path.exists()]:
        raise FileNotFoundError(f"{camera}: no frame at {', '.join(missing)}")

    with tempfile.TemporaryDirectory() as tmp:
        paths = _toned(paths, setting, step, Path(tmp))
        size = (frames[0].width, frames[0].height)
        return sequence(paths, size, step, run / f"{camera}.mp4", quality)


def sequence(
    paths: list[Path],
    size_px: tuple[int, int],
    step_s: float,
    out: Path,
    quality: Quality = RECORDED,
) -> Path:
    """Encode frames as H.264. Resets Blender to its factory settings.

    Args:
        paths: 8-bit frames in order, all in one folder.
        size_px: Their width and height.
        step_s: The time between frames.
        out: The mp4 to write, with its extension.
        quality: A Blender `constant_rate_factor` preset.

    Returns:
        `out`.
    """
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    strip = sc.sequence_editor_create().strips.new_image(out.stem, str(paths[0]), 1, 1)
    for path in paths[1:]:
        strip.elements.append(str(path.relative_to(paths[0].parent)))
    sc.frame_end = len(paths)
    r = sc.render
    r.resolution_x, r.resolution_y = size_px
    r.fps, r.fps_base = 1, step_s
    # The factory AgX would tone the frames a second time.
    sc.view_settings.view_transform = "Standard"
    r.image_settings.media_type = "VIDEO"
    r.ffmpeg.format, r.ffmpeg.codec = "MPEG4", "H264"
    r.ffmpeg.constant_rate_factor = quality
    # With its extension: without one Blender appends the frame range to the name.
    # Doubled braces are literal: Blender reads `{name}` as a path template.
    r.filepath = str(out).replace("{", "{{").replace("}", "}}")
    bpy.ops.render.render(animation=True)
    return out


def montage_video(scenario: Scenario, run: Path, out: Path) -> Path:
    """Every camera's frames in `run` laid out as `montage.sheet` lays them out, as
    one video at `PREVIEW` quality, at `out`. Resets Blender to its factory settings.

    Args:
        scenario: The scenario `run` rendered.
        run: The render's folder, holding labels.json.
        out: The mp4 to write.

    Returns:
        `out`.

    Raises:
        FileNotFoundError: No camera has frames, or a frame is missing.
        ValueError: The clips differ in length.
        RuntimeError: Blender failed to encode.
    """
    with tempfile.TemporaryDirectory() as tmp:
        pictures = montage.video_frames(scenario, run, Path(tmp))
        with PIL.Image.open(pictures[0]) as first:
            size = first.size
        return sequence(pictures, size, 1.0 / scenario.outputs.fps, out, PREVIEW)
