"""One H.264 video per camera, its frames in the order labels.json times them."""

import math
import tempfile
from itertools import pairwise
from pathlib import Path

import bpy
import cv2

from seascape import agc
from seascape.labels import FILENAME, Image, Labels


def encode(run: Path) -> list[Path]:
    run = run.resolve()
    images = Labels.model_validate_json((run / FILENAME).read_text()).images
    cameras: dict[str, list[Image]] = {}
    for image in sorted(images, key=lambda image: image.time_s):
        cameras.setdefault(image.camera, []).append(image)
    return [_encode(run, camera, frames) for camera, frames in cameras.items()]


def _toned(paths: list[Path], step_s: float, into: Path) -> list[Path]:
    """8-bit copies of 16-bit thermal frames, which the sequencer shows as flat grey;
    any other frame as it is."""
    if agc.kelvin(paths[0]) is None:
        return paths
    tone, toned = agc.Agc(step_s), []
    for i, path in enumerate(paths):
        t_k = agc.kelvin(path)
        if t_k is None:
            raise ValueError(f"{path}: not a 16-bit thermal frame like {paths[0]}")
        toned.append(into / f"{i:04d}.png")
        if not cv2.imwrite(str(toned[-1]), tone(t_k)):
            raise OSError(f"cannot write {toned[-1]}")
    return toned


def _encode(run: Path, camera: str, frames: list[Image]) -> Path:
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
        paths = _toned(paths, step, Path(tmp))
        bpy.ops.wm.read_factory_settings(use_empty=True)
        sc = bpy.context.scene
        strip = sc.sequence_editor_create().strips.new_image(
            camera, str(paths[0]), 1, 1
        )
        for path in paths[1:]:
            strip.elements.append(str(path.relative_to(paths[0].parent)))
        sc.frame_end = len(paths)
        r = sc.render
        r.resolution_x, r.resolution_y = frames[0].width, frames[0].height
        r.fps, r.fps_base = 1, step
        # The factory AgX would tone the frames a second time.
        sc.view_settings.view_transform = "Standard"
        r.image_settings.media_type = "VIDEO"
        r.ffmpeg.format, r.ffmpeg.codec = "MPEG4", "H264"
        r.ffmpeg.constant_rate_factor = "PERC_LOSSLESS"
        # With its extension: without one Blender appends the frame range to the name.
        out = run / f"{camera}.mp4"
        r.filepath = str(out)
        bpy.ops.render.render(animation=True)
        return out
