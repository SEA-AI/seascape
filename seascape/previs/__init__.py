"""A scenario's ownship, objects and cameras on a flat sea, from above and from
behind the ownship, with each object's range ringed: a preview of its shots before
the real render.

Each view is a `Lens`: `Top` and `Chase` say how their camera frames, moves and
scales; everything else is drawn the same for both.
"""

import shutil
import tempfile
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import repeat
from pathlib import Path

import bpy
import numpy as np

from seascape import processes
from seascape.config import Scenario
from seascape.outputs.previs import LINE_PX, VIEWS, View, grown, hud, hud_lines, samples
from seascape.previs.chase import Chase
from seascape.previs.lens import Lens
from seascape.previs.scene import (
    EDGE_PX,
    RING_Z,
    WIDTH_UNIT_M,
    Scene,
    build,
    on_water,
    turn_fans,
)
from seascape.previs.top import Top
from seascape.render import video

__all__ = ["VIEWS", "Drawn", "View", "draw"]

LENSES: dict[View, type[Lens]] = {lens.view: lens for lens in (Top, Chase)}


def _sizes(scene: Scene, lens: Lens) -> float | None:
    """Size everything drawn on the water for `lens`: each line to its pixel width
    where it lies, and each object. Returns the most an object in view is grown, if
    any is.
    """
    bpy.context.view_layer.update()
    most = _grow(scene, lens)
    k, up = lens.k, lens.up()
    for radius_m, ring in scene.rings.items():
        # Where the camera looks across it, which a chase view sees at its slantest.
        at = radius_m * up
        wide_m = LINE_PX * k * lens.m_per_px(at) / lens.slant(at)
        ring.data.shape_keys.key_blocks["wide"].value = wide_m / WIDTH_UNIT_M
    for _, fans in scene.fans.values():
        for f in fans:
            for edge, far in zip(f.edges, f.far, strict=True):
                keys = edge.data.shape_keys.key_blocks
                for name, end in (("near", f.near), ("far", far)):
                    at = edge.matrix_world @ end
                    keys[name].value = EDGE_PX * k * lens.m_per_px(at) / WIDTH_UNIT_M
    return most


def _grow(scene: Scene, lens: Lens) -> float | None:
    """Scale each object but the ownship about its origin, uniformly so it keeps its
    proportions, by `grown`. Returns the largest factor among those whose origin is
    in view, if any is.
    """
    seen = []
    for root in scene.roots[1:]:
        at = root.matrix_world.translation
        grow = grown(lens.m_per_px(at), max(root["length_m"], root["beam_m"]))
        root.scale = (grow,) * 3
        p = lens.px(at)
        if p and 0.0 <= p[0] <= lens.size_px[0] and 0.0 <= p[1] <= lens.size_px[1]:
            seen.append(grow)
    return max(seen, default=None)


def _key(scene: Scene, lens: Lens, frame: int) -> None:
    """Key at `frame` what `_sizes`, `turn_fans` and `lens` set."""
    for root in scene.roots[1:]:
        root.keyframe_insert("scale", frame=frame)
    lens.key(frame)
    lines = list(scene.rings.values())
    for _, fans in scene.fans.values():
        for f in fans:
            f.fan.keyframe_insert("location", frame=frame)
            f.fan.keyframe_insert("rotation_euler", frame=frame)
            lines += f.edges
    for line in lines:
        for key in line.data.shape_keys.key_blocks[1:]:
            key.keyframe_insert("value", frame=frame)


def _shoot(folder: Path, frames: np.ndarray) -> None:
    """Render the scene's camera at `frames`, evenly spaced, in one call, to `folder`
    named by frame.
    """
    sc = bpy.context.scene
    sc.frame_start, sc.frame_end = int(frames[0]), int(frames[-1])
    sc.frame_step = int(frames[1] - frames[0]) if len(frames) > 1 else 1
    settings = sc.render.image_settings
    settings.file_format, settings.color_mode = "PNG", "RGB"
    sc.render.filepath = str(folder / "#######")
    bpy.ops.render.render(animation=True)


def _frames(
    scenario: Scenario,
    scene: Scene,
    lens: Lens,
    frames: np.ndarray,
    folder: Path,
) -> list[Path]:
    """Render `lens` at `frames` into `folder`, with the HUD; the pictures in order."""
    sc = bpy.context.scene
    m_per_px = lens.aim(scene, frames)
    sc.camera = lens.camera
    sc.render.resolution_x, sc.render.resolution_y = lens.size_px
    times = scenario.outputs.times_s
    # Where one scale holds, the compass round the outer ring, still over the run.
    compass = None
    if scene.rings and m_per_px is not None:
        centre = lens.px((0.0, 0.0, 0.0))
        compass = None if centre is None else (centre, max(scene.rings) / m_per_px)
    huds, labels = [], []
    for k in frames:
        t_s = times[int(k)]
        sc.frame_set(int(k))
        turn_fans(scene, t_s)
        pose = scene.path(t_s)
        label_deg = lens.follow(scene, t_s)
        most = _sizes(scene, lens)
        _key(scene, lens, int(k))
        huds.append(hud_lines(t_s, pose.heading_rad, pose.speed_mps, most))
        anchors = {r: lens.px(on_water(label_deg, r, RING_Z)) for r in scene.rings}
        labels.append([(xy, r) for r, xy in anchors.items() if xy is not None])
    folder.mkdir(parents=True)
    _shoot(folder, frames)
    pictures = sorted(folder.glob("*.png"))
    with ThreadPoolExecutor() as threads:
        list(
            threads.map(hud, pictures, huds, repeat(m_per_px), labels, repeat(compass))
        )
    return pictures


@dataclass(frozen=True, slots=True)
class Drawn:
    """What `draw` wrote.

    Attributes:
        stills: Each view's picture at the start, by view.
        videos: Each view's mp4, by view; empty when no frames were asked for.
    """

    stills: dict[View, Path]
    videos: dict[View, Path]


def _view(
    scenario: Scenario,
    view: View,
    frames: np.ndarray | None,
    play_fps: float,
    still: Path,
    work: Path,
) -> None:
    """Draw `view` at the start to `still` and, with `frames`, its mp4 beside it,
    rendering in `work`.
    """
    kind = LENSES[view]
    scene = build(scenario, framed=kind.framed)
    lens = kind(scene)
    [start] = _frames(scenario, scene, lens, np.array([0]), work / "still")
    # The work folder can sit on another filesystem.
    shutil.move(start, still)
    if frames is not None:
        pictures = _frames(scenario, scene, lens, frames, work / "frames")
        out = still.with_suffix(".mp4")
        # A preview's quality, small enough to upload as it plays.
        video.sequence(pictures, lens.size_px, 1.0 / play_fps, out, "HIGH")


def draw(
    scenario: Scenario,
    folder: Path,
    title: str,
    video_fps: float | None = None,
    progress: Callable[[int, int], None] = lambda *_: None,
    views: Sequence[View] = VIEWS,
) -> Drawn:
    """Draw the scenario from above and from behind its ownship, the views at once.

    Args:
        scenario: The scenario to draw.
        folder: Where each view's picture goes and, with `video_fps`, an H.264 mp4 of
            each beside it.
        title: The pictures' name before the view's.
        video_fps: Frames a second of the run in each video, as `samples` takes
            them, played in real time; None draws no video.
        progress: Called about once a second with the video frames rendered so far
            and their total, of every view.
        views: The views to draw.

    Returns:
        The pictures and videos written.

    Raises:
        RuntimeError: A view failed; the message holds the end of its log.
    """
    folder.mkdir(parents=True, exist_ok=True)
    stills = {v: folder / f"{title}-{v}.png" for v in views}
    frames, play_fps, total = None, 0.0, 0
    if video_fps is not None:
        outputs = scenario.outputs
        frames, play_fps = samples(outputs.fps, len(outputs.times_s), video_fps)
        total = len(views) * len(frames)
    with tempfile.TemporaryDirectory() as tmp:
        work = {v: Path(tmp, v) for v in views}
        logs = {v: Path(tmp, f"{v}.log") for v in views}
        jobs = {v: (scenario, v, frames, play_fps, stills[v], work[v]) for v in views}

        def tick() -> None:
            done = sum(len(list(w.glob("frames/*.png"))) for w in work.values())
            progress(done, total)

        if failed := processes.at_once(_view, jobs, logs, tick):
            tails = "\n".join(logs[v].read_text()[-2000:] for v in failed)
            raise RuntimeError(f"drawing {', '.join(failed)} failed:\n{tails}")
    videos = (
        {} if frames is None else {v: s.with_suffix(".mp4") for v, s in stills.items()}
    )
    return Drawn(stills, videos)
