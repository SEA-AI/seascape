"""The foam a build remembers: the grids its cameras see, and their frames on disk."""

import math
import shutil
from collections.abc import Callable, Iterator
from functools import partial
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

from seascape.blender.sea.foam import FOAM_SEAM_CELLS
from seascape.blender.sea.wind import surface_drift
from seascape.config import Band, Mount, Outputs, Scenario, Sea
from seascape.physics import earth, waves
from seascape.physics.waves import (
    FoamGrid,
    Wave,
    foam_memory,
    foam_memory_thresholds_g,
    whitecap_fraction,
)

# A foam memory's finest cells resolve waves a metre long, whose crests break in
# whitecaps as wide and as brief as field video measures (Korinenko & Malinovsky 2023;
# Korinenko et al. 2022).
FOAM_NEAR_CELL_M = 0.25
# Korinenko & Malinovsky 2023 filmed breakers 3.7-18.7 m long: a grid resolving no wave
# that short would lay whitecaps larger than any they measured, so the sea past it
# takes Monahan's mean.
FOAM_LONGEST_BREAKER_M = 18.7
# Numerics: how much wider each next grid's cells are, taking over where a pixel spans
# a quarter of one; rays per side of a frame its view is sampled at; a grid's widest
# side, in cells.
FOAM_LEVEL_RATIO = 4
FOAM_RAYS = 9
FOAM_MAX_CELLS = 1024


def _rays(
    scenario: Scenario, mounts: list[Mount], cameras: dict[str, bpy.types.Object]
) -> list[tuple[Vector, Vector, float]]:
    """(origin, unit direction, the camera's pixel in rad) of rays across each camera's
    frame, in every frame, on the foam memory's drifting grid."""
    sc = bpy.context.scene
    drift = surface_drift(scenario.sea, scenario.outputs)
    rays = []
    for frame, t_s in enumerate(scenario.outputs.times_s):
        sc.frame_set(frame)
        # matrix_world is stale until the depsgraph runs.
        bpy.context.view_layer.update()
        for mount in mounts:
            camera = cameras[mount.name]
            pixel_rad = camera.data.angle_x / mount.camera.width_px
            turn = camera.matrix_world.to_3x3()
            # Not `view_frame`: it takes the scene's aspect, which the build sets last.
            across = math.tan(camera.data.angle_x / 2)
            up = across * mount.camera.height_px / mount.camera.width_px
            corners = [
                Vector((x * across, y * up, -1.0))
                for x, y in ((1, 1), (1, -1), (-1, -1), (-1, 1))
            ]
            for a in np.linspace(0.0, 1.0, FOAM_RAYS):
                for b in np.linspace(0.0, 1.0, FOAM_RAYS):
                    toward = (corners[2].lerp(corners[1], a)).lerp(
                        corners[3].lerp(corners[0], a), b
                    )
                    # Water at x is on the grid at x - v t.
                    origin = camera.matrix_world.translation - Vector(
                        (drift[0] * t_s, drift[1] * t_s, 0.0)
                    )
                    rays.append((origin, (turn @ toward).normalized(), pixel_rad))
    sc.frame_set(0)
    return rays


def _box(
    rays: list[tuple[Vector, Vector, float]], cell_m: float, shrink: float
) -> tuple[float, float, float, float]:
    """(west, south, width, height) on the lattice of `cell_m` round where the rays meet
    the sea, or pass over it, out to where their camera's pixel spans `2 * shrink`
    cells, plus the grid's fading edge."""
    xs, ys = [], []
    for origin, ray, pixel_rad in rays:
        flat = Vector((ray.x, ray.y))
        run = 2 * shrink * cell_m / pixel_rad
        if ray.z < 0.0:
            run = min(run, origin.z / -ray.z * flat.length)
        flat.normalize()
        xs += [origin.x, origin.x + flat.x * run]
        ys += [origin.y, origin.y + flat.y * run]
    margin = (FOAM_SEAM_CELLS + 1) * cell_m
    west = math.floor((min(xs) - margin) / cell_m) * cell_m
    south = math.floor((min(ys) - margin) / cell_m) * cell_m
    return west, south, max(xs) + margin - west, max(ys) + margin - south


def _foam_box(
    rays: list[tuple[Vector, Vector, float]], cell_m: float
) -> tuple[float, float, float, float, float] | None:
    """`_box` at the largest shrink up to 1 that holds `FOAM_MAX_CELLS` a side, and the
    shrink; None if the cameras stand too far apart for even their origins to fit."""
    most_m = FOAM_MAX_CELLS * cell_m
    least_m = max(_box(rays, cell_m, 0.0)[2:])
    if least_m >= most_m:
        return None
    shrink = 1.0
    box = _box(rays, cell_m, shrink)
    while max(box[2:]) > most_m:
        # The excess over the origins' spread shrinks at most in proportion, so this
        # may repeat.
        shrink *= (most_m - least_m) / (max(box[2:]) - least_m)
        box = _box(rays, cell_m, shrink)
    return (*box, shrink)


def foam_grids(
    scenario: Scenario,
    band: Band,
    cameras: dict[str, bpy.types.Object],
    wind: tuple[waves.Wave, ...],
) -> tuple[waves.FoamGrid, ...]:
    """Grids over every stretch of sea the band's cameras see in any frame, each on a
    lattice of its own cells, so a cell remembers the same foam whatever else is in
    view. The finest has `FOAM_NEAR_CELL_M` cells, out to where a camera's pixel spans
    one; each next is `FOAM_LEVEL_RATIO` coarser, out to the horizon or to cells too
    coarse for any breaker. A grid too wide to hold reaches less far."""
    if not wind:
        return ()
    mounts = [m for m in scenario.mounts if m.camera.band == band]
    rays = _rays(scenario, mounts, cameras)
    horizon = earth.horizon_m(
        max(origin.z for origin, _, _ in rays), scenario.sea.refraction_k
    )
    longest_m = max(2 * math.pi / w.k_rad_m for w in wind)
    finest_rad = min(pixel_rad for _, _, pixel_rad in rays)
    grids = []
    cell_m = FOAM_NEAR_CELL_M
    while True:
        fit = _foam_box(rays, cell_m)
        reach_m = 0.0
        if fit is not None:
            west, south, width, height, shrink = fit
            columns, rows = math.ceil(width / cell_m), math.ceil(height / cell_m)
            grids.append(waves.FoamGrid(west, south, cell_m, columns, rows))
            reach_m = shrink * cell_m / finest_rad
        cell_m *= FOAM_LEVEL_RATIO
        shortest_m = waves.FOAM_CELLS_PER_WAVE * cell_m
        if reach_m >= horizon or shortest_m > min(longest_m, FOAM_LONGEST_BREAKER_M):
            return tuple(grids)


def _write_frames(
    folder: Path,
    grid: FoamGrid,
    frames: Iterator[tuple[np.ndarray, np.ndarray]],
) -> None:
    """Each frame's (cover, breaking) as an EXR in `folder`."""
    folder.mkdir(parents=True)
    image = bpy.data.images.new(
        "sea_foam_frame", grid.columns, grid.rows, float_buffer=True
    )
    image.colorspace_settings.name = "Non-Color"
    # `Image.save` writes 32-bit floats uncompressed; a scene's output settings write
    # half floats, which keep cover to a part in 2048, zipped.
    frames_scene = bpy.data.scenes.new("sea_foam_frames")
    settings = frames_scene.render.image_settings
    settings.file_format, settings.color_mode = "OPEN_EXR", "RGB"
    settings.color_depth, settings.exr_codec = "16", "ZIP"
    rgba = np.zeros((grid.rows, grid.columns, 4), np.float32)
    rgba[..., 3] = 1.0
    for i, (cover, breaking) in enumerate(frames):
        rgba[..., 0], rgba[..., 1] = cover, breaking
        image.pixels.foreach_set(rgba.ravel())
        image.save_render(str(folder / f"foam_{i + 1:04d}.exr"), scene=frames_scene)
    bpy.data.scenes.remove(frames_scene)
    bpy.data.images.remove(image)


# The foam memories the last build asked for and has not simulated: a build that never
# renders never pays for them.
_PENDING: list[Callable[[], None]] = []


def _frames_folder() -> Path:
    """Where a build writes its foam memory: Blender's session temporary folder, which
    it deletes on exit."""
    return Path(bpy.app.tempdir) / "sea_foam"


def forget_foam() -> None:
    """Drop the last build's foam memory, pending or written: every build starts here,
    with foam or without."""
    _PENDING.clear()
    shutil.rmtree(_frames_folder(), ignore_errors=True)


def _foam_sequences() -> list[bpy.types.Image]:
    return [
        image
        for image in bpy.data.images
        if image.name.startswith("sea_foam_") and image.source == "SEQUENCE"
    ]


def remember_foam(*_: object) -> None:
    """Simulate every foam memory the last build left pending, and point their image
    sequences at the frames: on `render_pre`, and by `keep_foam`."""
    while _PENDING:
        _PENDING.pop()()
    for image in _foam_sequences():
        image.reload()


def keep_foam(blend: Path) -> None:
    """Simulate the foam memory and move its frames beside `blend`, where a `.blend`
    about to be saved there finds them, as Blender keeps a simulation's bake."""
    remember_foam()
    # A rebuild without foam leaves no frames of the last one beside it.
    beside = blend.with_suffix(".foam")
    if beside.exists():
        shutil.rmtree(beside)
    frames = _frames_folder()
    if not frames.is_dir():
        return
    shutil.move(frames, beside)
    for image in _foam_sequences():
        moved = beside / Path(image.filepath).relative_to(frames)
        image.filepath = bpy.path.relpath(str(moved), start=str(blend.parent))


def foam_frames(
    sea: Sea,
    wind: tuple[Wave, ...],
    outputs: Outputs,
    grids: tuple[FoamGrid, ...],
) -> list[tuple[FoamGrid, bpy.types.Image]]:
    """Each grid's foam memory over the clip, an image sequence of (cover, breaking),
    simulated before the first render, breaking at Monahan's cover for the mean wind.

    No gust moves the threshold: whitecaps break off the drawn waves, which grow too
    slowly to follow a gust (Plant 1982), as `wind._gusty` holds them at the mean
    wind."""
    drift = surface_drift(sea, outputs)
    fraction = whitecap_fraction(sea.wind_speed_mps)
    if remember_foam not in bpy.app.handlers.render_pre:
        bpy.app.handlers.render_pre.append(remember_foam)
    out = []
    for grid in grids:
        folder = _frames_folder() / f"{grid.cell_m:g}m"
        _PENDING.append(
            partial(_simulate, wind, outputs, grid, folder, drift, fraction)
        )
        sequence = bpy.data.images.new(
            f"sea_foam_{grid.cell_m:g}m_frames", 1, 1, float_buffer=True
        )
        sequence.source = "SEQUENCE"
        sequence.filepath = str(folder / "foam_0001.exr")
        sequence.colorspace_settings.name = "Non-Color"
        out.append((grid, sequence))
    return out


def _simulate(
    wind: tuple[Wave, ...],
    outputs: Outputs,
    grid: FoamGrid,
    folder: Path,
    drift: tuple[float, float],
    fraction: float,
) -> None:
    """One grid's foam memory over the clip, written to `folder`."""
    step = 1 / outputs.fps
    (threshold,) = foam_memory_thresholds_g(wind, drift, grid.cell_m, step, (fraction,))
    times = outputs.times_s
    _write_frames(
        folder,
        grid,
        foam_memory(wind, drift, grid, times[0], step, len(times), float(threshold)),
    )
