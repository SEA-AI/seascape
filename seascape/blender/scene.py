"""Build a Blender scene from a scenario. Nothing here renders."""

import math
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import chain

import bpy
import numpy as np
from mathutils import Matrix, Vector

from seascape.blender import sea
from seascape.blender.compositor import _output, respond
from seascape.blender.heat import _heat
from seascape.blender.meshes import _corners, _meshes
from seascape.blender.rigs import _rigs
from seascape.blender.sky import _haze, _light, _sky
from seascape.blender.vessels import OverlapError, _overlaps, _ownship, _placed, _ride
from seascape.calibration import CameraCalibration, Matrix4
from seascape.config import (
    Band,
    Mount,
    Scenario,
    substream,
)
from seascape.physics import wakes, waves

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


def wind_waves(scenario: Scenario) -> tuple[waves.Wave, ...]:
    sea = scenario.sea
    return waves.wind_sea(
        sea.wind_speed_mps,
        sea.wind_from_deg,
        sea.fetch_m,
        scenario.outputs.period_s,
        substream(scenario.seed, "sea/surface"),
    )


def swell_waves(scenario: Scenario) -> tuple[waves.Wave, ...]:
    swell = scenario.sea.swell
    if swell is None:
        return ()
    return waves.swell(
        swell.height_m,
        swell.period_s,
        swell.from_deg,
        scenario.outputs.period_s,
        substream(scenario.seed, "sea/swell"),
    )


def wave_field(scenario: Scenario) -> tuple[waves.Wave, ...]:
    return wind_waves(scenario) + swell_waves(scenario)


# Blender's camera looks down -Z with +Y up; OpenCV's looks down +Z with +Y down.
_BLENDER_TO_CV = Matrix.Diagonal((1.0, -1.0, -1.0, 1.0))


@dataclass(frozen=True)
class Built:
    """What `build` made, by role, valid until the next build. A name in `bpy.data`
    belongs to whoever took it first, and an imported asset can carry any name."""

    vessel: bpy.types.Object
    rigs: dict[str, bpy.types.Object]
    cameras: dict[str, bpy.types.Object]
    targets: dict[str, list[bpy.types.Object]]  # by asset


def calibrate(built: Built, mount: Mount, image: str) -> CameraCalibration:
    """A built camera's geometry, read off the scene."""
    camera = built.cameras[mount.name]
    world = camera.matrix_world @ _BLENDER_TO_CV
    vessel = built.vessel.matrix_world
    rig = built.rigs[mount.rig_name].matrix_world
    width, height = mount.camera.width_px, mount.camera.height_px
    f = (width / 2) / math.tan(camera.data.angle_x / 2)
    return CameraCalibration(
        name=mount.name,
        band=mount.camera.band,
        image=image,
        rig=mount.rig_name,
        model=mount.rig.model,
        width_px=width,
        height_px=height,
        # Blender's frame spans pixel edges, so its centre is half a pixel past
        # OpenCV's.
        K=((f, 0.0, (width - 1) / 2), (0.0, f, (height - 1) / 2), (0.0, 0.0, 1.0)),
        extrinsics={
            "world": _rows(world),
            "vessel": _rows(vessel.inverted() @ world),
            "rig": _rows(rig.inverted() @ world),
        },
    )


def _rows(m: Matrix) -> Matrix4:
    x, y, z, w = map(tuple, m)
    return x, y, z, w


def _viewport(near_m: float, far_m: float) -> None:
    """Blender's view clip defaults to 0.01-1000 m, which cuts a sea reaching tens
    of km."""
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type != "VIEW_3D":
                continue
            for space in area.spaces:
                if space.type == "VIEW_3D":
                    space.clip_start, space.clip_end = near_m, far_m


def _sightline_m(scenario: Scenario, anchors: Iterable[bpy.types.Object]) -> float:
    """Past the camera's horizon plus the tallest hull's own, every hull is below the
    horizon."""
    # The parts' matrix_world is stale until the depsgraph runs.
    bpy.context.view_layer.update()
    tops_m = [
        max(c.z for c in _corners(_meshes([anchor])))
        - anchor.matrix_world.translation.z
        for anchor in anchors
    ]
    if not tops_m:
        return 0.0
    heave = scenario.ownship.heave
    eye_m = max(rig.height_m for rig in scenario.rigs.values()) + (
        heave.amplitude_m if heave is not None else 0.0
    )
    k = scenario.sea.refraction_k
    return waves.horizon_m(eye_m, k) + waves.horizon_m(max(tops_m), k)


def _finest_footprint_m(
    scenario: Scenario, band: Band, cameras: dict[str, bpy.types.Object]
) -> float:
    """The finest pixel any of the band's cameras lays on the sea in any frame: its
    height over the sea below it, which the curved sea's crests come no nearer than,
    times the angle of a pixel."""
    sc = bpy.context.scene
    radius_m = waves.earth_radius_m(scenario.sea.refraction_k)
    tall, _ = sea.displaced((wind_waves(scenario), swell_waves(scenario)))
    crest_m = sum(w.amplitude_m for w in tall)
    finest = math.inf
    for frame in range(len(scenario.outputs.times_s)):
        sc.frame_set(frame)
        # matrix_world is stale until the depsgraph runs.
        bpy.context.view_layer.update()
        for mount in scenario.mounts:
            if mount.camera.band != band:
                continue
            camera = cameras[mount.name]
            east, north, up = camera.matrix_world.translation
            height_m = max(up - waves.sea_z_m(east, north, radius_m) - crest_m, 0.0)
            pixel_rad = camera.data.angle_x / mount.camera.width_px
            finest = min(finest, height_m * pixel_rad)
    sc.frame_set(0)
    return finest


def _rays(
    scenario: Scenario, mounts: list[Mount], cameras: dict[str, bpy.types.Object]
) -> list[tuple[Vector, Vector, float]]:
    """(origin, unit direction, the camera's pixel in rad) of rays across each camera's
    frame, in every frame, on the foam memory's drifting grid."""
    sc = bpy.context.scene
    drift = sea.surface_drift(scenario.sea, scenario.outputs)
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
    margin = (sea.FOAM_SEAM_CELLS + 1) * cell_m
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


def _foam_grids(
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
    horizon = waves.horizon_m(
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


def build(scenario: Scenario, band: Band = "eo", cull: bool = False) -> Built:
    """Replace the current Blender session's contents with `scenario` in one band.

    `cull` draws only the waves the scenario's own cameras resolve, at their own
    resolution: a camera zoomed, enlarged or added later loses its finest ripples."""
    if not any(mount.camera.band == band for mount in scenario.mounts):
        raise ValueError(f"no rig has an {band} camera to build an {band} scene for")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sea.forget_foam()
    sides = [
        max(m.camera.width_px, m.camera.height_px)
        for m in scenario.mounts
        if m.camera.band == band
    ]
    graded = any(
        m.camera.sensor.full_well_e is not None
        for m in scenario.mounts
        if m.camera.band == band
    )
    _output(scenario.outputs, band, max(sides), graded)
    bpy.context.scene.world = _sky(scenario.sky, band)
    reach_m = sea.sea_reach_m(scenario)
    far_m = 1.5 * reach_m  # the sea's corner is reach * sqrt(2) away
    outputs = scenario.outputs
    wind, swell = wind_waves(scenario), swell_waves(scenario)
    rigs = _rigs(scenario, far_m)
    hulls: dict[str, list[bpy.types.Object]] = {}
    tall, _ = sea.displaced((wind, swell))
    heat = _heat(scenario) if band == "ir" else None
    vessel = _ownship(
        scenario.ownship, scenario.sky, heat, rigs.root, hulls, outputs, tall
    )
    radius_m = waves.earth_radius_m(scenario.sea.refraction_k)
    trails: list[wakes.Wake] = []
    vessels, turbines = _placed(scenario, radius_m, heat, hulls, trails)
    targets = {**vessels, **turbines}  # no asset is both
    far_m = max(far_m, _sightline_m(scenario, chain(*targets.values())))
    for camera in rigs.cameras.values():
        camera.data.clip_end = far_m
    # After the hulls, whose poses and beams set the wakes.
    material = sea.material(
        scenario.sea,
        wind,
        swell,
        outputs,
        (
            substream(scenario.seed, "sea/gust"),
            substream(scenario.seed, "sea/slick"),
            substream(scenario.seed, "sea/foam"),
        ),
        tuple(trails),
        _light(scenario.sky, band),
        sea.Seen(
            _finest_footprint_m(scenario, band, rigs.cameras) if cull else 0.0,
            _foam_grids(scenario, band, rigs.cameras, wind),
        ),
    )
    sea.water(scenario.sea, reach_m, material)
    # After the last material.
    _haze(scenario.sky, band, far_m)
    # The object-index pass reads 0 for sky and sea. The ownship takes the index after
    # the targets: no box, and no target's background.
    anchors = [*chain(*targets.values()), vessel]
    for index, anchor in enumerate(anchors, start=1):
        for part in [anchor, *anchor.children_recursive]:
            part.pass_index = index
    # The first camera may belong to the other band, whose optics this build lacks.
    first = next(mount for mount in scenario.mounts if mount.camera.band == band)
    sc = bpy.context.scene
    sc.camera = rigs.cameras[first.name]
    sc.render.resolution_x, sc.render.resolution_y = (
        first.camera.width_px,
        first.camera.height_px,
    )
    respond(first, substream(scenario.seed, f"sensor/{first.name}"))
    _viewport(min(rig.near_clip_m for rig in scenario.rigs.values()), far_m)
    # After the last import, which sets fps and fps_base to the file's own. The
    # factory scene starts at frame 1.
    sc.frame_start = sc.frame_current = 0
    sc.frame_end = len(outputs.times_s) - 1
    sc.render.fps, sc.render.fps_base = outputs.fps, 1.0
    # ponytail: a floating turbine stands still; give it its platform's motion
    # if a sea rocks one visibly.
    _ride(list(chain(*vessels.values())), wind + swell, tall, outputs)
    solid = list(chain(*targets.values()))
    if scenario.ownship.asset is not None:
        solid.append(vessel)
    if meeting := _overlaps(solid, len(outputs.times_s)):
        raise OverlapError(f"hulls inside each other: {meeting}")
    sc.frame_set(0)
    # Until the depsgraph runs, every child still reports its pre-parenting
    # matrix_world, so anything measuring the scene reads the wrong place.
    bpy.context.view_layer.update()
    return Built(vessel, rigs.rigs, rigs.cameras, targets)
