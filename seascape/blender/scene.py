"""Build a Blender scene from a scenario. Nothing here renders."""

import math
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import chain

import bpy
from mathutils import Matrix

from seascape.blender import sea
from seascape.blender.compositor import respond, set_output
from seascape.blender.heat import surroundings
from seascape.blender.meshes import box_corners, mesh_objects
from seascape.blender.rigs import add_rigs
from seascape.blender.sky import add_haze, add_world, sea_light
from seascape.blender.vessels import (
    OverlapError,
    add_ownship,
    add_vessels,
    overlaps,
    ride,
)
from seascape.config import (
    Band,
    Mount,
    Scenario,
    substream,
)
from seascape.outputs.calibration import CameraCalibration, Matrix4
from seascape.physics import earth, wakes

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
        pitch_um=mount.camera.microbolometer.pitch_um,
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
        max(c.z for c in box_corners(mesh_objects([anchor])))
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
    return earth.horizon_m(eye_m, k) + earth.horizon_m(max(tops_m), k)


def _finest_footprint_m(
    scenario: Scenario, band: Band, cameras: dict[str, bpy.types.Object]
) -> float:
    """The finest pixel any of the band's cameras lays on the sea in any frame: its
    height over the sea below it, which the curved sea's crests come no nearer than,
    times the angle of a pixel."""
    sc = bpy.context.scene
    radius_m = earth.earth_radius_m(scenario.sea.refraction_k)
    tall, _ = sea.displaced((scenario.wind_waves(), scenario.swell_waves()))
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
            height_m = max(up - earth.sea_z_m(east, north, radius_m) - crest_m, 0.0)
            pixel_rad = camera.data.angle_x / mount.camera.width_px
            finest = min(finest, height_m * pixel_rad)
    sc.frame_set(0)
    return finest


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
    set_output(scenario.outputs, band, max(sides), graded)
    bpy.context.scene.world = add_world(scenario.sky, band)
    reach_m = sea.sea_reach_m(scenario)
    far_m = 1.5 * reach_m  # the sea's corner is reach * sqrt(2) away
    outputs = scenario.outputs
    wind, swell = scenario.wind_waves(), scenario.swell_waves()
    rigs = add_rigs(scenario, far_m)
    hulls: dict[str, list[bpy.types.Object]] = {}
    tall, _ = sea.displaced((wind, swell))
    heat = surroundings(scenario) if band == "ir" else None
    vessel = add_ownship(
        scenario.ownship, scenario.sky, heat, rigs.root, hulls, outputs, tall
    )
    radius_m = earth.earth_radius_m(scenario.sea.refraction_k)
    trails: list[wakes.Wake] = []
    vessels, turbines = add_vessels(scenario, radius_m, heat, hulls, trails)
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
        sea_light(scenario.sky, band),
        sea.Seen(
            _finest_footprint_m(scenario, band, rigs.cameras) if cull else 0.0,
            sea.foam_grids(scenario, band, rigs.cameras, wind),
        ),
    )
    sea.water(scenario.sea, reach_m, material)
    # After the last material.
    add_haze(scenario.sky, band, far_m)
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
    # Limitation: a floating turbine stands still; give it its platform's motion
    # if a sea rocks one visibly.
    ride(list(chain(*vessels.values())), wind + swell, tall, outputs)
    solid = list(chain(*targets.values()))
    if scenario.ownship.asset is not None:
        solid.append(vessel)
    if meeting := overlaps(solid, len(outputs.times_s)):
        raise OverlapError(f"hulls inside each other: {meeting}")
    sc.frame_set(0)
    # Until the depsgraph runs, every child still reports its pre-parenting
    # matrix_world, so anything measuring the scene reads the wrong place.
    bpy.context.view_layer.update()
    return Built(vessel, rigs.rigs, rigs.cameras, targets)
