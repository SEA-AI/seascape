"""Build a Blender scene from a scenario. Nothing here renders."""

import math
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import chain

import bpy
import numpy as np
from mathutils import Matrix

from seascape.blender import sea
from seascape.blender.compositor import respond, set_output
from seascape.blender.heat import surroundings
from seascape.blender.rigs import View, add_rigs, stabilise, views
from seascape.blender.sky import add_haze, add_world, sea_light
from seascape.blender.vessels import (
    Afloat,
    Hulls,
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
from seascape.motion import attitude_deg, track
from seascape.outputs.calibration import CameraCalibration, Matrix4
from seascape.physics import earth, wakes

# Blender's camera looks down -Z with +Y up; OpenCV's looks down +Z with +Y down.
_BLENDER_TO_CV = Matrix.Diagonal((1.0, -1.0, -1.0, 1.0))


@dataclass(frozen=True)
class Built:
    """What `build` made, by role, valid until the next build. A name in `bpy.data`
    belongs to whoever took it first, and an imported asset can carry any name.
    """

    vessel: bpy.types.Object
    deck: (
        bpy.types.Object
    )  # what the rigs are bolted to: a hull's draught lift, which rides the sea
    rigs: dict[str, bpy.types.Object]
    cameras: dict[str, bpy.types.Object]
    targets: dict[str, list[bpy.types.Object]]  # by asset


def calibrate(built: Built, mount: Mount, image: str) -> CameraCalibration:
    """A built camera's geometry, read off the scene."""
    camera = built.cameras[mount.name]
    world = camera.matrix_world @ _BLENDER_TO_CV
    vessel = built.deck.matrix_world
    rig = built.rigs[mount.rig_name].matrix_world
    width, height = mount.camera.width_px, mount.camera.height_px
    if mount.camera.lens.fisheye is None:
        f = (width / 2) / math.tan(camera.data.angle_x / 2)
    else:
        f = mount.camera.fisheye_focal_px
    return CameraCalibration(
        name=mount.name,
        band=mount.camera.band,
        image=image,
        rig=mount.rig_name,
        model=mount.rig.model,
        pitch_um=mount.camera.pitch_um,
        width_px=width,
        height_px=height,
        # Blender's frame spans pixel edges, so its centre is half a pixel past
        # OpenCV's.
        K=((f, 0.0, (width - 1) / 2), (0.0, f, (height - 1) / 2), (0.0, 0.0, 1.0)),
        distortion=mount.camera.lens.fisheye,
        extrinsics={
            "world": _rows(world),
            "vessel": _rows(vessel.inverted() @ world),
            "rig": _rows(rig.inverted() @ world),
        },
    )


def look_through(
    cameras: dict[str, bpy.types.Object], mount: Mount, rng: np.random.Generator
) -> None:
    """Render through `mount`'s camera at its size, its sensor noise from `rng`."""
    sc = bpy.context.scene
    sc.camera = cameras[mount.name]
    sc.render.resolution_x = mount.camera.width_px
    sc.render.resolution_y = mount.camera.height_px
    respond(mount, rng)


def _rows(m: Matrix) -> Matrix4:
    x, y, z, w = map(tuple, m)
    return x, y, z, w


def _viewport(far_m: float) -> None:
    """Blender's view clip ends at 1000 m, short of a sea reaching tens of km."""
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type != "VIEW_3D":
                continue
            for space in area.spaces:
                if space.type == "VIEW_3D":
                    space.clip_end = far_m


def _sightline_m(scenario: Scenario, anchors: Iterable[bpy.types.Object]) -> float:
    """Past the camera's horizon plus the tallest hull's own, every hull is below the
    horizon.
    """
    # Stored at import, blade up: measured now, a rotor's top follows its spin.
    tops_m = [anchor["height_m"] for anchor in anchors]
    if not tops_m:
        return 0.0
    heave = scenario.ownship.heave
    eye_m = (
        max(rig.height_m for rig in scenario.rigs.values())
        + scenario.ownship.rise_m
        + (heave.amplitude_m if heave is not None else 0.0)
    )
    k = scenario.sea.refraction_k
    return earth.horizon_m(eye_m, k) + earth.horizon_m(max(tops_m), k)


def _finest_footprint_m(scenario: Scenario, seen: list[View]) -> float:
    """The finest pixel any camera lays on the sea in any frame: its height over the
    sea below it, which the curved sea's crests come no nearer than, times the angle
    of a pixel.

    Args:
        scenario: Its crests and refraction set how near the sea comes.
        seen: Every camera's pose in every frame.
    """
    radius_m = earth.earth_radius_m(scenario.sea.refraction_k)
    tall, _ = sea.displaced((scenario.wind_waves(), scenario.swell_waves()))
    crest_m = sum(w.amplitude_m for w in tall)
    finest = math.inf
    for view in seen:
        east, north, up = view.matrix.translation
        height_m = max(up - earth.sea_z_m(east, north, radius_m) - crest_m, 0.0)
        finest = min(finest, height_m * view.pixel_rad)
    return finest


def build(scenario: Scenario, band: Band = "eo", cull: bool = False) -> Built:
    """Replace the current Blender session's contents with a scenario.

    Args:
        scenario: The scene to build, and the times it is keyed at.
        band: `eo` for visible light, `ir` for 8-14 um thermal; only that band's
            cameras are built, and some rig must carry one.
        cull: Draw only the waves the scenario's own cameras resolve, at their own
            resolution: a camera zoomed, enlarged or added later loses its finest
            ripples.
    """
    mounts = scenario.band_mounts(band)
    if not mounts:
        raise ValueError(f"no rig has an {band} camera to build an {band} scene for")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sea.forget_foam()
    outputs = scenario.outputs
    set_output(
        outputs,
        band,
        max(max(m.camera.width_px, m.camera.height_px) for m in mounts),
        any(m.camera.sensor.full_well_e is not None for m in mounts),
    )
    bpy.context.scene.world = add_world(scenario.sky, band)
    reach_m = sea.sea_reach_m(scenario)
    far_m = 1.5 * reach_m  # the sea's corner is reach * sqrt(2) from the origin
    wind, swell = scenario.wind_waves(), scenario.swell_waves()
    rigs = add_rigs(scenario)
    heat = surroundings(scenario) if band == "ir" else None
    hulls = Hulls(scenario.sky, heat, scenario.worn)
    tall, _ = sea.displaced((wind, swell))
    radius_m = earth.earth_radius_m(scenario.sea.refraction_k)
    afloat: list[Afloat] = []
    path = track(scenario.ownship, outputs, scenario.turn)
    vessel = add_ownship(
        scenario.ownship, path, rigs.root, hulls, outputs, tall, radius_m, afloat
    )
    trails: list[wakes.Wake] = []
    vessels, turbines = add_vessels(scenario, radius_m, hulls, trails, afloat)
    targets = {**vessels, **turbines}  # no asset is both
    # Before `views`, which reads where the cameras look as the hulls rock.
    # Limitation: a floating turbine stands still; give it its platform's motion
    # if a sea rocks one visibly.
    ride(afloat, wind + swell, tall, outputs)
    if any(rig.ptu for rig in scenario.rigs.values()):
        hull = next((a.hull for a in afloat if a.anchor is vessel), None)
        attitude = attitude_deg(
            scenario.ownship, path, hull, wind + swell, tall, outputs
        )
        stabilise(rigs.rigs, scenario, attitude)
    solid = [a.anchor for a in afloat] + list(chain(*turbines.values()))
    if meeting := overlaps(solid, len(outputs.times_s)):
        raise OverlapError(f"hulls inside each other: {meeting}")
    far_m = max(far_m, _sightline_m(scenario, chain(*targets.values())))
    for camera in rigs.cameras.values():
        camera.data.clip_end = far_m
    seen = views(scenario, band, rigs.cameras)
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
            _finest_footprint_m(scenario, seen) if cull else 0.0,
            sea.foam_grids(scenario, seen, wind),
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
    first = mounts[0]
    look_through(rigs.cameras, first, substream(scenario.seed, f"sensor/{first.name}"))
    _viewport(far_m)
    # After the last import, which sets fps and fps_base to the file's own. The
    # factory scene starts at frame 1.
    sc = bpy.context.scene
    sc.frame_start = sc.frame_current = 0
    sc.frame_end = len(outputs.times_s) - 1
    sc.render.fps, sc.render.fps_base = outputs.fps, 1.0
    sc.frame_set(0)
    return Built(vessel, rigs.root.parent, rigs.rigs, rigs.cameras, targets)
