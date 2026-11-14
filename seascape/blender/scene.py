"""Build a Blender scene from a scenario. Nothing here renders."""

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import chain

import bpy
import numpy as np

from seascape.blender import sea
from seascape.blender.blend import empty
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
from seascape.motion import (
    Path,
    at_sea_level,
    lifted,
    ownship_up_m,
    tilted,
    track,
    waterline_pose,
)
from seascape.outputs.calibration import CameraCalibration, intrinsics, matrix4
from seascape.physics import earth, wakes


@dataclass(frozen=True, slots=True, kw_only=True)
class Built:
    """What `build` made, by role, valid until the next build. A name in `bpy.data`
    belongs to whoever took it first, and an imported asset can carry any name.
    """

    # The hull's root, named ownship in Blender: `motion.waterline_pose`, heave
    # included.
    root: bpy.types.Object
    ownship: bpy.types.Object  # the rigs' parent: a hull's draught, or the root
    heads: dict[str, bpy.types.Object]
    cameras: dict[str, bpy.types.Object]
    targets: dict[str, list[bpy.types.Object]]  # by asset
    # What Blender was keyed from, at a frame's time: `T_world_ownship`, and each
    # rig's `T_ownship_head` by name.
    ownship_pose: Callable[[float], np.ndarray]
    head_poses: dict[str, Callable[[float], np.ndarray]]


def calibrate(built: Built, mount: Mount, image: str, t_s: float) -> CameraCalibration:
    """A built camera's geometry at `t_s`, a frame's time."""
    T_head_camera = mount.camera.pose()
    T_ownship_camera = built.head_poses[mount.rig_name](t_s) @ T_head_camera
    T_world_camera = built.ownship_pose(t_s) @ T_ownship_camera
    width, height = mount.camera.width_px, mount.camera.height_px
    return CameraCalibration(
        name=mount.name,
        band=mount.camera.band,
        image=image,
        rig=mount.rig_name,
        model=mount.rig.model,
        pitch_um=mount.camera.pitch_um,
        width_px=width,
        height_px=height,
        K=intrinsics(mount.camera),
        distortion=mount.camera.lens.fisheye,
        extrinsics={
            "world": matrix4(T_world_camera),
            "ownship": matrix4(T_ownship_camera),
            "head": matrix4(T_head_camera),
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


def _viewport(far_m: float) -> None:
    """Blender's view clip ends at 1000 m, short of a sea reaching tens of km."""
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type != "VIEW_3D":
                continue
            for space in area.spaces:
                if space.type == "VIEW_3D":
                    space.clip_end = far_m


def _sightline_m(scenario: Scenario, roots: Iterable[bpy.types.Object]) -> float:
    """Past the camera's horizon plus the tallest hull's own, every hull is below the
    horizon.
    """
    # Stored at import, blade up: measured now, a rotor's top follows its spin.
    tops_m = [root["height_m"] for root in roots]
    if not tops_m:
        return 0.0
    heave = scenario.ownship.heave
    eye_m = max(map(scenario.head_height_m, scenario.rigs)) + (
        heave.amplitude_m if heave is not None else 0.0
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


def ownship_afloat(scenario: Scenario, path: Path) -> Callable[[float], np.ndarray]:
    """`T_world_waterline` at a time: the ownship's hull, measured from its imported
    mesh, riding the sea along `path`; no rigs, no targets. Its height is sea level
    plus the sea's heave, without the scripted heave or the earth's curve.
    """
    bpy.ops.wm.read_factory_settings(use_empty=True)
    afloat: list[Afloat] = []
    up_m = ownship_up_m(
        scenario.ownship,
        scenario.outputs,
        (),
        earth.earth_radius_m(scenario.sea.refraction_k),
    )
    vessel = add_ownship(
        scenario.ownship,
        path,
        up_m,
        empty("rigs"),
        Hulls(scenario.sky, None),
        scenario.outputs.times_s,
        afloat,
    )
    wind, swell = scenario.wind_waves(), scenario.swell_waves()
    tall, _ = sea.displaced((wind, swell))
    hull = next((a.hull for a in afloat if a.root is vessel), None)
    return waterline_pose(path, at_sea_level, hull, wind + swell, tall)


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
    up_m = ownship_up_m(scenario.ownship, outputs, tall, radius_m)
    vessel = add_ownship(
        scenario.ownship, path, up_m, rigs.root, hulls, outputs.times_s, afloat
    )
    trails: list[wakes.Wake] = []
    vessels, turbines = add_vessels(scenario, radius_m, hulls, trails, afloat)
    targets = {**vessels, **turbines}  # no asset is both
    # Before `views`, which reads where the cameras look as the hulls rock.
    # Limitation: a floating turbine stands still; give it its platform's motion
    # if a sea rocks one visibly.
    ride(afloat, wind + swell, tall, outputs)
    hull = next((a.hull for a in afloat if a.root is vessel), None)
    tilt = tilted(path, hull, wind + swell, tall)
    head_poses = stabilise(rigs.heads, scenario, tilt)
    T_waterline_ownship = lifted(scenario.ownship.rise_m)
    waterline = waterline_pose(path, up_m, hull, wind + swell, tall)
    solid = [a.root for a in afloat] + list(chain(*turbines.values()))
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
    roots = [*chain(*targets.values()), vessel]
    for index, root in enumerate(roots, start=1):
        for part in [root, *root.children_recursive]:
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
    return Built(
        root=vessel,
        ownship=rigs.root.parent,
        heads=rigs.heads,
        cameras=rigs.cameras,
        targets=targets,
        ownship_pose=lambda t_s: waterline(t_s) @ T_waterline_ownship,
        head_poses=head_poses,
    )
