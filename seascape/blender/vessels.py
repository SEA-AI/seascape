"""Hulls and objects on the sea: their poses, orbits, wakes and ride on the waves."""

import math
from collections import defaultdict
from collections.abc import Sequence
from itertools import combinations

import bpy
import numpy as np
from mathutils import Vector

from seascape.blender.blend import (
    animate,
    place,
    sine,
    yaw,
)
from seascape.blender.heat import Heat, thermal_skin, zone_skin
from seascape.blender.meshes import box_corners, import_asset, mesh_objects
from seascape.config import (
    Drift,
    Farm,
    Object,
    Outputs,
    Ownship,
    Scenario,
    Sky,
    Targets,
    substream,
)
from seascape.physics import earth, wakes, waves


def heading_rad(anchor: bpy.types.Object) -> float:
    """Where a hull's bow points, clockwise from north: bow to +Y, as fitted."""
    bow = anchor.matrix_world.to_3x3() @ Vector((0.0, 1.0, 0.0))
    return math.atan2(bow.x, bow.y)


def _vessel(
    name: str,
    t_k: float | None,
    engine_off_min: float,
    paint_absorptance: float | None,
    sky: Sky,
    heat: Heat | None,
    hulls: dict[str, list[bpy.types.Object]],
) -> bpy.types.Object:
    """A hull fitted and anchored at the origin under an empty. `heat` is the LWIR
    build's, and None in EO.

    `hulls` holds each asset's first import for the rest of the build: the FBX
    importer slows with every material already in the file.
    """
    if name in hulls:
        parts = [_copy_tree(part, None) for part in hulls[name]]
    else:
        parts = hulls[name] = import_asset(name, "eo" if heat is None else "ir")

    if heat is not None:
        heat = heat.stopped(engine_off_min)
        uniform = None if t_k is None else thermal_skin(f"{name}_ir", t_k, sky)
        for mesh in mesh_objects(parts):
            for slot, zone in zip(mesh.material_slots, mesh["zones"], strict=True):
                # Copies share mesh data, and a hull given its own t_k differs.
                slot.link = "OBJECT"
                # A rotor's frame is its hub, so its z says nothing of the sea.
                slot.material = uniform or zone_skin(
                    zone, heat, paint_absorptance, wet=not mesh.get("rotor")
                )

    anchor = bpy.data.objects.new(name, None)
    for size in ("beam_m", "length_m", "height_m"):
        anchor[size] = parts[0][size]
    anchor["engine_off_min"] = engine_off_min
    # Pitch and roll go here, so the anchor keeps the pose labels read.
    attitude = bpy.data.objects.new(f"{name}_attitude", None)
    for obj in (anchor, attitude):
        bpy.context.collection.objects.link(obj)
    attitude.parent = anchor
    for part in parts:
        part.parent = attitude
    place(anchor, 0.0, 0.0, 0.0)
    place(attitude, 0.0, 0.0, 0.0)
    # YXZ euler is Rz @ Rx @ Ry: roll about the keel, innermost.
    attitude.rotation_mode = "YXZ"
    return anchor


def add_ownship(
    ownship: Ownship,
    sky: Sky,
    heat: Heat | None,
    rigs: bpy.types.Object,
    hulls: dict[str, list[bpy.types.Object]],
    outputs: Outputs,
    tall: tuple[waves.Wave, ...],
) -> bpy.types.Object:
    """At the origin, bow to +Y, carrying the rigs: their offsets are in this frame.
    It heaves on the `tall` waves, which are geometry, and its script."""
    if ownship.asset is None:
        anchor = bpy.data.objects.new("ownship", None)
        bpy.context.collection.objects.link(anchor)
        place(anchor, 0.0, 0.0, 0.0)
    else:
        anchor = _vessel(ownship.asset, ownship.t_k, 0.0, None, sky, heat, hulls)
    anchor.name = "ownship"
    rigs.parent = anchor
    # YXZ euler is Rz @ Rx @ Ry: roll about the keel, innermost.
    anchor.rotation_mode = "YXZ"
    anchor.rotation_euler = (
        math.radians(ownship.pitch_deg),
        math.radians(ownship.roll_deg),
        0.0,
    )
    for index, mean_deg, swing in (
        (0, ownship.pitch_deg, ownship.pitch),
        (1, ownship.roll_deg, ownship.roll),
    ):
        if swing is not None:
            value_at = sine(
                math.radians(mean_deg),
                math.radians(swing.amplitude_deg),
                outputs.period_s(swing.period_s),
            )
            animate(anchor, "rotation_euler", outputs.times_s, value_at, index)
    scripted = sine(anchor.location.z, 0.0, 1.0)
    if (heave := ownship.heave) is not None:
        scripted = sine(
            anchor.location.z, heave.amplitude_m, outputs.period_s(heave.period_s)
        )
    if heave is not None or tall:
        # Bow to +Y, a heading of 0; a rig without a hull rides the water under it.
        length_m, beam_m = _waterplane(anchor) if ownship.asset else (0.0, 0.0)

        def value_at(t_s: float) -> float:
            rest = map(
                float, waves.gerstner_rest_m(tall, np.zeros(()), np.zeros(()), t_s)
            )
            return scripted(t_s) + waves.heave_m(
                tall, *rest, 0.0, length_m, beam_m, t_s
            )

        animate(anchor, "location", outputs.times_s, value_at, index=2)
    return anchor


def _pose(
    anchor: bpy.types.Object,
    range_m: float,
    bearing_deg: float,
    heading_deg: float,
    speed_mps: float,
    drift: Drift | None,
    radius_m: float,
    outputs: Outputs,
) -> None:
    """Put a hull on the sea at a bearing and range at t = 0, underway along its
    heading and drifting about that pose.

    A hull left at z = 0 flies above the curved sea.

    Not tilted to the local vertical: at any range the sea reaches, range / R moves a
    hull's ends far less than its draught.
    """
    bearing, heading = math.radians(bearing_deg), math.radians(heading_deg)
    sway = surge = sine(0.0, 0.0, 1.0)
    if drift is not None:
        period_s = outputs.period_s(drift.period_s)
        sway = sine(0.0, drift.sway_m, period_s)
        surge = sine(0.0, drift.surge_m, period_s / 2.0)

    def at(t_s: float) -> tuple[float, float, float]:
        along, across = speed_mps * t_s + surge(t_s), sway(t_s)
        # Starboard of the heading is (cos, -sin).
        east = (
            range_m * math.sin(bearing)
            + along * math.sin(heading)
            + across * math.cos(heading)
        )
        north = (
            range_m * math.cos(bearing)
            + along * math.cos(heading)
            - across * math.sin(heading)
        )
        return east, north, earth.sea_z_m(east, north, radius_m)

    place(anchor, *at(0.0))
    animate(anchor, "location", outputs.times_s, at)
    anchor.rotation_euler = (0.0, 0.0, yaw(heading_deg))


def _orbit(
    anchor: bpy.types.Object,
    range_m: float,
    bearing_deg: float,
    lap_s: float,
    radius_m: float,
    times_s: Sequence[float],
) -> None:
    def bearing_at_deg(t_s: float) -> float:
        return bearing_deg + 360.0 * t_s / lap_s

    def at(t_s: float) -> tuple[float, float, float]:
        bearing = math.radians(bearing_at_deg(t_s))
        east, north = range_m * math.sin(bearing), range_m * math.cos(bearing)
        return east, north, earth.sea_z_m(east, north, radius_m)

    place(anchor, *at(0.0))
    animate(anchor, "location", times_s, at)
    # Clockwise, the bow runs a right angle ahead of the bearing.
    animate(
        anchor,
        "rotation_euler",
        times_s,
        lambda t: yaw(bearing_at_deg(t) + 90),
        2,
    )


def _copy_tree(
    obj: bpy.types.Object, parent: bpy.types.Object | None
) -> bpy.types.Object:
    """Duplicate an object tree. `copy()` shares `data`, so N targets cost one mesh.

    The parent inverse copies too, so the tree keeps its shape rather than flattening.
    """
    clone = obj.copy()
    bpy.context.collection.objects.link(clone)
    clone.parent = parent
    for child in obj.children:
        _copy_tree(child, clone)
    return clone


def _targets(
    spec: Targets,
    radius_m: float,
    sky: Sky,
    heat: Heat | None,
    hulls: dict[str, list[bpy.types.Object]],
    outputs: Outputs,
    trails: list[wakes.Wake],
) -> dict[str, list[bpy.types.Object]]:
    """The ring's hulls, by asset."""
    anchors: dict[str, list[bpy.types.Object]] = {}
    for i, (asset, bearing_deg, heading_deg) in enumerate(spec.poses()):
        anchor = _vessel(
            asset,
            spec.t_k,
            spec.engine_off_min,
            spec.paint_absorptance,
            sky,
            heat,
            hulls,
        )
        anchor.name = f"target_{i}"
        _pose(
            anchor,
            spec.range_m,
            bearing_deg,
            heading_deg,
            spec.speed_mps,
            spec.drift,
            radius_m,
            outputs,
        )
        trails.extend(_wake(spec, anchor, bearing_deg, heading_deg))
        anchors.setdefault(asset, []).append(anchor)
    return anchors


def _object(
    spec: Object,
    radius_m: float,
    sky: Sky,
    heat: Heat | None,
    hulls: dict[str, list[bpy.types.Object]],
    outputs: Outputs,
    trails: list[wakes.Wake],
) -> list[bpy.types.Object]:
    """`spec`'s hulls, each wake under way added to `trails`."""
    anchor = _vessel(
        spec.asset,
        spec.t_k,
        spec.engine_off_min,
        spec.paint_absorptance,
        sky,
        heat,
        hulls,
    )
    orbit = spec.orbit
    if orbit is None:
        _pose(
            anchor,
            spec.range_m,
            spec.bearing_deg,
            spec.course_deg,
            spec.speed_mps,
            spec.drift,
            radius_m,
            outputs,
        )
        trails.extend(_wake(spec, anchor, spec.bearing_deg, spec.course_deg))
        return [anchor]
    anchors = [anchor, *(_copy_tree(anchor, None) for _ in range(orbit.count - 1))]
    lap_s = orbit.count * outputs.period_s(orbit.period_s / orbit.count)
    for i, hull in enumerate(anchors):
        bearing_deg = spec.bearing_deg + 360.0 * i / orbit.count
        _orbit(hull, spec.range_m, bearing_deg, lap_s, radius_m, outputs.times_s)
    trails.append(
        wakes.Wake(
            speed_mps=2 * math.pi * spec.range_m / lap_s,
            length_m=anchor["length_m"],
            beam_m=anchor["beam_m"],
            orbit_m=spec.range_m,
            start_bearing_rad=math.radians(spec.bearing_deg),
            lap_s=lap_s,
            count=orbit.count,
        )
    )
    return anchors


def _turbine(
    farm: Farm,
    range_m: float,
    bearing_deg: float,
    rng: np.random.Generator,
    scenario: Scenario,
    radius_m: float,
    heat: Heat | None,
    hulls: dict[str, list[bpy.types.Object]],
) -> bpy.types.Object:
    """A turbine facing the wind, its rotor starting at an angle of its own so a farm's
    rotors turn out of step."""
    anchor = _vessel(
        farm.asset, None, 0.0, farm.paint_absorptance, scenario.sky, heat, hulls
    )
    outputs = scenario.outputs
    _pose(
        anchor,
        range_m,
        bearing_deg,
        scenario.sea.wind_from_deg,
        0.0,
        None,
        radius_m,
        outputs,
    )
    rpm = farm.rpm(scenario.sea)
    # A loop closes on a third of a turn.
    third_s = outputs.period_s(20.0 / rpm) if rpm else math.inf
    start = rng.uniform(0.0, 2.0 * math.pi)
    (rotor,) = (o for o in anchor.children_recursive if o.get("rotor"))
    rest = rotor["rest_rad"]
    # A copy shares its original's action.
    rotor.animation_data_clear()
    # Clockwise seen from upwind, which its +Y faces.
    animate(
        rotor,
        "rotation_euler",
        outputs.times_s,
        lambda t: rest - start - 2.0 * math.pi / 3.0 * t / third_s,
        index=1,
    )
    return anchor


def _wake(
    spec: Object | Targets,
    anchor: bpy.types.Object,
    bearing_deg: float,
    heading_deg: float,
) -> list[wakes.Wake]:
    """The wake `_pose` leaves, if the hull is under way.

    Limitation: along the heading from the start, without the drift's figure-eight; give
    the wake the drift's offset if a drifting hull under way shows its wake beside it.
    """
    if spec.speed_mps <= 0.0:
        return []
    bearing = math.radians(bearing_deg)
    return [
        wakes.Wake(
            speed_mps=spec.speed_mps,
            length_m=anchor["length_m"],
            beam_m=anchor["beam_m"],
            start_m=(
                spec.range_m * math.sin(bearing),
                spec.range_m * math.cos(bearing),
            ),
            heading_rad=math.radians(heading_deg),
        )
    ]


def _waterplane(anchor: bpy.types.Object) -> tuple[float, float]:
    """A hull's length and beam, in its own frame, bow to +Y."""
    # The parts' matrix_world is stale until the depsgraph runs.
    bpy.context.view_layer.update()
    corners = box_corners(mesh_objects([anchor]), anchor.matrix_world.inverted())
    xs, ys = [c.x for c in corners], [c.y for c in corners]
    return max(ys) - min(ys), max(xs) - min(xs)


def ride(
    anchors: Sequence[bpy.types.Object],
    field: tuple[waves.Wave, ...],
    tall: tuple[waves.Wave, ...],
    outputs: Outputs,
) -> None:
    """Key each hull's pitch and roll to the sea under its keyed pose, and its heave
    to the `tall` waves, which are geometry: the rest are shading, which a heaving
    hull would break its waterline on."""
    sc = bpy.context.scene
    hulls = []
    for anchor in anchors:
        (attitude,) = anchor.children
        hulls.append((anchor, attitude, *_waterplane(anchor)))
    keyed = len(outputs.times_s) > 1
    for frame, t_s in enumerate(outputs.times_s):
        sc.frame_set(frame)
        for anchor, attitude, length_m, beam_m in hulls:
            east, north, _ = anchor.matrix_world.translation
            heading = heading_rad(anchor)
            # The tall waves are the geometry the hull floats on; the rest only shade.
            rest = [
                float(r)
                for r in waves.gerstner_rest_m(
                    tall, np.array(east), np.array(north), t_s
                )
            ]
            pitch, roll = waves.attitude(field, *rest, heading, length_m, beam_m, t_s)
            # +x turns the bow up; +y turns starboard down.
            attitude.rotation_euler = (pitch, -roll, 0.0)
            attitude.location.z = waves.heave_m(
                tall, *rest, heading, length_m, beam_m, t_s
            )
            if keyed:
                attitude.keyframe_insert("rotation_euler", index=0, frame=frame)
                attitude.keyframe_insert("rotation_euler", index=1, frame=frame)
                attitude.keyframe_insert("location", index=2, frame=frame)


class OverlapError(ValueError):
    """Two targets' hulls meet."""


def overlaps(anchors: list[bpy.types.Object], frames: int) -> list[tuple[str, str]]:
    """Pairs of hulls whose footprints meet in any frame: each a circle of half its
    larger horizontal extent about its anchor, where the frame has moved it."""
    radius_m = {a.name: max(a["length_m"], a["beam_m"]) / 2 for a in anchors}
    sc = bpy.context.scene
    found = set()
    for frame in range(frames):
        sc.frame_set(frame)
        for a, b in combinations(anchors, 2):
            apart_m = (a.matrix_world.translation - b.matrix_world.translation).xy
            if apart_m.length < radius_m[a.name] + radius_m[b.name]:
                found.add((a.name, b.name))
    return sorted(found)


def add_vessels(
    scenario: Scenario,
    radius_m: float,
    heat: Heat | None,
    hulls: dict[str, list[bpy.types.Object]],
    trails: list[wakes.Wake],
) -> tuple[dict[str, list[bpy.types.Object]], dict[str, list[bpy.types.Object]]]:
    """Every object and ring hull, then every turbine, by asset; each wake under way
    added to `trails`."""
    sky, outputs = scenario.sky, scenario.outputs
    vessels: defaultdict[str, list[bpy.types.Object]] = defaultdict(list)
    for spec in scenario.objects:
        vessels[spec.asset] += _object(
            spec, radius_m, sky, heat, hulls, outputs, trails
        )
    if scenario.targets is not None:
        ring = _targets(scenario.targets, radius_m, sky, heat, hulls, outputs, trails)
        for asset, anchors in ring.items():
            vessels[asset] += anchors
    turbines: defaultdict[str, list[bpy.types.Object]] = defaultdict(list)
    for f, farm in enumerate(scenario.farms):
        for k, (range_m, bearing_deg) in enumerate(farm.positions()):
            rng = substream(scenario.seed, f"farms/{f}/{k}/rotor")
            turbines[farm.asset].append(
                _turbine(
                    farm, range_m, bearing_deg, rng, scenario, radius_m, heat, hulls
                )
            )
    return vessels, turbines


def waterline_m(anchor: bpy.types.Object) -> np.ndarray:
    """Where a hull's edges cross its waterline, as world east and north, (N, 2).

    The waterline is z = 0 in the attitude's frame, which heaves, pitches and rolls
    with the waves: `_fit` puts the keel a draught below it.
    """
    # Limitation: the crossings, not the segments between them, so the nearest one to a
    # camera at range r is up to side^2 / 2r further than a side facing it.
    (attitude,) = anchor.children
    to_world = np.array(attitude.matrix_world)
    to_hull = np.linalg.inv(to_world)
    crossings = []
    for part in mesh_objects([anchor]):
        mesh = part.data
        local = np.empty(3 * len(mesh.vertices), dtype=np.float32)
        mesh.vertices.foreach_get("co", local)
        ends = np.empty(2 * len(mesh.edges), dtype=np.int32)
        mesh.edges.foreach_get("vertices", ends)
        m = to_hull @ np.array(part.matrix_world)
        hull = local.reshape(-1, 3) @ m[:3, :3].T + m[:3, 3]
        a, b = hull[ends.reshape(-1, 2)].transpose(1, 0, 2)
        cross = a[:, 2] * b[:, 2] < 0
        a, b = a[cross], b[cross]
        t = -a[:, 2] / (b[:, 2] - a[:, 2])
        at = a + t[:, None] * (b - a)
        crossings.append((at @ to_world[:3, :3].T + to_world[:3, 3])[:, :2])
    return np.concatenate(crossings)
