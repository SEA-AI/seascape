"""Hulls and objects on the sea: their poses, orbits, wakes and ride on the waves."""

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import chain, combinations

import bpy
import numpy as np
from mathutils import Vector

from seascape.assets import manifest
from seascape.blender import faces
from seascape.blender.blend import (
    animate,
    place,
    sine,
    yaw,
)
from seascape.blender.heat import Heat, thermal_skin, zone_skin
from seascape.blender.meshes import import_asset, mesh_objects, prefix_names
from seascape.blender.wear import CREVICE, dirty, mark
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


@dataclass
class Hulls:
    """Each asset's first import, copied for the rest of the build: the FBX importer
    slows with every material already in the file. `heat` is the LWIR build's, None
    in EO. An asset in `worn` is tagged."""

    sky: Sky
    heat: Heat | None
    worn: frozenset[str] = frozenset()
    imported: dict[str, list[bpy.types.Object]] = field(default_factory=dict)


@dataclass(frozen=True)
class Condition:
    """How a hull is kept and run: one temperature for all of it or none, minutes
    since its engines stopped, its paint's solar absorptance and its wear."""

    t_k: float | None = None
    engine_off_min: float = 0.0
    paint_absorptance: float | None = None
    wear: float = 0.0

    @classmethod
    def of(cls, spec: Object | Targets) -> "Condition":
        """The condition an object or a ring of targets asks for."""
        return cls(spec.t_k, spec.engine_off_min, spec.paint_absorptance, spec.wear)


def heading_rad(anchor: bpy.types.Object) -> float:
    """Where a hull's bow points, clockwise from north: bow to +Y, as fitted."""
    bow = anchor.matrix_world.to_3x3() @ Vector((0.0, 1.0, 0.0))
    return math.atan2(bow.x, bow.y)


def _vessel(
    name: str, *, condition: Condition, hulls: Hulls, rise_m: float = 0.0
) -> bpy.types.Object:
    """A hull fitted and anchored at the origin under an empty.

    Args:
        name: The asset, a key of `assets.toml`.
        condition: How the hull is kept and run; in LWIR, what its skins radiate.
        hulls: Where an asset already imported is copied from.
        rise_m: Lift above the asset's draught; the LWIR skin's waterline follows
            it."""
    heat = hulls.heat
    if name in hulls.imported:
        parts = [_copy_tree(part, None) for part in hulls.imported[name]]
    else:
        band = "eo" if heat is None else "ir"
        worn = name in hulls.worn
        parts = hulls.imported[name] = import_asset(name, band, worn=worn)
        if heat is None:
            prefix_names(name, parts)
            if worn:
                dirty(parts)

    if heat is not None:
        heat = heat._replace(env=heat.env.stopped(condition.engine_off_min * 60.0))
        t_k = condition.t_k
        uniform = None if t_k is None else thermal_skin(f"{name}_ir", t_k, hulls.sky)
        worn_on = manifest()[name].material if name in hulls.worn else None
        for mesh in mesh_objects(parts):
            # Where the skin finds the water. Set on every hull: a copy inherits it.
            mesh["rise_m"] = rise_m
            for slot, zone in zip(mesh.material_slots, mesh["zones"], strict=True):
                # Copies share mesh data, and a hull given its own t_k differs.
                slot.link = "OBJECT"
                # A rotor's frame is its hub, so its z says nothing of the sea.
                slot.material = uniform or zone_skin(
                    zone,
                    heat,
                    condition.paint_absorptance,
                    wet=not mesh.get("rotor"),
                    wear_material=worn_on,
                    crevice=mesh.get(CREVICE),
                )

    anchor = bpy.data.objects.new(name, None)
    for size in ("beam_m", "length_m"):
        anchor[size] = parts[0][size]
    anchor["height_m"] = parts[0]["height_m"] + rise_m
    anchor["engine_off_min"] = condition.engine_off_min
    anchor["wear"] = condition.wear
    # Pitch and roll go here, so the anchor keeps the pose labels read.
    attitude = bpy.data.objects.new(f"{name}_attitude", None)
    # Lifts the hull inside the attitude, so it pitches and rolls about its waterline.
    draught = bpy.data.objects.new(f"{name}_draught", None)
    for obj in (anchor, attitude, draught):
        bpy.context.collection.objects.link(obj)
    attitude.parent = anchor
    draught.parent = attitude
    for part in parts:
        part.parent = draught
    place(anchor, 0.0, 0.0, 0.0)
    place(attitude, 0.0, 0.0, 0.0)
    place(draught, 0.0, 0.0, rise_m)
    # YXZ euler is Rz @ Rx @ Ry: roll about the keel, innermost.
    attitude.rotation_mode = "YXZ"
    return anchor


def add_ownship(
    ownship: Ownship,
    rigs: bpy.types.Object,
    hulls: Hulls,
    outputs: Outputs,
    tall: tuple[waves.Wave, ...],
) -> bpy.types.Object:
    """At the origin, bow to +Y, carrying the rigs.

    Args:
        ownship: Its hull, attitude and scripted motion.
        rigs: Parented to the ownship; their offsets are in its frame.
        hulls: Where an asset already imported is copied from.
        outputs: The times keyed and the periods they loop on.
        tall: The waves that are geometry, which it heaves on as well as its script."""
    if ownship.asset is None:
        anchor = bpy.data.objects.new("ownship", None)
        bpy.context.collection.objects.link(anchor)
        place(anchor, 0.0, 0.0, 0.0)
    else:
        anchor = _vessel(
            ownship.asset, condition=Condition(t_k=ownship.t_k), hulls=hulls
        )
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
        length_m, beam_m = (
            (anchor["length_m"], anchor["beam_m"]) if ownship.asset else (0.0, 0.0)
        )

        def value_at(t_s: float) -> float:
            rest = map(
                float, waves.gerstner_rest_m(tall, np.zeros(()), np.zeros(()), t_s)
            )
            hull = waves.Waterplane(*rest, 0.0, length_m, beam_m)
            return scripted(t_s) + waves.heave_m(tall, hull, t_s)

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
    hulls: Hulls,
    outputs: Outputs,
    trails: list[wakes.Wake],
) -> dict[str, list[bpy.types.Object]]:
    """The ring's hulls, by asset."""
    anchors: dict[str, list[bpy.types.Object]] = {}
    for i, (asset, bearing_deg, heading_deg) in enumerate(spec.poses()):
        anchor = _vessel(asset, condition=Condition.of(spec), hulls=hulls)
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
    hulls: Hulls,
    outputs: Outputs,
    trails: list[wakes.Wake],
) -> list[bpy.types.Object]:
    """`spec`'s hulls, each wake under way added to `trails`."""
    anchor = _vessel(
        spec.asset, condition=Condition.of(spec), hulls=hulls, rise_m=spec.rise_m
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
    hulls: Hulls,
) -> bpy.types.Object:
    """A turbine facing the wind, its rotor starting at an angle of its own so a farm's
    rotors turn out of step."""
    anchor = _vessel(
        farm.asset,
        condition=Condition(paint_absorptance=farm.paint_absorptance, wear=farm.wear),
        hulls=hulls,
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


def ride(
    anchors: Sequence[bpy.types.Object],
    field: tuple[waves.Wave, ...],
    tall: tuple[waves.Wave, ...],
    outputs: Outputs,
) -> None:
    """Key each hull's pitch and roll to the sea under its keyed pose, and its heave
    to the waves that are geometry.

    Args:
        anchors: The hulls, each over its attitude empty.
        field: Every wave, which the pitch and roll follow.
        tall: The waves that are geometry, which alone heave the hull: the rest are
            shading, which a heaving hull would break its waterline on.
        outputs: The times keyed, one per frame."""
    sc = bpy.context.scene
    hulls = []
    for anchor in anchors:
        (attitude,) = anchor.children
        hulls.append((anchor, attitude, anchor["length_m"], anchor["beam_m"]))
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
            hull = waves.Waterplane(*rest, heading, length_m, beam_m)
            pitch, roll = waves.attitude(field, hull, t_s)
            # +x turns the bow up; +y turns starboard down.
            attitude.rotation_euler = (pitch, -roll, 0.0)
            attitude.location.z = waves.heave_m(tall, hull, t_s)
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
    hulls: Hulls,
    trails: list[wakes.Wake],
) -> tuple[dict[str, list[bpy.types.Object]], dict[str, list[bpy.types.Object]]]:
    """Every object and ring hull, then every turbine, by asset.

    Worn once all are built: a hull is copied from its asset's first, which would carry
    that one's wear.

    Args:
        scenario: Its objects, target ring and farms.
        radius_m: Earth's effective radius, which drops a distant hull to the curved
            sea.
        hulls: Where an asset already imported is copied from.
        trails: Each wake under way is added to it."""
    outputs = scenario.outputs
    wearing: list[tuple[list[bpy.types.Object], float, str]] = []
    vessels: defaultdict[str, list[bpy.types.Object]] = defaultdict(list)
    for i, spec in enumerate(scenario.objects):
        anchors = _object(spec, radius_m, hulls, outputs, trails)
        wearing.append((anchors, spec.wear, f"wear/objects/{i}"))
        vessels[spec.asset] += anchors
    if (ring := scenario.targets) is not None:
        hulls_by = _targets(ring, radius_m, hulls, outputs, trails)
        wearing.append((list(chain(*hulls_by.values())), ring.wear, "wear/targets"))
        for asset, anchors in hulls_by.items():
            vessels[asset] += anchors
    turbines: defaultdict[str, list[bpy.types.Object]] = defaultdict(list)
    for f, farm in enumerate(scenario.farms):
        for k, (range_m, bearing_deg) in enumerate(farm.positions()):
            rng = substream(scenario.seed, f"farms/{f}/{k}/rotor")
            anchor = _turbine(
                farm, range_m, bearing_deg, rng, scenario, radius_m, hulls
            )
            wearing.append(([anchor], farm.wear, f"wear/farms/{f}/{k}"))
            turbines[farm.asset].append(anchor)
    for anchors, amount, stream in wearing:
        if amount > 0.0:
            rng = substream(scenario.seed, stream)
            for anchor in anchors:
                mark(anchor, amount, rng)
    return vessels, turbines


def waterline_m(anchor: bpy.types.Object) -> np.ndarray:
    """Where a hull's edges cross its waterline, as world east and north, (N, 2).

    The waterline is z = 0 in the attitude's frame, which heaves, pitches and rolls
    with the waves: `_fit` puts the keel the asset's draught below it, and `_vessel`
    lifts it by `rise_m`.
    """
    # Limitation: the crossings, not the segments between them, so the nearest one to a
    # camera at range r is up to side^2 / 2r further than a side facing it.
    (attitude,) = anchor.children
    to_world = np.array(attitude.matrix_world)
    to_hull = np.linalg.inv(to_world)
    crossings = []
    for part in mesh_objects([anchor]):
        mesh = part.data
        local = faces.floats(mesh.vertices, "co", 3 * len(mesh.vertices))
        ends = faces.ints(mesh.edges, "vertices", 2 * len(mesh.edges))
        m = to_hull @ np.array(part.matrix_world)
        hull = local.reshape(-1, 3) @ m[:3, :3].T + m[:3, 3]
        a, b = hull[ends.reshape(-1, 2)].transpose(1, 0, 2)
        cross = a[:, 2] * b[:, 2] < 0
        a, b = a[cross], b[cross]
        t = -a[:, 2] / (b[:, 2] - a[:, 2])
        at = a + t[:, None] * (b - a)
        crossings.append((at @ to_world[:3, :3].T + to_world[:3, 3])[:, :2])
    return np.concatenate(crossings)
