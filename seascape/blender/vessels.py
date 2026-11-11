"""Hulls and objects on the sea: their poses, orbits, wakes and ride on the waves."""

import math
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from itertools import chain, combinations

import bpy
import numpy as np
from mathutils import Euler, Vector

from seascape import motion
from seascape.assets import Hull, manifest
from seascape.blender import faces
from seascape.blender.blend import (
    animate,
    empty,
    mesh_objects,
    yaw,
)
from seascape.blender.heat import Heat, thermal_skin, zone_skin
from seascape.blender.meshes import (
    box_m,
    import_asset,
    keel_m,
    prefix_names,
    triangles,
)
from seascape.blender.paint import paint, recolourable
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
    in EO. An asset in `worn` is tagged.
    """

    sky: Sky
    heat: Heat | None
    worn: frozenset[str] = frozenset()
    imported: dict[str, list[bpy.types.Object]] = field(default_factory=dict)
    # A hull's `keel_m` heights and cell area, as fitted.
    keels: dict[str, tuple[np.ndarray, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class Condition:
    """How a hull is kept and run: one temperature for all of it or none, minutes
    since its engines stopped and its paint's solar absorptance.
    """

    t_k: float | None = None
    engine_off_min: float = 0.0
    paint_absorptance: float | None = None

    @classmethod
    def of(cls, spec: Object | Targets) -> "Condition":
        """The condition an object or a ring of targets asks for."""
        return cls(spec.t_k, spec.engine_off_min, spec.paint_absorptance)


@dataclass(frozen=True)
class Afloat:
    """A hull `ride` keys: its anchor, the path keyed on it, and its waterplane's
    size.
    """

    anchor: bpy.types.Object
    path: motion.Path
    hull: waves.Waterplane


def heading_rad(anchor: bpy.types.Object) -> float:
    """Where a hull's bow points, clockwise from north: bow to +Y, as fitted."""
    bow = anchor.matrix_world.to_3x3() @ Vector((0.0, 1.0, 0.0))
    return math.atan2(bow.x, bow.y)


def _parts(name: str, hulls: Hulls) -> list[bpy.types.Object]:
    """`name`'s parts: a copy of an asset already imported, else imported, and a
    hull's keel measured as fitted.
    """
    if name in hulls.imported:
        return [_copy_tree(part, None) for part in hulls.imported[name]]
    worn = name in hulls.worn
    band = "eo" if hulls.heat is None else "ir"
    parts = hulls.imported[name] = import_asset(name, band, worn=worn)
    if isinstance(manifest()[name], Hull):
        bpy.context.view_layer.update()
        hulls.keels[name] = keel_m(triangles(mesh_objects(parts)))
    if hulls.heat is None:
        # Before the dirt, so it sits on the new paint, and before the prefix, so
        # the materials carry the names their zones key on.
        if (asset := manifest()[name]).colours:
            recolourable(mesh_objects(parts), asset.paint_hull)
        prefix_names(name, parts)
        if worn:
            dirty(parts)
    return parts


def _vessel(
    name: str, *, condition: Condition, hulls: Hulls, rise_m: float = 0.0
) -> tuple[bpy.types.Object, waves.Waterplane]:
    """A hull fitted and anchored at the origin under an empty, and its size as the
    waves see it.

    Args:
        name: The asset, a key of `assets.toml`.
        condition: How the hull is kept and run; in LWIR, what its skins radiate.
        hulls: Where an asset already imported is copied from.
        rise_m: Lift above the asset's draught; the LWIR skin's waterline and the
            wear's follow it.
    """
    heat = hulls.heat
    parts = _parts(name, hulls)
    for mesh in mesh_objects(parts):
        # Where the skin and the wear find the water. Set on every hull: a copy
        # inherits it.
        mesh["rise_m"] = rise_m
    if heat is not None:
        heat = heat._replace(env=heat.env.stopped(condition.engine_off_min * 60.0))
        t_k = condition.t_k
        uniform = None if t_k is None else thermal_skin(f"{name}_ir", t_k, hulls.sky)
        worn_on = manifest()[name].material if name in hulls.worn else None
        for mesh in mesh_objects(parts):
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

    length_m, beam_m = parts[0]["length_m"], parts[0]["beam_m"]
    # Only a `Hull` has a box; without one it follows the water.
    box_beam_m = box_draught_m = 0.0
    if name in hulls.keels:
        box_beam_m, box_draught_m = box_m(*hulls.keels[name], length_m, rise_m)
    hull = waves.Waterplane(
        length_m=length_m,
        beam_m=beam_m,
        box_beam_m=box_beam_m,
        box_draught_m=box_draught_m,
    )
    anchor = empty(name)
    anchor["length_m"], anchor["beam_m"] = length_m, beam_m
    anchor["height_m"] = parts[0]["height_m"] + rise_m
    anchor["engine_off_min"] = condition.engine_off_min
    # Pitch and roll go here, so the anchor keeps the pose labels read.
    attitude = empty(f"{name}_attitude", anchor)
    # Lifts the hull inside the attitude, so it pitches and rolls about its waterline.
    draught = empty(f"{name}_draught", attitude, (0.0, 0.0, rise_m))
    for part in parts:
        part.parent = draught
    # YXZ euler is Rz @ Rx @ Ry: roll about the keel, innermost.
    attitude.rotation_mode = "YXZ"
    return anchor, hull


def add_ownship(
    ownship: Ownship,
    path: motion.Path,
    rigs: bpy.types.Object,
    hulls: Hulls,
    outputs: Outputs,
    tall: tuple[waves.Wave, ...],
    radius_m: float,
    afloat: list[Afloat],
) -> bpy.types.Object:
    """Bow to +Y at the origin, carrying the rigs, and sailing from it. A hull rides
    the sea once `ride` keys it.

    Args:
        ownship: Its hull, temperature, rise and heave.
        path: Its pose at a time, from `motion.track`.
        rigs: Parented to the hull's draught, or to a bare ownship, in its frame.
        hulls: Where an asset already imported is copied from.
        outputs: The times keyed and the periods they loop on.
        tall: The waves that are geometry, which a bare rig heaves on.
        radius_m: Earth's effective radius, which drops it to the curved sea.
        afloat: Its hull is added to it; a bare rig has none.
    """
    hull = None
    if ownship.asset is None:
        anchor = empty("ownship")
        rigs.parent = anchor
    else:
        anchor, hull = _vessel(
            ownship.asset,
            condition=Condition(t_k=ownship.t_k),
            hulls=hulls,
            rise_m=ownship.rise_m,
        )
        (attitude,) = anchor.children
        (draught,) = attitude.children
        rigs.parent = draught
    anchor.name = "ownship"
    # Limitation: a hull-less ownship has no footprint, so it sails through targets.
    if hull is not None:
        afloat.append(Afloat(anchor, path, hull))
    # Limitation: it leaves no wake; `wakes.Wake` draws only a straight run.
    scripted = motion.sine(0.0, 0.0, 1.0)
    if (heave := ownship.heave) is not None:
        scripted = motion.sine(0.0, heave.amplitude_m, outputs.period_s(heave.period_s))
    # A bare rig is a point: it heaves, but would tilt with every ripple.
    bare = ownship.asset is None and bool(tall)

    def up_m(pose: motion.Pose, t_s: float) -> float:
        z = earth.sea_z_m(pose.east_m, pose.north_m, radius_m) + scripted(t_s)
        if bare:
            point = motion.waterplane(pose, tall, t_s, waves.Waterplane())
            z += waves.heave_m(tall, point, t_s)
        return z

    _follow(anchor, path, outputs.times_s, up_m)
    return anchor


def _follow(
    anchor: bpy.types.Object,
    path: motion.Path,
    times_s: Sequence[float],
    up_m: Callable[[motion.Pose, float], float],
) -> None:
    """Key `anchor` along `path`: its position, `up_m` high, and its pitch, roll and
    heading.
    """

    def location(t_s: float) -> tuple[float, float, float]:
        pose = path(t_s)
        return pose.east_m, pose.north_m, up_m(pose, t_s)

    def rotation(t_s: float) -> tuple[float, float, float]:
        pose = path(t_s)
        return pose.pitch_rad, pose.roll_rad, yaw(math.degrees(pose.heading_rad))

    # YXZ euler is Rz @ Rx @ Ry: roll about the keel, innermost.
    anchor.rotation_mode = "YXZ"
    for data_path, value_at in (("location", location), ("rotation_euler", rotation)):
        values = np.array([value_at(t) for t in times_s])
        for index in range(3):
            # Only what moves is keyed.
            if np.ptp(values[:, index]) == 0.0:
                getattr(anchor, data_path)[index] = values[0, index]
            else:
                animate(
                    anchor,
                    data_path,
                    times_s,
                    lambda t, at=value_at, i=index: at(t)[i],
                    index,
                )


def _on_sea(radius_m: float) -> Callable[[motion.Pose, float], float]:
    """The curved sea's height under a pose: a hull left at z = 0 flies above it.

    Not tilted to the local vertical: at any range the sea reaches, range / R moves a
    hull's ends far less than its draught.
    """
    return lambda pose, _t_s: earth.sea_z_m(pose.east_m, pose.north_m, radius_m)


def _pose(
    anchor: bpy.types.Object,
    range_m: float,
    bearing_deg: float,
    heading_deg: float,
    speed_mps: float,
    drift: Drift | None,
    radius_m: float,
    outputs: Outputs,
) -> motion.Path:
    """Put a hull on the sea at a bearing and range at t = 0, underway along its
    heading and drifting about that pose.
    """
    path = motion.drifting(range_m, bearing_deg, heading_deg, speed_mps, drift, outputs)
    _follow(anchor, path, outputs.times_s, _on_sea(radius_m))
    return path


def _orbit(
    anchor: bpy.types.Object,
    range_m: float,
    bearing_deg: float,
    lap_s: float,
    radius_m: float,
    times_s: Sequence[float],
) -> motion.Path:
    path = motion.orbiting(range_m, bearing_deg, lap_s)
    _follow(anchor, path, times_s, _on_sea(radius_m))
    return path


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
    afloat: list[Afloat],
) -> dict[str, list[bpy.types.Object]]:
    """The ring's hulls, by asset."""
    anchors: dict[str, list[bpy.types.Object]] = {}
    for i, (asset, bearing_deg, heading_deg) in enumerate(spec.poses()):
        anchor, hull = _vessel(asset, condition=Condition.of(spec), hulls=hulls)
        anchor.name = f"target_{i}"
        path = _pose(
            anchor,
            spec.range_m,
            bearing_deg,
            heading_deg,
            spec.speed_mps,
            spec.drift,
            radius_m,
            outputs,
        )
        afloat.append(Afloat(anchor, path, hull))
        trails.extend(_wake(spec, anchor, bearing_deg, heading_deg))
        anchors.setdefault(asset, []).append(anchor)
    return anchors


def _object(
    spec: Object,
    radius_m: float,
    hulls: Hulls,
    outputs: Outputs,
    trails: list[wakes.Wake],
    afloat: list[Afloat],
) -> list[bpy.types.Object]:
    """`spec`'s hulls, each wake under way added to `trails` and each hull to
    `afloat`.
    """
    anchor, hull = _vessel(
        spec.asset, condition=Condition.of(spec), hulls=hulls, rise_m=spec.rise_m
    )
    orbit = spec.orbit
    if orbit is None:
        path = _pose(
            anchor,
            spec.range_m,
            spec.bearing_deg,
            spec.course_deg,
            spec.speed_mps,
            spec.drift,
            radius_m,
            outputs,
        )
        afloat.append(Afloat(anchor, path, hull))
        trails.extend(_wake(spec, anchor, spec.bearing_deg, spec.course_deg))
        return [anchor]
    anchors = [anchor, *(_copy_tree(anchor, None) for _ in range(orbit.count - 1))]
    lap_s = orbit.count * outputs.period_s(orbit.period_s / orbit.count)
    for i, copy in enumerate(anchors):
        bearing_deg = spec.bearing_deg + 360.0 * i / orbit.count
        path = _orbit(copy, spec.range_m, bearing_deg, lap_s, radius_m, outputs.times_s)
        afloat.append(Afloat(copy, path, hull))
    trails.append(
        wakes.Wake(
            speed_mps=path(0.0).speed_mps,
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
    rotors turn out of step.
    """
    anchor, _ = _vessel(
        farm.asset,
        condition=Condition(paint_absorptance=farm.paint_absorptance),
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
    spec = farm.turbine.rotor
    # A loop closes on a turn of `symmetry_rad`.
    symmetry_s = outputs.period_s(spec.symmetry_period_s(rpm)) if rpm else math.inf
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
        lambda t: rest - start - spec.symmetry_rad * t / symmetry_s,
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
    afloat: Sequence[Afloat],
    field: tuple[waves.Wave, ...],
    tall: tuple[waves.Wave, ...],
    outputs: Outputs,
) -> None:
    """Key each hull's pitch and roll to the sea under it along its path, and its heave
    to the waves that are geometry.

    Args:
        afloat: The hulls, each anchor over its attitude empty.
        field: Every wave, which the pitch and roll follow.
        tall: The waves that are geometry, which alone heave the hull: the rest are
            shading, which a heaving hull would break its waterline on.
        outputs: The times keyed, one per frame.
    """
    keyed = len(outputs.times_s) > 1
    for hull in afloat:
        (attitude,) = hull.anchor.children
        for frame, t_s in enumerate(outputs.times_s):
            pose = hull.path(t_s)
            plane = motion.waterplane(pose, tall, t_s, hull.hull)
            pitch, roll = waves.attitude(field, plane, t_s)
            # +x turns the bow up; +y turns starboard down.
            attitude.rotation_euler = (pitch, -roll, 0.0)
            # Straight up in the world, through the tilt keyed on the anchor.
            tilted = Euler(
                (pose.pitch_rad, pose.roll_rad, yaw(math.degrees(pose.heading_rad))),
                "YXZ",
            )
            up = Vector((0.0, 0.0, waves.heave_m(tall, plane, t_s)))
            attitude.location = tilted.to_quaternion().inverted() @ up
            if keyed:
                attitude.keyframe_insert("rotation_euler", index=0, frame=frame)
                attitude.keyframe_insert("rotation_euler", index=1, frame=frame)
                attitude.keyframe_insert("location", frame=frame)


class OverlapError(ValueError):
    """Two targets' hulls meet."""


def overlaps(anchors: list[bpy.types.Object], frames: int) -> list[tuple[str, str]]:
    """Pairs of hulls whose footprints meet in any frame: each a circle of half its
    larger horizontal extent about its anchor, where the frame has moved it.
    """
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
    afloat: list[Afloat],
) -> tuple[dict[str, list[bpy.types.Object]], dict[str, list[bpy.types.Object]]]:
    """Every object and ring hull, then every turbine, by asset.

    Worn once all are built: a hull is copied from its asset's first, which would carry
    that one's wear.

    Args:
        scenario: Its objects, target ring and farms.
        radius_m: Earth's effective radius, which drops a distant hull to the curved
            sea.
        hulls: Where an asset already imported is copied from.
        trails: Each wake under way is added to it.
        afloat: Each hull that rides the sea is added to it; a turbine stands.
    """
    outputs = scenario.outputs
    wearing: list[tuple[list[bpy.types.Object], float, str]] = []
    painting: list[tuple[list[bpy.types.Object], str]] = []
    vessels: defaultdict[str, list[bpy.types.Object]] = defaultdict(list)
    for i, spec in enumerate(scenario.objects):
        anchors = _object(spec, radius_m, hulls, outputs, trails, afloat)
        if spec.colour is not None:
            painting.append((anchors, manifest()[spec.asset].colours[spec.colour]))
        wearing.append((anchors, spec.wear, f"wear/objects/{i}"))
        vessels[spec.asset] += anchors
    if (ring := scenario.targets) is not None:
        hulls_by = _targets(ring, radius_m, hulls, outputs, trails, afloat)
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
    _dress(scenario.seed, wearing, painting)
    return vessels, turbines


def _dress(
    seed: int,
    wearing: list[tuple[list[bpy.types.Object], float, str]],
    painting: list[tuple[list[bpy.types.Object], str]],
) -> None:
    """Wear and paint hulls once all are built: a later copy of the same import
    would inherit either.
    """
    for anchors, amount, stream in wearing:
        if amount > 0.0:
            rng = substream(seed, stream)
            for anchor in anchors:
                mark(anchor, amount, rng)
    for anchors, srgb_hex in painting:
        for anchor in anchors:
            paint(anchor, srgb_hex)


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
