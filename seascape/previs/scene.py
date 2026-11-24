"""What every previs view draws: the scenario's hulls on a flat sea under EEVEE, each
camera's field as a fan and each object's range as a ring.
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any

import bpy
import numpy as np
import numpy.typing as npt
from mathutils import Vector

from seascape import angles, motion
from seascape.blender import faces
from seascape.blender.blend import (
    Num,
    attribute,
    clamped,
    empty,
    feed,
    mix_shader,
    node,
    separate,
    vector,
)
from seascape.blender.vessels import Hulls, add_ownship, add_vessels
from seascape.config import Mount, Scenario
from seascape.motion import cameras_at_start, head_path, ownship_up_m, track
from seascape.outputs import canvas
from seascape.outputs.previs import (
    ring_ranges_m,
)
from seascape.physics.colour import srgb_hex_decode


def _linear(srgb_hex: str) -> tuple[float, float, float, float]:
    """An sRGB colour as linear RGBA, opaque."""
    return (*srgb_hex_decode(srgb_hex), 1.0)


# Judgement, after Tesla's driving view: a near-black blue-grey canvas, objects in
# their own materials' colours, and each camera's cone in its own.
WATER = _linear("#141A22")
SKY = _linear("#1B212B")
RING = _linear("#3A4350")
# Judgement, in pixels of a 1000 px tall view, scaled with the view: a fan's edges'
# width.
EDGE_PX = 1.5
# Judgement: a fan ends where a pixel spans this on the water, `focal_px` times this
# out; past that, small objects stop resolving.
FOOTPRINT_M = 0.1
# Heights above the flat sea that stack the flat layers: fans, their edges, then
# rings. Judgement: far enough apart for the chase camera's depth buffer.
FAN_Z, EDGE_Z, RING_Z = 0.1, 0.15, 0.2
# A line's shape key widens it by this much at value 1; `_widening` caps the value.
WIDTH_UNIT_M = 10.0
# Judgement: points round a circle, smooth at any size it is drawn.
RING_STEPS = 256


def _row_bearings_deg(scenario: Scenario, mount: Mount) -> np.ndarray:
    """Bearings across `mount`'s middle row at the start, left to right, unwrapped."""
    # Judgement: enough points for a smooth arc.
    steps = 64
    spec = mount.camera
    x = np.linspace(-spec.width_px / 2, spec.width_px / 2, steps)
    R_world_camera = cameras_at_start(scenario, mount.rig_name)[mount.camera_name]
    world = spec.rays(x, np.zeros_like(x)) @ R_world_camera.T
    return np.unwrap(angles.bearing_deg(world[:, 0], world[:, 1]), period=360.0)


def _flat(
    name: str,
    verts: Sequence[Sequence[float]] | np.ndarray,
    polygons: Sequence[Sequence[int]],
    colour: tuple[float, ...],
) -> bpy.types.Object:
    """A mesh object in `colour`, linked to the scene."""
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([tuple(v) for v in verts], [], polygons)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.color = colour
    return obj


def _widening(obj: bpy.types.Object, name: str, verts: np.ndarray) -> None:
    """Add a shape key `name` to `obj` with its vertices at `verts`, (n, 3)."""
    if obj.data.shape_keys is None:
        obj.shape_key_add(name="basis")
    key = obj.shape_key_add(name=name, from_mix=False)
    key.data.foreach_set("co", verts.ravel())
    key.slider_max = 10.0


def on_water(
    bearing_deg: npt.ArrayLike, range_m: npt.ArrayLike, z_m: float
) -> np.ndarray:
    """The points `range_m` from the ownship's start, where its track begins at the
    origin, at `bearing_deg` and `z_m` up, broadcast, on the last axis.
    """
    at = np.asarray(range_m)[..., None] * angles.direction(bearing_deg, 0.0)
    at[..., 2] = z_m
    return at


def _circle(radius_m: np.ndarray, z_m: float = RING_Z) -> np.ndarray:
    """(n, 3) points round the origin at `z_m`, `RING_STEPS` per radius, by angle
    within each radius.
    """
    a = np.linspace(0.0, 360.0, RING_STEPS, endpoint=False)
    return on_water(a, radius_m[:, None], z_m).reshape(-1, 3)


def _ring(name: str, radius_m: float) -> bpy.types.Object:
    """A band round the origin, flat so a slant view does not see it as a wall, its
    width the `wide` shape key's value in `WIDTH_UNIT_M`.
    """
    n = RING_STEPS
    quads = [(k, (k + 1) % n, n + (k + 1) % n, n + k) for k in range(n)]
    ring = _flat(name, _circle(np.array([radius_m] * 2)), quads, RING)
    half = WIDTH_UNIT_M / 2
    _widening(ring, "wide", _circle(np.array([radius_m - half, radius_m + half])))
    return ring


def _quad(a: Vector, b: Vector, wide: tuple[float, float]) -> np.ndarray:
    """(4, 3) corners of a line from `a` to `b`, `wide` across at each end, in XY."""
    along = (b - a).normalized()
    across = Vector((-along.y, along.x, 0.0))
    co = [a - wide[0] / 2 * across, b - wide[1] / 2 * across]
    co += [b + wide[1] / 2 * across, a + wide[0] / 2 * across]
    return np.array([tuple(c) for c in co])


def _edge(
    name: str, a: Vector, b: Vector, colour: tuple[float, ...]
) -> bpy.types.Object:
    """A line from `a` to `b`, its width at each end the `near` and `far` shape keys'
    values in `WIDTH_UNIT_M`.
    """
    edge = _flat(name, _quad(a, b, (0.0, 0.0)), [(0, 1, 2, 3)], colour)
    _widening(edge, "near", _quad(a, b, (WIDTH_UNIT_M, 0.0)))
    _widening(edge, "far", _quad(a, b, (0.0, WIDTH_UNIT_M)))
    return edge


@dataclass(frozen=True, slots=True, kw_only=True)
class _Fan:
    """A camera's horizontal field from its rig, and the lines along its sides."""

    fan: bpy.types.Object
    edges: list[bpy.types.Object]
    # Its sides' ends in the fan's frame.
    near: Vector
    far: tuple[Vector, Vector]
    reach_m: float


@dataclass(frozen=True, slots=True, kw_only=True)
class Scene:
    """What a view moves between frames."""

    # The ownship's root first, then each object's.
    roots: list[bpy.types.Object]
    # Each rig's fans, and where its head is at a time.
    fans: dict[str, tuple[motion.Path, list[_Fan]]]
    # Each ring by its range.
    rings: dict[float, bpy.types.Object]
    path: motion.Path
    heading_rad: float
    length_m: float
    # How far the scene reaches: its objects, and its fans' far ends.
    reach_m: float

    def look_rad(self, t_s: float) -> float:
        """The heading of the first rig's head, where its cameras look; the
        ownship's without a rig.
        """
        first = next(iter(self.fans.values()), None)
        return (first[0] if first else self.path)(t_s).heading_rad


def _fans(scenario: Scenario) -> dict[str, tuple[motion.Path, list[_Fan]]]:
    """Each camera's horizontal field as a fan from its rig, fading out where a pixel
    spans `FOOTPRINT_M`, with a line along each side; by rig, with its head's path.
    """
    # Judgement: a fan's fill's and edges' opacity at its rig.
    fill_alpha, edge_alpha = 0.22, 0.8
    # Judgement: the height between stacked fans, well under a pixel from either
    # camera.
    gap_m = 0.01
    fans: dict[str, tuple[motion.Path, list[_Fan]]] = {}
    for i, (mount, rgb) in enumerate(
        zip(scenario.mounts, canvas.palette(len(scenario.mounts)), strict=True)
    ):
        bearings_deg = _row_bearings_deg(scenario, mount)
        reach_m = mount.camera.focal_px * FOOTPRINT_M
        fan = empty(mount.name)
        # Apart in height, or overlapping fans fight for depth in stripes.
        fan.location.z = gap_m * i
        verts = [(0.0, 0.0, FAN_Z), *on_water(bearings_deg, reach_m, FAN_Z)]
        # As rig-view colours its frusta.
        fill = _flat(
            f"{mount.name}_fill",
            verts,
            [tuple(range(len(verts)))],
            (*rgb / 255, fill_alpha),
        )
        near = Vector((0.0, 0.0, EDGE_Z))
        far = [Vector(p) for p in on_water(bearings_deg[[0, -1]], reach_m, EDGE_Z)]
        edges = [
            _edge(f"{mount.name}_{side}", near, end, (*rgb / 255, edge_alpha))
            for side, end in zip(("left", "right"), far, strict=True)
        ]
        for part in (fill, *edges):
            # `_fading` reads it.
            part.parent, part["fade"] = fan, 1.0 / reach_m
        name = mount.rig_name
        if name not in fans:
            fans[name] = (head_path(scenario, name), [])
        fans[name][1].append(
            _Fan(fan=fan, edges=edges, near=near, far=(far[0], far[1]), reach_m=reach_m)
        )
    return fans


def _emission(t: bpy.types.NodeTree, colour: tuple[float, ...] | None = None) -> Any:
    """An unlit shader in `colour`, or its object's colour."""
    if colour is None:
        colour = node(t, "ShaderNodeObjectInfo").outputs["Color"]
    return node(t, "ShaderNodeEmission", {"Color": colour}).outputs["Emission"]


def _see_through(t: bpy.types.NodeTree, alpha: Any, shader: Any) -> Any:
    """`shader` blended over what lies behind by `alpha`."""
    clear = node(t, "ShaderNodeBsdfTransparent").outputs["BSDF"]
    return mix_shader(t, alpha, clear, shader)


def _material(
    name: str, surface: Callable[[bpy.types.NodeTree], Any], *, blended: bool
) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    if blended:
        material.surface_render_method = "BLENDED"
    t = material.node_tree
    t.nodes.clear()
    node(t, "ShaderNodeOutputMaterial", {"Surface": surface(t)})
    return material


def _fading(t: bpy.types.NodeTree, *, framed: bool) -> Any:
    """Unlit in its object's colour, fading out where its object coordinates' length
    times its `fade` reaches 1; `framed`, also over the frame's border, so a fan past
    it ends without a hard line.
    """
    coords = node(t, "ShaderNodeTexCoord").outputs
    far = Num(t, vector(t, "LENGTH", coords["Object"]))
    far *= attribute(t, "fade", "OBJECT").outputs["Fac"]
    alpha = Num(t, clamped(t, 1.0 - far))
    alpha *= node(t, "ShaderNodeObjectInfo").outputs["Alpha"]
    if framed:
        # Judgement: the share of the frame it fades over.
        share = 0.1
        x, y = (Num(t, c) for c in separate(t, coords["Window"])[:2])
        edge = x.op("MINIMUM", 1.0 - x).op("MINIMUM", y.op("MINIMUM", 1.0 - y))
        alpha *= clamped(t, edge / share)
    return _see_through(t, alpha.socket, _emission(t))


def _sea(t: bpy.types.NodeTree) -> Any:
    """The sea, see-through: its colour blended over what lies below, the more the
    flatter it is seen. No reflection, so no glint.
    """
    # Judgement: the sea's opacity looking straight down and at grazing, so a hull's
    # parts just under it show near the boat and the far sea reads solid.
    facing = node(t, "ShaderNodeLayerWeight").outputs["Facing"]
    alpha = node(
        t, "ShaderNodeMapRange", {"Value": facing, "To Min": 0.55, "To Max": 0.95}
    )
    return _see_through(t, alpha.outputs["Result"], _emission(t, WATER))


def _drown(material: bpy.types.Material) -> None:
    """Mix `material` toward the sea's colour with depth below the waterline."""
    # Judgement: all of it this far down, so a hull's drives show faintly and its
    # keel not at all.
    depth_m = 1.0
    t = material.node_tree
    out = t.get_output_node("EEVEE")
    if out is None or not out.inputs["Surface"].is_linked:
        return
    shader = out.inputs["Surface"].links[0].from_socket
    z = separate(t, node(t, "ShaderNodeNewGeometry").outputs["Position"])[2]
    deep = node(
        t,
        "ShaderNodeMapRange",
        {"Value": z, "From Min": 0.0, "From Max": -depth_m},
        interpolation_type="SMOOTHSTEP",
    )
    drowned = mix_shader(t, deep.outputs["Result"], shader, _emission(t, WATER))
    feed(t, out.inputs["Surface"], drowned)


def _eevee(heading_deg: float, *, framed: bool) -> None:
    """EEVEE without shadows, rays or motion blur. The camera sees `WATER` below the
    horizon and `SKY` above; hulls are lit by Blender's studio light and a sun off a
    bow heading `heading_deg`; drawn marks are unlit in their object colour, and a
    fan fades over the frame's border when `framed`.
    """
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_EEVEE"
    # Judgement: EEVEE's samples a pixel and its pixel filter's width: 8 keep a hull's
    # fine detail from shimmering as it turns.
    sc.eevee.taa_render_samples, sc.render.filter_size = 8, 1.5
    sc.eevee.use_shadows = sc.eevee.use_raytracing = False
    sc.render.use_motion_blur = False
    t = sc.world.node_tree
    t.nodes.clear()
    # Judgement: the horizon's softness, as the sine of the elevation over which the
    # camera's view blends from sea to sky: about half a degree.
    horizon_sin = 0.01
    z = separate(t, node(t, "ShaderNodeTexCoord").outputs["Generated"])[2]
    up = node(
        t,
        "ShaderNodeMapRange",
        {"Value": z, "From Min": -horizon_sin, "From Max": horizon_sin},
        interpolation_type="SMOOTHSTEP",
    )
    seen = node(
        t,
        "ShaderNodeMix",
        {"Factor": up.outputs["Result"], "A": WATER, "B": SKY},
        data_type="RGBA",
    )
    # Judgement: the light a hull's materials see: Blender's studio HDRI all round, and
    # a sun low on the ownship's port quarter at the start, so a hull's sides read
    # apart and a camera astern sees their lit side. No shadows: they double a frame's
    # time.
    world_dir = bpy.utils.system_resource("DATAFILES", path="studiolights/world")
    hdri = node(
        t,
        "ShaderNodeTexEnvironment",
        image=bpy.data.images.load(str(Path(world_dir) / "studio.exr")),
    )
    lit, shown = (
        node(t, "ShaderNodeBackground", {"Color": c}).outputs["Background"]
        for c in (hdri.outputs["Color"], seen.outputs["Result"])
    )
    camera = node(t, "ShaderNodeLightPath").outputs["Is Camera Ray"]
    node(t, "ShaderNodeOutputWorld", {"Surface": mix_shader(t, camera, lit, shown)})
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
    sun.data.energy, sun.data.use_shadow = 3.0, False
    toward = angles.direction(heading_deg + 210.0, 20.0)
    # A sun shines along its -Z.
    sun.rotation_euler = Vector(toward).to_track_quat("Z", "Y").to_euler()
    sc.collection.objects.link(sun)
    # Judgement: every material at least this rough. A glossy glint is smaller than a
    # pixel at this size and aliases into sparkle.
    roughness = 0.6
    for material in bpy.data.materials:
        shader = faces.bsdf(material)
        if shader is None:
            continue
        rough = shader.inputs["Roughness"]
        if not rough.is_linked:
            rough.default_value = max(rough.default_value, roughness)
        _drown(material)
    solid = _material("flat", _emission, blended=False)
    fading = _material("fade", lambda t: _fading(t, framed=framed), blended=True)
    for obj in sc.objects:
        if obj.type == "MESH" and not obj.material_slots:
            obj.data.materials.append(fading if obj.color[3] < 1.0 else solid)
            obj.visible_shadow = False


def build(scenario: Scenario, *, framed: bool) -> Scene:
    """Every hull at its pose, keyed per output frame, on a flat sea under EEVEE; a
    fan fades over the frame's border when `framed`.

    The sea is flat and Earth too: the top view is a map, and a curved sea would sink a
    far hull below the flat plane.
    """
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    # Colours as drawn: the factory AgX and dither would shift them.
    sc.view_settings.view_transform = "Standard"
    sc.render.dither_intensity = 0.0
    sc.world = bpy.data.worlds.new("previs")
    times = scenario.outputs.times_s
    path = track(scenario.ownship, scenario.outputs, scenario.turn)
    hulls = Hulls(scenario.sky, None)  # a previs draws no wear
    ownship = add_ownship(
        scenario.ownship,
        path,
        ownship_up_m(scenario.ownship, scenario.outputs, (), math.inf),
        empty("rigs"),
        hulls,
        times,
        [],
    )
    vessels, turbines = add_vessels(scenario, math.inf, hulls, [], [])
    roots = [ownship, *chain(*vessels.values(), *turbines.values())]

    sc.frame_set(0)  # runs the depsgraph: matrix_world is stale before
    ranges_m = [r.matrix_world.translation.xy.length for r in roots[1:]]
    rings = {r: _ring(f"ring_{r:.0f}", r) for r in ring_ranges_m(ranges_m)}

    fans = _fans(scenario)
    # Judgement: a fan's length where no object is farther than half of it.
    least_m = 500.0
    reach_m = max(
        [2.0 * r for r in ranges_m]
        + [least_m]
        + [f.reach_m for _, rig in fans.values() for f in rig]
    )
    # The sea's surface out to the far clip plane, over what lies below the
    # waterline; past it the world's sea, the same colour, goes on to the horizon.
    # Blended objects draw farthest origin first: its origin this far below puts it
    # under every fan from either camera.
    sink_m = 1.0e4
    sea = _flat(
        "sea",
        _circle(np.array([1.0]), sink_m / (2.0 * reach_m)),
        [tuple(range(RING_STEPS))],
        WATER,
    )
    sea.scale = (2.0 * reach_m,) * 3
    sea.location.z = -sink_m
    sea.data.materials.append(_material("sea", _sea, blended=True))
    start = path(0.0)
    _eevee(math.degrees(start.heading_rad), framed=framed)
    # Judgement: an ownship without a hull is drawn as one this long.
    bare_length_m = 10.0
    asset = scenario.ownship.asset
    return Scene(
        roots=roots,
        fans=fans,
        rings=rings,
        path=path,
        heading_rad=start.heading_rad,
        length_m=bare_length_m if asset is None else float(ownship["length_m"]),
        reach_m=reach_m,
    )


def turn_fans(scene: Scene, t_s: float) -> None:
    """Each rig's fans to its head's place and turn at `t_s`.

    Limitation: a fan turns by its head's heading since the start, about the vertical,
    which is exact on a level hull; a pitched or rolled one shifts an off-axis
    camera's bearing too, as `rigs.boresight_deg` reads it.
    """
    for at, fans in scene.fans.values():
        head = at(t_s)
        turn = head.heading_rad - at(0.0).heading_rad
        for f in fans:
            f.fan.location.xy = (head.east_m, head.north_m)
            f.fan.rotation_euler.z = angles.yaw(math.degrees(turn))
