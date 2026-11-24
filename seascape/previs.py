"""A scenario's ownship, objects and cameras on a flat sea, from above and from
behind the ownship, with each object's range ringed.
"""

import math
import shutil
import tempfile
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import chain, repeat
from pathlib import Path
from typing import Any, Literal

import bpy
import numpy as np
import numpy.typing as npt
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector

from seascape import angles, motion, processes
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
    LABEL_PX,
    LINE_PX,
    TICK_LABEL_PX,
    TICK_STEP_DEG,
    Px,
    grown,
    hud,
    hud_lines,
    label_bearing_deg,
    ring_ranges_m,
    samples,
    top_view,
)
from seascape.physics.colour import srgb_hex_decode
from seascape.render import video

type View = Literal["top", "chase"]
VIEWS: tuple[View, ...] = ("top", "chase")
# Square from above, as range rings are round.
SIZE_PX: dict[View, tuple[int, int]] = {"top": (720, 720), "chase": (1280, 720)}
# Each view's picture, `<title>-<name>.png`.


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
# Judgement: the chase camera, in ownship lengths behind and above its waterline, and
# its pitch down.
CHASE_BACK, CHASE_UP = 2.2, 1.32
CHASE_PITCH_DEG = -18.5
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


def _on_water(
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
    return _on_water(a, radius_m[:, None], z_m).reshape(-1, 3)


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
class _Scene:
    """What `draw` moves between frames."""

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
        verts = [(0.0, 0.0, FAN_Z), *_on_water(bearings_deg, reach_m, FAN_Z)]
        # As rig-view colours its frusta.
        fill = _flat(
            f"{mount.name}_fill",
            verts,
            [tuple(range(len(verts)))],
            (*rgb / 255, fill_alpha),
        )
        near = Vector((0.0, 0.0, EDGE_Z))
        far = [Vector(p) for p in _on_water(bearings_deg[[0, -1]], reach_m, EDGE_Z)]
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


def _eevee(heading_deg: float, view: View) -> None:
    """EEVEE without shadows, rays or motion blur. The camera sees `WATER` below the
    horizon and `SKY` above; hulls are lit by Blender's studio light and a sun off a
    bow heading `heading_deg`; drawn marks are unlit in their object colour.
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
    top = view == "top"
    fading = _material("fade", lambda t: _fading(t, framed=top), blended=True)
    for obj in sc.objects:
        if obj.type == "MESH" and not obj.material_slots:
            obj.data.materials.append(fading if obj.color[3] < 1.0 else solid)
            obj.visible_shadow = False


def _build(scenario: Scenario, view: View) -> _Scene:
    """Every hull at its pose, keyed per output frame, on a flat sea under EEVEE.

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
    _eevee(math.degrees(start.heading_rad), view)
    # Judgement: an ownship without a hull is drawn as one this long.
    bare_length_m = 10.0
    asset = scenario.ownship.asset
    return _Scene(
        roots=roots,
        fans=fans,
        rings=rings,
        path=path,
        heading_rad=start.heading_rad,
        length_m=bare_length_m if asset is None else float(ownship["length_m"]),
        reach_m=reach_m,
    )


def _turn_fans(scene: _Scene, t_s: float) -> None:
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


def _chase(length_m: float, east_m: float, north_m: float, look_rad: float) -> Matrix:
    """`T_world_camera`, in Blender's camera axes, behind and above an ownship
    `length_m` long at `east_m`, `north_m`, looking along the bearing `look_rad`.
    """
    bow = Vector(angles.direction(math.degrees(look_rad), 0.0))
    at = Vector((east_m, north_m, 0.0)) - CHASE_BACK * length_m * bow
    at.z = CHASE_UP * length_m
    toward = Vector(angles.direction(math.degrees(look_rad), CHASE_PITCH_DEG))
    return Matrix.LocRotScale(at, toward.to_track_quat("-Z", "Y"), None)


@dataclass(frozen=True)
class _Lens:
    """A view's camera, its size, the bearing its rings' labels sit at at the start,
    and what a pixel of it spans on the water.
    """

    view: View
    camera: bpy.types.Object
    size_px: tuple[int, int]
    label_deg: float

    @property
    def top(self) -> bool:
        return self.view == "top"

    @property
    def k(self) -> float:
        """Pixels of this view per pixel of a 1000 px tall one."""
        return self.size_px[1] / 1000.0

    def m_per_px(self, at: Vector) -> float:
        data = self.camera.data
        if self.top:
            return data.ortho_scale / self.size_px[0]
        eye = self.camera.matrix_world.translation
        return (at - eye).length * 2.0 * math.tan(data.angle_x / 2) / self.size_px[0]

    def slant(self, at: Vector) -> float:
        """A flat mark at `at` seen at a slant is foreshortened by this much: the
        sine of its elevation. Judgement: no less than 0.15, past which it is a line.
        """
        if self.top:
            return 1.0
        return max((self.camera.matrix_world.translation - at).normalized().z, 0.15)

    def up(self) -> Vector:
        """The unit direction on the water that runs up the view: the camera's +Y
        from above, the way it looks from behind.
        """
        axis = (0.0, 1.0, 0.0) if self.top else (0.0, 0.0, -1.0)
        along = self.camera.matrix_world.to_3x3() @ Vector(axis)
        return Vector((along.x, along.y, 0.0)).normalized()


def _camera(scene: _Scene, view: View) -> _Lens:
    """`view`'s camera, its rings' labels where no hull is at the start: from above
    astern, which the chase camera does not see, between the compass ticks; from
    behind, inside its field.
    """
    sc = bpy.context.scene
    camera = bpy.data.objects.new(view, bpy.data.cameras.new(view))
    if view == "top":
        camera.data.type = "ORTHO"
    else:
        # Judgement: the chase camera's field of view.
        camera.data.angle_x = math.radians(75.0)
    sc.collection.objects.link(camera)
    camera.data.sensor_fit = "HORIZONTAL"
    camera.data.clip_start = 1.0
    # The far clip plane cuts in silence: past the sea's edge, from either camera.
    camera.data.clip_end = 2.0 * scene.reach_m
    sc.frame_set(0)
    hulls_deg = [
        angles.bearing_deg(*r.matrix_world.translation.xy) for r in scene.roots[1:]
    ]
    if view == "top":
        about_deg = math.degrees(scene.heading_rad)
        off_deg = np.arange(105.0, 256.0, TICK_STEP_DEG)
    else:
        about_deg = math.degrees(scene.look_rad(0.0))
        off_deg = np.arange(-30.0, 31.0, 5.0)
    label_deg = label_bearing_deg(about_deg + off_deg, hulls_deg)
    return _Lens(view, camera, SIZE_PX[view], label_deg)


def _hulls(scene: _Scene, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Every hull's east and north over `frames`, (n, 2), and half its diagonal, the
    circle that holds it however it turns, (n,).
    """
    sc = bpy.context.scene
    sides = np.array(
        [(r.get("length_m", scene.length_m), r.get("beam_m", 0.0)) for r in scene.roots]
    )
    found = []
    for k in frames:
        sc.frame_set(int(k))
        found += [tuple(r.matrix_world.translation.xy) for r in scene.roots]
    return np.array(found), np.tile(np.hypot(*sides.T) / 2, len(frames))


def _aim_top(scene: _Scene, lens: _Lens, frames: np.ndarray) -> float:
    """Frame every hull whole over `frames`, every ring's label and the compass with
    room for its labels; the metres a pixel spans. An object grows to at most
    `OBJECT_PX` across, well inside `MARGIN`, so its true size is fitted.
    """
    compass_deg = math.degrees(scene.heading_rad) + np.arange(0.0, 360.0, 90.0)
    outer = max(scene.rings, default=0.0)
    labels = _on_water(lens.label_deg, np.array(list(scene.rings)), 0.0)
    hulls, reach_m = _hulls(scene, frames)
    m_per_px = 0.0
    # How far the compass's labels reach follows the scale, so each pass fits what
    # the last one's scale draws. ponytail: three passes settle any scene seen so
    # far; iterate to a tolerance if one does not.
    for _ in range(3):
        pad_m = (TICK_LABEL_PX + LABEL_PX) * lens.k * m_per_px if scene.rings else 0.0
        marks = np.concatenate(
            [labels[:, :2], _on_water(compass_deg, outer + pad_m, 0.0)[:, :2]]
        )
        view = top_view(
            np.concatenate([hulls, marks]),
            scene.heading_rad,
            lens.size_px,
            np.concatenate([reach_m, np.zeros(len(marks))]),
        )
        m_per_px = view.m_per_px
    lens.camera.data.ortho_scale = view.m_per_px * lens.size_px[0]
    lens.camera.matrix_world = Matrix(view.matrix(0.5 * lens.camera.data.clip_end))
    return view.m_per_px


def _sizes(scene: _Scene, lens: _Lens) -> float | None:
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


def _grow(scene: _Scene, lens: _Lens) -> float | None:
    """Scale each object but the ownship about its origin, uniformly so it keeps its
    proportions, by `grown`. Returns the largest factor among those whose origin is
    in view, if any is.
    """
    seen = []
    for root in scene.roots[1:]:
        at = root.matrix_world.translation
        grow = grown(lens.m_per_px(at), max(root["length_m"], root["beam_m"]))
        root.scale = (grow,) * 3
        p = _px(lens, at)
        if p and 0.0 <= p[0] <= lens.size_px[0] and 0.0 <= p[1] <= lens.size_px[1]:
            seen.append(grow)
    return max(seen, default=None)


def _px(lens: _Lens, at: npt.ArrayLike) -> Px | None:
    """Where `at` falls in `lens`'s picture, in pixels from its top left; none behind
    the camera.
    """
    p = world_to_camera_view(bpy.context.scene, lens.camera, Vector(at))
    if p.z <= 0.0:
        return None
    return p.x * lens.size_px[0], (1.0 - p.y) * lens.size_px[1]


def _key(scene: _Scene, lens: _Lens, frame: int) -> None:
    """Key at `frame` what `_sizes`, `_turn_fans` and the chase camera set. The top
    camera, still over the run, stays unkeyed.
    """
    for root in scene.roots[1:]:
        root.keyframe_insert("scale", frame=frame)
    if not lens.top:
        for path in ("location", "rotation_euler"):
            lens.camera.keyframe_insert(path, frame=frame)
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
    scene: _Scene,
    lens: _Lens,
    frames: np.ndarray,
    folder: Path,
) -> list[Path]:
    """Render `lens` at `frames` into `folder`, with the HUD; the pictures in order."""
    sc = bpy.context.scene
    m_per_px = _aim_top(scene, lens, frames) if lens.top else None
    sc.camera = lens.camera
    sc.render.resolution_x, sc.render.resolution_y = lens.size_px
    times = scenario.outputs.times_s
    look_0 = scene.look_rad(0.0)
    # From above, the compass round the outer ring, still over the run.
    compass = None
    if lens.top and scene.rings and m_per_px is not None:
        centre = _px(lens, (0.0, 0.0, 0.0))
        compass = None if centre is None else (centre, max(scene.rings) / m_per_px)
    huds, labels = [], []
    for k in frames:
        t_s = times[int(k)]
        sc.frame_set(int(k))
        _turn_fans(scene, t_s)
        pose = scene.path(t_s)
        label_deg = lens.label_deg
        if not lens.top:
            look = scene.look_rad(t_s)
            lens.camera.matrix_world = _chase(
                scene.length_m, pose.east_m, pose.north_m, look
            )
            # The labels turn with the view, to stay in its field.
            label_deg += math.degrees(look - look_0)
        most = _sizes(scene, lens)
        _key(scene, lens, int(k))
        huds.append(hud_lines(t_s, pose.heading_rad, pose.speed_mps, most))
        anchors = {r: _px(lens, _on_water(label_deg, r, RING_Z)) for r in scene.rings}
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
    scene = _build(scenario, view)
    lens = _camera(scene, view)
    [start] = _frames(scenario, scene, lens, np.array([0]), work / "still")
    # The work folder can sit on another filesystem.
    shutil.move(start, still)
    if frames is not None:
        pictures = _frames(scenario, scene, lens, frames, work / "frames")
        out = still.with_suffix(".mp4")
        # A preview's quality, small enough to upload as it plays.
        video.sequence(pictures, SIZE_PX[view], 1.0 / play_fps, out, "HIGH")


def draw(
    scenario: Scenario,
    folder: Path,
    title: str,
    video_fps: float | None = None,
    progress: Callable[[int, int], None] = lambda *_: None,
) -> Drawn:
    """Draw the scenario from above and from behind its ownship, both views at once.

    Args:
        scenario: The scenario to draw.
        folder: Where each view's picture goes and, with `video_fps`, an H.264 mp4 of
            each beside it.
        title: The pictures' name before the view's.
        video_fps: Frames a second of the run in each video, as `samples` takes
            them, played in real time; None draws no video.
        progress: Called about once a second with the video frames rendered so far
            and their total, of both views.

    Returns:
        The pictures and videos written.

    Raises:
        RuntimeError: A view failed; the message holds the end of its log.
    """
    folder.mkdir(parents=True, exist_ok=True)
    stills = {v: folder / f"{title}-{v}.png" for v in VIEWS}
    frames, play_fps, total = None, 0.0, 0
    if video_fps is not None:
        outputs = scenario.outputs
        frames, play_fps = samples(outputs.fps, len(outputs.times_s), video_fps)
        total = len(VIEWS) * len(frames)
    with tempfile.TemporaryDirectory() as tmp:
        work = {v: Path(tmp, v) for v in VIEWS}
        logs = {v: Path(tmp, f"{v}.log") for v in VIEWS}
        jobs = {v: (scenario, v, frames, play_fps, stills[v], work[v]) for v in VIEWS}

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
