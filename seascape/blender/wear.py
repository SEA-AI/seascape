"""Wear attributes written at import, in the fitted hull frame: metres, bow +Y,
waterline z = 0.
"""

from typing import NamedTuple

import bpy
import cv2
import numpy as np

from seascape.assets import Material
from seascape.blender import faces
from seascape.blender.blend import (
    Num,
    attribute,
    clamped,
    feed,
    mesh_objects,
    mix_shader,
    node,
    scalar,
    separate,
    source,
    vector,
)
from seascape.physics.thermal import ZONE_NAMES, ZoneName

# Occlusion is ray-traced and only the shape decides it, so each asset bakes its
# crevices once into an image, read through the corner attribute UV.
UV = "wear_uv"
CREVICE = "wear_crevice"  # on each mesh object: its asset's baked image
CREVICE_FORMAT = 1  # raise to rebake every cached crevice image
# Judgement: a crevice is what lies within 0.6 m of other surfaces.
CREVICE_M = 0.6
# Judgement.
TEXEL_M = CREVICE_M / 4


class Weights(NamedTuple):
    """How much of each layer a zone collects, and whether runoff streaks it."""

    grime: float
    rust: float
    stain: float
    fouling: float
    soot: float
    scuff: float
    dropping: float
    fade: float
    runoff: float


# The layers the masks cover; `runoff` only gates streaks.
LAYERS = Weights._fields[:-1]
# What covers the surface, bottom up; `fade` only tints the paint.
COVERS = ("grime", "rust", "stain", "fouling", "soot", "scuff", "dropping")
# A mask group output past `LAYERS`: where a dropping's dark core is.
CORE = "dropping_core"

# Judgement, all of it. A stain and fouling are the waterline's, so only what the sea
# wets takes them; cloth sags and flexes, so water runs off it without leaving a streak.
PROFILE: dict[ZoneName, Weights] = {
    "paint_hull": Weights(1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.5, 1.0),
    "paint_superstructure": Weights(0.8, 0.7, 0.0, 0.0, 0.0, 0.7, 0.6, 0.5, 1.0),
    "deck": Weights(1.0, 0.6, 0.0, 0.0, 0.0, 0.6, 0.3, 0.3, 1.0),
    "glass": Weights(0.35, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    "bare_metal": Weights(0.6, 0.6, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 1.0),
    "rubber": Weights(0.6, 0.0, 0.6, 0.6, 0.0, 1.0, 0.3, 0.7, 1.0),
    "fabric": Weights(0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.7, 0.0),
    "funnel": Weights(0.5, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.3, 1.0),
    "exhaust": Weights(0.4, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    "engine_cowl": Weights(0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.3, 0.7, 1.0),
    "wood_composite": Weights(0.8, 0.0, 1.0, 1.0, 0.0, 0.5, 0.3, 0.6, 1.0),
}


def weights(material: Material) -> np.ndarray:
    """(zones, weights), zones in vocabulary order."""
    table = np.array([PROFILE[z] for z in ZONE_NAMES], dtype=np.float32)
    # Judgement: other hulls' fittings are stainless; a chip on steel shows rust.
    gone = "scuff" if material == "steel" else "rust"
    table[:, Weights._fields.index(gone)] = 0.0
    return table


def below_edge_m(position_m: np.ndarray, zone: np.ndarray) -> np.ndarray:
    """Each point's depth below the highest point of its zone over the same patch of
    plan: for a hull side the sheer, for a house wall its roof edge.
    """
    # shortcut: a plan grid, not a walk up the surface; a tier stepped back by less
    # than a patch reads its upper tier's edge.
    length_m = float(np.ptp(position_m[:, :2], axis=0).max())
    # Judgement: patches 1/200 of the length, no less than 10 cm, each reaching its
    # eight neighbours, so a side flared out by a patch still finds its sheer.
    patch = np.floor(position_m[:, :2] / max(length_m / 200.0, 0.1)).astype(np.int64)
    patch -= patch.min(axis=0)
    rows, cols = patch.max(axis=0) + 1
    cell = (zone * rows + patch[:, 0]) * cols + patch[:, 1]
    top = np.full((zone.max() + 1) * rows * cols, -np.inf)
    np.maximum.at(top, cell, position_m[:, 2])
    reach = np.stack(
        [cv2.dilate(z, np.ones((3, 3), np.uint8)) for z in top.reshape(-1, rows, cols)]
    )
    return reach.ravel()[cell] - position_m[:, 2]


def tag(
    objects: list[bpy.types.Object],
    zones: list[np.ndarray],
    material: Material,
    *,
    still: bool,
) -> None:
    """`objects` already fitted; a skinned mesh is read as posed. A `still` asset is
    never under way, so nothing polishes off its fouling.
    """
    bpy.context.view_layer.update()  # the fit leaves matrix_world stale
    depsgraph = bpy.context.evaluated_depsgraph_get()
    table = weights(material)
    corners_m, corner_zones = [], []
    for obj, zone in zip(objects, zones, strict=True):
        mesh = obj.data
        posed = obj.evaluated_get(depsgraph).to_mesh() if obj.modifiers else mesh
        if len(posed.vertices) != len(mesh.vertices):
            raise ValueError(f"{obj.name}: its modifiers change its vertex count")
        world = np.array(obj.matrix_world)
        position_m = (
            faces.floats(posed.vertices, "co", 3 * len(mesh.vertices)).reshape(-1, 3)
            @ world[:3, :3].T
            + world[:3, 3]
        )
        normal = faces.floats(posed.polygons, "normal", 3 * len(zone)).reshape(-1, 3)
        if posed is not mesh:
            obj.evaluated_get(depsgraph).to_mesh_clear()
        # A normal takes the inverse transpose; a mirrored part's winding turns it in.
        linear = world[:3, :3]
        normal = normal @ np.linalg.inv(linear) * np.sign(np.linalg.det(linear))
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
        _write(mesh, "wear_position_m", "POINT", position_m)
        _write(mesh, "wear_normal", "FACE", normal)
        for i, name in enumerate(Weights._fields):
            _write(mesh, f"wear_{name}", "FACE", table[zone, i])
        n_loops = len(mesh.loops)
        totals = faces.ints(mesh.polygons, "loop_total", len(zone))
        corners_m.append(position_m[faces.ints(mesh.loops, "vertex_index", n_loops)])
        corner_zones.append(np.repeat(zone, totals))
    corner_m = np.concatenate(corners_m)
    below_m = below_edge_m(corner_m, np.concatenate(corner_zones))
    # Per asset, on every face, so they survive the LWIR import's join.
    length_m = float(np.ptp(corner_m[:, 1]))
    steel = float(material == "steel")
    for obj, at in zip(objects, np.cumsum([len(c) for c in corners_m]), strict=True):
        mesh = obj.data
        n, faces_n = len(mesh.loops), len(mesh.polygons)
        _write(mesh, "wear_below_m", "CORNER", below_m[at - n : at])
        _write(mesh, "wear_length_m", "FACE", np.full(faces_n, length_m))
        _write(mesh, "wear_steel", "FACE", np.full(faces_n, steel))
        _write(mesh, "wear_still", "FACE", np.full(faces_n, float(still)))


def _write(mesh: bpy.types.Mesh, name: str, domain: str, values: np.ndarray) -> None:
    vector = values.ndim == 2
    kind = "FLOAT_VECTOR" if vector else "FLOAT"
    if old := mesh.attributes.get(name):
        mesh.attributes.remove(old)
    attribute = mesh.attributes.new(name, kind, domain)
    attribute.data.foreach_set(
        "vector" if vector else "value", values.astype(np.float32).ravel()
    )


def unwrap(objects: list[bpy.types.Object]) -> None:
    """One atlas over every mesh, written as the corner attribute UV; their own UV maps
    stay as they were. A shared mesh is split first, so each object has texels of its
    own.

    Args:
        objects: What to lay out; those that are not meshes with faces are skipped.
    """
    meshes = [o for o in objects if o.type == "MESH" and o.data.polygons]
    if not meshes:
        return
    actives = []
    for obj in meshes:
        if obj.data.users > 1:
            obj.data = obj.data.copy()
        layers = obj.data.uv_layers
        actives.append(layers.active.name if layers.active else None)
        layers.active = layers.new(name=UV)
    # Edit mode leaves out a hidden object, whose old UVs would overlap the atlas.
    hidden = [(o, o.hide_get(), o.hide_viewport, o.hide_select) for o in meshes]
    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.hide_set(False)
        obj.hide_viewport = obj.hide_select = False
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.object.mode_set(mode="EDIT")
    try:
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.uv.smart_project(island_margin=0.0, scale_to_bounds=False)
        # Judgement: 0.002 of the atlas; the bake's margin fills the gap.
        bpy.ops.uv.pack_islands(margin=0.002, rotate=True, scale=True)
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")
        for obj, hide, viewport, select in hidden:
            obj.hide_set(hide)
            obj.hide_viewport, obj.hide_select = viewport, select
    for obj, active in zip(meshes, actives, strict=True):
        mesh = obj.data
        layer = mesh.uv_layers[UV]
        uv = faces.floats(layer.data, "uv", 2 * len(mesh.loops)).reshape(-1, 2)
        mesh.uv_layers.remove(layer)
        if active is not None:
            mesh.uv_layers.active = mesh.uv_layers[active]
        # A vector, not a UV map: islands, zones and image nodes read UV maps.
        _write(mesh, UV, "CORNER", np.column_stack([uv, np.zeros(len(uv))]))


def bake_crevice(joined: bpy.types.Object, name: str) -> bpy.types.Image:
    """The crevice mask over an asset, into an image sized to TEXEL_M on its atlas.

    Args:
        joined: Every mesh of the asset as one, in metres, with the atlas `unwrap`
            wrote.
        name: Names the baked image and the bake's throwaway material.
    """
    mesh = joined.data
    n_loops = len(mesh.loops)
    uv = faces.floats(mesh.attributes[UV].data, "vector", 3 * n_loops).reshape(-1, 3)
    layer = mesh.uv_layers.new(name="wear_bake")
    layer.data.foreach_set("uv", uv[:, :2].ravel())
    image = bpy.data.images.new(name, *[_texels(mesh, uv[:, :2])] * 2, is_data=True)

    material = bpy.data.materials.new(name)
    t = material.node_tree
    t.nodes.clear()
    occlusion = node(t, "ShaderNodeAmbientOcclusion", only_local=True)
    occlusion.inputs["Distance"].default_value = CREVICE_M
    crevice = _smooth(1.0 - Num(t, occlusion.outputs["AO"]), 0.05, 0.6)
    emission = node(t, "ShaderNodeEmission", {"Color": crevice})
    node(t, "ShaderNodeOutputMaterial", {"Surface": emission.outputs["Emission"]})
    t.nodes.active = node(t, "ShaderNodeTexImage", image=image)
    mesh.materials.clear()
    mesh.materials.append(material)

    sc = bpy.context.scene
    # On the device the build chose: a switch mid-process compiles kernels again.
    held = sc.render.engine, sc.cycles.samples
    sc.render.engine = "CYCLES"
    # Judgement: 16 samples of the node's 16 rays; a few grey levels of noise.
    sc.cycles.samples = 16
    # Numerics: texels each island is extended by, so a bilinear read at its edge stays
    # on baked texels.
    try:
        with bpy.context.temp_override(
            active_object=joined, selected_objects=[joined], object=joined
        ):
            bpy.ops.object.bake(
                type="EMIT", uv_layer=layer.name, margin=4, margin_type="EXTEND"
            )
    finally:
        sc.render.engine, sc.cycles.samples = held
        bpy.data.materials.remove(material)
    return image


def _texels(mesh: bpy.types.Mesh, uv: np.ndarray) -> int:
    """The square image's side that gives `mesh`'s atlas TEXEL_M."""
    area_m2 = float(faces.floats(mesh.polygons, "area", len(mesh.polygons)).sum())
    mesh.calc_loop_triangles()
    n = len(mesh.loop_triangles)
    t = uv[faces.ints(mesh.loop_triangles, "loops", 3 * n).reshape(-1, 3)]
    ab, ac = t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
    covered = float(np.abs(ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]).sum()) / 2
    side = np.sqrt(area_m2 / max(covered, 1e-12)) / TEXEL_M
    # shortcut: one image, capped at 8k; past that, the texel grows; tile it then.
    return int(np.clip(np.ceil(side), 16, 8192))


def _smooth(x: Num, low: Num | float, high: Num | float) -> Num:
    made = node(
        x.tree,
        "ShaderNodeMapRange",
        {"Value": x, "From Min": low, "From Max": high},
        interpolation_type="SMOOTHSTEP",
    )
    return Num(x.tree, made.outputs["Result"])


class _Dropping(NamedTuple):
    cover: Num
    core: Num  # 1 where its dark faeces are


class _Masks:
    """Judgement, every number: eyeballed against photographs of worn hulls."""

    def __init__(self, tree: bpy.types.NodeTree, crevice: bpy.types.Image) -> None:
        self.tree = t = tree
        self.crevice = crevice
        self.position = attribute(t, "wear_position_m").outputs["Vector"]
        self.x, self.y, z = (Num(t, s) for s in separate(t, self.position))
        # Above the water: a copy lifted off its asset's draught meets it lower.
        self.z = z + Num(t, attribute(t, "rise_m", "OBJECT").outputs["Factor"])
        normal = attribute(t, "wear_normal").outputs["Vector"]
        self.nx, self.ny, self.up = (Num(t, s) for s in separate(t, normal))
        self.below, self.length, self.steel, self.still, stain = (
            Num(t, attribute(t, name).outputs["Factor"])
            for name in (
                "wear_below_m",
                "wear_length_m",
                "wear_steel",
                "wear_still",
                "wear_stain",
            )
        )
        self.amount, self.offset = (
            Num(t, attribute(t, name, "OBJECT").outputs["Factor"])
            for name in ("wear", "wear_offset")
        )
        self.shell = stain.op("GREATER_THAN", 0.0)  # what the sea wets
        self.edge_m = (self.z + self.below).op("MAXIMUM", 0.01)
        # 0 at the waterline, 1 at the edge above.
        self.height = self.z / self.edge_m
        bad = _smooth(self.noise(0.04, 21.0, 2.0), 0.3, 0.7)
        self.region = bad.madd(1.3, 0.2)

    def noise(self, per_m: float, offset: float, detail: float = 6.0) -> Num:
        t = self.tree
        made = node(
            t,
            "ShaderNodeTexNoise",
            {
                "Vector": self.position,
                "Scale": per_m,
                "Detail": detail,
                "W": self.offset + offset,
            },
            noise_dimensions="4D",
        )
        return Num(t, made.outputs["Factor"])

    def cover(self, field: Num, soft: float, boost: Num | float = 0.0) -> Num:
        threshold = (self.amount + boost).madd(-0.45, 0.72)
        return _smooth(field, threshold, threshold + soft)

    def grime(self) -> Num:
        t = self.tree
        uv = attribute(t, UV).outputs["Vector"]
        texture = node(t, "ShaderNodeTexImage", image=self.crevice)
        t.links.new(uv, texture.inputs["Vector"])
        crevice = Num(t, texture.outputs["Color"])  # grey: any channel
        grime = crevice * self.cover(self.noise(0.8, 1.0), 0.3)
        pooled = (
            _smooth(self.up, 0.6, 0.95) * 0.5 * self.cover(self.noise(1.5, 4.0), 0.3)
        )
        return grime.op("MAXIMUM", pooled)

    def streaks(self) -> Num:
        """Along the edge is y on a side, x on a face looking fore or aft."""
        t = self.tree
        side = self.nx.op("ABSOLUTE").op("GREATER_THAN", self.ny.op("ABSOLUTE"))
        along = side * self.y + (1.0 - side) * self.x
        # Judgement: a source every 1/30 of the length, 0.6-4 m apart.
        spacing_m = (self.length / 30.0).op("MAXIMUM", 0.6).op("MINIMUM", 4.0)
        cells = node(t, "ShaderNodeTexVoronoi", voronoi_dimensions="1D")
        feed(t, cells.inputs["W"], along / spacing_m + self.offset * 7.31)
        draw = node(t, "ShaderNodeSeparateColor")
        t.links.new(cells.outputs["Color"], draw.inputs["Color"])
        run, width, live = (Num(t, draw.outputs[k]) for k in ("Red", "Green", "Blue"))
        active = live.op("GREATER_THAN", self.amount.madd(-0.7, 0.85))
        # Judgement: a streak runs a quarter to all of its edge's height above the
        # water, longer at higher wear, at most 8 m, so a turbine's tower does not
        # streak from its nacelle to the sea; it holds for the first half of its run.
        run_m = run.madd(0.75, 0.25) * self.edge_m * self.amount.madd(0.7, 0.3)
        run_m = run_m.op("MINIMUM", 8.0)
        width = width.madd(0.12, 0.03)  # of a cell
        core = _smooth(Num(t, cells.outputs["Distance"]), width * 0.4, width)
        falling = _smooth(self.below, run_m * 0.5, run_m)
        wall = _smooth(self.up.op("ABSOLUTE"), 0.2, 0.5)
        streak = (1.0 - core) * ((1.0 - falling) * (1.0 - wall))
        broken = _smooth(self.noise(2.0, 17.0, 4.0), 0.25, 0.6)
        broken = broken * Num(t, attribute(t, "wear_runoff").outputs["Factor"])
        return streak * active * (broken * self.region)

    def chips(self) -> Num:
        t = self.tree
        bevel = node(t, "ShaderNodeBevel")
        # Judgement: coating fails within 8 cm of an edge.
        bevel.inputs["Radius"].default_value = 0.08
        geometry = node(t, "ShaderNodeNewGeometry")
        facing = vector(
            t, "DOT_PRODUCT", bevel.outputs["Normal"], geometry.outputs["Normal"]
        )
        edge = _smooth(1.0 - Num(t, facing), 0.01, 0.08)
        return edge * self.cover(self.noise(3.0, 7.0), 0.15) * self.region

    def droppings(self) -> "_Dropping":
        # Judgement: splashes 1-3 cm over a few percent of a level top at full wear;
        # only the noise's peaks pass, which `cover`'s broad threshold would not keep.
        threshold = self.amount.madd(-0.08, 0.72)
        field = self.noise(25.0, 9.0, 1.0)
        splash = _smooth(field, threshold, threshold + 0.03)
        # Judgement: the dark faeces sit at the splash's highest peaks, inside the
        # white urate.
        core = _smooth(field, threshold + 0.07, threshold + 0.08)
        return _Dropping(_smooth(self.up, 0.5, 0.9) * splash * self.region, core)

    def stain(self) -> Num:
        bow = _smooth(self.y / self.length, 0.32, 0.45)
        # Judgement: a waterline stain 0.8% of the length tall on GRP, a hand's width
        # on a yacht; 1.8% on steel, capped at a 1.2 m boot top; higher at the bow.
        tall = self.steel.madd(0.01, 0.008)
        top_m = (self.length * tall).op("MINIMUM", 1.2) * bow.madd(0.8, 1.0)
        patchy = self.noise(0.5, 5.0, 2.0)
        top_m = top_m * patchy.madd(0.8, 0.6)
        # Judgement: it reaches 0.3 m under the still waterline.
        band = _smooth(self.z, -0.3, 0.0) * (1.0 - _smooth(self.z, top_m * 0.5, top_m))
        return band * self.cover(patchy, 0.3, 0.3)

    def fouling(self) -> Num:
        """Weed and slime from the keel up into the splash zone. A hull under way
        polishes it off, so on one it needs high wear and stays near the waterline.
        """
        # Judgement: a third of the freeboard on a still asset, a tenth on a hull, at
        # most 2.5 m before the patches stretch it by up to 1.4; a still asset covers
        # at 0.3 less wear, a hull at 0.3 more.
        reach = self.still.madd(0.23, 0.1)
        top_m = (self.edge_m * reach).op("MINIMUM", 2.5)
        patchy = self.noise(2.0, 23.0, 4.0)
        top_m = top_m * patchy.madd(0.8, 0.6)
        band = 1.0 - _smooth(self.z, top_m * 0.5, top_m)
        return band * self.cover(patchy, 0.3, self.still.madd(0.6, -0.3))

    def layers(self) -> dict[str, Num]:
        """Each of `LAYERS`' masks, and `CORE`, by name."""
        streak = self.streaks()
        not_steel = 1.0 - self.steel
        grime = self.grime().op("MAXIMUM", streak * not_steel)
        washed = self.shell * (1.0 - _smooth(self.height, 0.0, 0.5))
        grime = grime * (1.0 - washed)
        chips = self.chips()
        rust = chips.op("MAXIMUM", streak * self.steel)
        soot = self.cover(self.noise(0.5, 13.0), 0.4, 0.3)
        sky = _smooth(self.up, 0.0, 1.0).madd(0.5, 0.5)
        dropping = self.droppings()
        return {
            "grime": grime,
            "rust": rust,
            "stain": self.stain(),
            "fouling": self.fouling(),
            "soot": soot,
            "scuff": chips,
            "dropping": dropping.cover,
            "fade": self.amount * sky,
            CORE: dropping.core,
        }


def masks(crevice: bpy.types.Image) -> bpy.types.NodeTree:
    """Each layer's cover, 0-1, for the asset whose crevices `crevice` holds."""
    name = f"wear_masks/{crevice.name}"
    if (group := bpy.data.node_groups.get(name)) is not None:
        return group
    group = bpy.data.node_groups.new(name, "ShaderNodeTree")
    for layer in (*LAYERS, CORE):
        group.interface.new_socket(
            layer, in_out="OUTPUT", socket_type="NodeSocketFloat"
        )
    g = _Masks(group, crevice)
    out = node(group, "NodeGroupOutput")
    # An unworn copy shares its worn twin's import and materials: no cover at all.
    worn = g.amount.op("GREATER_THAN", 0.0)
    layers = g.layers()
    for layer in LAYERS:
        mask = layers[layer]
        weight = Num(group, attribute(group, f"wear_{layer}").outputs["Factor"]) * worn
        feed(group, out.inputs[layer], clamped(group, (mask * weight).socket))
    # Only where a dropping covers, so it needs no weight of its own.
    feed(group, out.inputs[CORE], layers[CORE].socket)
    return group


def mark(anchor: bpy.types.Object, amount: float, rng: np.random.Generator) -> None:
    """Give a hull's meshes their wear and a noise offset of their own, so copies of
    one asset wear apart.
    """
    offset = float(rng.uniform(0.0, 1000.0))
    for part in mesh_objects([anchor]):
        part["wear"] = amount
        part["wear_offset"] = offset


# Judgement: linear albedo of each layer, from photographs of worn hulls.
GRIME = (0.045, 0.038, 0.030, 1.0)
RUST = (0.20, 0.065, 0.022, 1.0)
STAIN_STEEL = (0.06, 0.07, 0.035, 1.0)  # algae on a ship's boot top
STAIN = (0.22, 0.14, 0.05, 1.0)  # a yacht's: tannin, oil and algae
SOOT = (0.015, 0.014, 0.013, 1.0)
# Judgement: abraded paint and stress-whitened plastic.
SCUFF = (0.42, 0.41, 0.38, 1.0)
# Judgement: bird lime, dried.
DROPPING = (0.62, 0.61, 0.57, 1.0)
# Judgement: the faeces in its middle, dried.
DROPPING_CORE = (0.05, 0.045, 0.03, 1.0)
# Judgement: green weed over brown diatom slime, wet.
FOULING = (0.035, 0.06, 0.018, 1.0)


def dirty(parts: list[bpy.types.Object]) -> None:
    """Dirty every material of `parts` by their asset's baked crevices."""
    meshes = mesh_objects(parts)
    crevice = meshes[0][CREVICE]
    for material in {s.material for o in meshes for s in o.material_slots} - {None}:
        _dirty(material, crevice)


def _dirty(material: bpy.types.Material, crevice: bpy.types.Image) -> None:
    """A second dielectric BSDF, not a tinted one: tinting a metallic surface colours
    its mirror (rust reads as copper), and mixing the one BSDF's inputs would need
    Metallic, Roughness and Coat mixed in every material.
    """
    t = material.node_tree
    shader = faces.bsdf(material)
    if shader is None:
        return
    group = node(t, "ShaderNodeGroup", node_tree=masks(crevice))
    cover = {layer: Num(t, group.outputs[layer]) for layer in LAYERS}

    # Paint fades; metal does not.
    paint = scalar(t, "SUBTRACT", 1.0, 0.0)
    feed(t, paint.node.inputs[1], source(shader.inputs["Metallic"]))
    fade = cover["fade"] * Num(t, paint)
    faded = node(t, "ShaderNodeHueSaturation")
    feed(t, faded.inputs["Color"], source(shader.inputs["Base Color"]))
    # Judgement: full fade halves saturation and lifts value 15%.
    feed(t, faded.inputs["Saturation"], fade.madd(-0.5, 1.0))
    feed(t, faded.inputs["Value"], fade.madd(0.15, 1.0))
    t.links.new(faded.outputs["Color"], shader.inputs["Base Color"])

    dirt = node(t, "ShaderNodeBsdfPrincipled")
    stain = node(t, "ShaderNodeMix", data_type="RGBA")
    feed(t, stain.inputs["Factor"], attribute(t, "wear_steel").outputs["Factor"])
    stain.inputs["A"].default_value = STAIN
    stain.inputs["B"].default_value = STAIN_STEEL
    dropping = node(t, "ShaderNodeMix", data_type="RGBA")
    feed(t, dropping.inputs["Factor"], group.outputs[CORE])
    dropping.inputs["A"].default_value = DROPPING
    dropping.inputs["B"].default_value = DROPPING_CORE
    layered: bpy.types.NodeSocket | tuple[float, ...] = GRIME
    for layer, albedo in zip(
        COVERS[1:],
        (
            RUST,
            stain.outputs["Result"],
            FOULING,
            SOOT,
            SCUFF,
            dropping.outputs["Result"],
        ),
        strict=True,
    ):
        mix = node(t, "ShaderNodeMix", data_type="RGBA")
        feed(t, mix.inputs["Factor"], cover[layer])
        feed(t, mix.inputs["A"], layered)
        feed(t, mix.inputs["B"], albedo)
        layered = mix.outputs["Result"]
    feed(t, dirt.inputs["Base Color"], layered)
    # Judgement: dry grime and rust are matte; wet fouling shines.
    feed(t, dirt.inputs["Roughness"], cover["fouling"].madd(-0.6, 0.9))
    # Glass and decals stay as clear, and lamps as bright, under their dirt.
    for name in ("Alpha", "Normal", "Emission Color", "Emission Strength"):
        if name != "Normal" or shader.inputs[name].is_linked:
            feed(t, dirt.inputs[name], source(shader.inputs[name]))

    total = cover["grime"]
    for layer in COVERS[1:]:
        total = total.op("MAXIMUM", cover[layer])
    consumers = [link.to_socket for link in shader.outputs["BSDF"].links]
    mixed = mix_shader(t, total, shader.outputs["BSDF"], dirt.outputs["BSDF"])
    for socket in consumers:
        t.links.new(mixed, socket)
