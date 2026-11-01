"""Wear attributes written at import, in the fitted hull frame: metres, bow +Y,
waterline z = 0."""

from typing import NamedTuple

import bpy
import numpy as np

from seascape.assets import Material
from seascape.blender import faces
from seascape.blender.blend import clamped, feed, scalar, vector
from seascape.physics.thermal import ZONE_NAMES, ZoneName


class Weights(NamedTuple):
    """How much of each layer a zone collects, and whether runoff streaks it."""

    grime: float
    rust: float
    stain: float
    soot: float
    fade: float
    runoff: float


# The layers the masks cover; `runoff` only gates streaks.
LAYERS = Weights._fields[:-1]

# Judgement, all of it. A stain is the waterline's, so only what the sea wets takes
# one; cloth sags and flexes, so water runs off it without leaving a streak.
PROFILE: dict[ZoneName, Weights] = {
    "paint_hull": Weights(1.0, 1.0, 1.0, 0.0, 0.5, 1.0),
    "paint_superstructure": Weights(0.8, 0.7, 0.0, 0.0, 0.5, 1.0),
    "deck": Weights(1.0, 0.6, 0.0, 0.0, 0.3, 1.0),
    "glass": Weights(0.35, 0.0, 0.0, 0.0, 0.0, 1.0),
    "bare_metal": Weights(0.6, 0.6, 0.0, 0.0, 0.0, 1.0),
    "rubber": Weights(0.6, 0.0, 0.6, 0.0, 0.7, 1.0),
    "fabric": Weights(0.8, 0.0, 0.0, 0.0, 0.7, 0.0),
    "funnel": Weights(0.5, 0.5, 0.0, 1.0, 0.3, 1.0),
    "exhaust": Weights(0.4, 0.0, 0.0, 1.0, 0.0, 1.0),
    "engine_cowl": Weights(0.5, 0.0, 0.0, 0.0, 0.7, 1.0),
    "wood_composite": Weights(0.8, 0.0, 1.0, 0.0, 0.6, 1.0),
}


def weights(material: Material) -> np.ndarray:
    """(zones, weights), zones in vocabulary order."""
    table = np.array([PROFILE[z] for z in ZONE_NAMES], dtype=np.float32)
    if material != "steel":
        # Judgement: other hulls' fittings are stainless.
        table[:, Weights._fields.index("rust")] = 0.0
    return table


def below_edge_m(position_m: np.ndarray, zone: np.ndarray) -> np.ndarray:
    """Each point's depth below the highest point of its zone over the same patch of
    plan: for a hull side the sheer, for a house wall its roof edge."""
    # shortcut: a plan grid, not a walk up the surface; a tier stepped back by less
    # than a patch reads its upper tier's edge.
    length_m = float(np.ptp(position_m[:, :2], axis=0).max())
    # Judgement: patches 1/200 of the length, no less than 10 cm, each reaching its
    # eight neighbours, so a side flared out by a patch still finds its sheer.
    patch = np.floor(position_m[:, :2] / max(length_m / 200.0, 0.1)).astype(np.int64)
    # An empty border, so np.roll never wraps a patch into the plan.
    patch -= patch.min(axis=0) - 1
    rows, cols = patch.max(axis=0) + 2
    cell = (zone * rows + patch[:, 0]) * cols + patch[:, 1]
    order = np.argsort(cell, kind="stable")
    first = np.flatnonzero(np.diff(cell[order], prepend=-1))
    top = np.full((zone.max() + 1) * rows * cols, -np.inf)
    top[cell[order][first]] = np.maximum.reduceat(position_m[order, 2], first)
    top = top.reshape(-1, rows, cols)
    reach = top.copy()
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            np.maximum(reach, np.roll(top, (di, dj), axis=(1, 2)), out=reach)
    return reach.ravel()[cell] - position_m[:, 2]


def tag(
    objects: list[bpy.types.Object], zones: list[np.ndarray], material: Material
) -> None:
    """`objects` already fitted; a skinned mesh is read as posed."""
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


def _write(mesh: bpy.types.Mesh, name: str, domain: str, values: np.ndarray) -> None:
    vector = values.ndim == 2
    kind = "FLOAT_VECTOR" if vector else "FLOAT"
    if old := mesh.attributes.get(name):
        mesh.attributes.remove(old)
    attribute = mesh.attributes.new(name, kind, domain)
    attribute.data.foreach_set(
        "vector" if vector else "value", values.astype(np.float32).ravel()
    )


def _node(tree: bpy.types.NodeTree, kind: str, **props: object) -> bpy.types.Node:
    node = tree.nodes.new(kind)
    for key, value in props.items():
        setattr(node, key, value)
    return node


def _smooth(
    tree: bpy.types.NodeTree,
    x: bpy.types.NodeSocket | float,
    low: bpy.types.NodeSocket | float,
    high: bpy.types.NodeSocket | float,
) -> bpy.types.NodeSocket:
    node = _node(tree, "ShaderNodeMapRange", interpolation_type="SMOOTHSTEP")
    for name, value in (("Value", x), ("From Min", low), ("From Max", high)):
        feed(tree, node.inputs[name], value)
    return node.outputs["Result"]


def _attribute(
    tree: bpy.types.NodeTree, name: str, kind: str = "GEOMETRY"
) -> bpy.types.Node:
    return _node(tree, "ShaderNodeAttribute", attribute_type=kind, attribute_name=name)


def _xyz(
    tree: bpy.types.NodeTree, vector: bpy.types.NodeSocket
) -> list[bpy.types.NodeSocket]:
    node = _node(tree, "ShaderNodeSeparateXYZ")
    tree.links.new(vector, node.inputs["Vector"])
    return [node.outputs[k] for k in "XYZ"]


class _Masks:
    """Judgement, every number: eyeballed against photographs of worn hulls."""

    def __init__(self, tree: bpy.types.NodeTree) -> None:
        self.tree = t = tree
        self.position = _attribute(t, "wear_position_m").outputs["Vector"]
        self.x, self.y, self.z = _xyz(t, self.position)
        normal = _attribute(t, "wear_normal").outputs["Vector"]
        self.nx, self.ny, self.up = _xyz(t, normal)
        self.below, self.length, self.steel, stain = (
            _attribute(t, name).outputs["Factor"]
            for name in ("wear_below_m", "wear_length_m", "wear_steel", "wear_stain")
        )
        self.amount, self.offset = (
            _attribute(t, name, "OBJECT").outputs["Factor"]
            for name in ("wear", "wear_offset")
        )
        self.shell = scalar(t, "GREATER_THAN", stain, 0.0)  # what the sea wets
        freeboard_m = scalar(t, "MAXIMUM", scalar(t, "ADD", self.z, self.below), 0.01)
        # 0 at the waterline, 1 at the edge above.
        self.height = scalar(t, "DIVIDE", self.z, freeboard_m)
        bad = _smooth(t, self.noise(0.04, 21.0, 2.0), 0.3, 0.7)
        self.region = scalar(t, "MULTIPLY_ADD", bad, 1.3, 0.2)

    def noise(
        self,
        per_m: float,
        offset: float,
        detail: float = 6.0,
        stretch: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> bpy.types.NodeSocket:
        """`stretch` scales the features per axis."""
        t = self.tree
        node = _node(t, "ShaderNodeTexNoise", noise_dimensions="4D")
        feed(t, node.inputs["Vector"], vector(t, "MULTIPLY", self.position, stretch))
        node.inputs["Scale"].default_value = per_m
        node.inputs["Detail"].default_value = detail
        feed(t, node.inputs["W"], scalar(t, "ADD", self.offset, offset))
        return node.outputs["Factor"]

    def cover(
        self, field: bpy.types.NodeSocket, soft: float, boost: float = 0.0
    ) -> bpy.types.NodeSocket:
        t = self.tree
        worse = scalar(t, "ADD", self.amount, boost)
        threshold = scalar(t, "MULTIPLY_ADD", worse, -0.45, 0.72)
        return _smooth(t, field, threshold, scalar(t, "ADD", threshold, soft))

    def grime(self) -> bpy.types.NodeSocket:
        t = self.tree
        occlusion = _node(t, "ShaderNodeAmbientOcclusion", only_local=True)
        # Judgement: a crevice is what lies within 0.6 m of other surfaces.
        occlusion.inputs["Distance"].default_value = 0.6
        crevice = _smooth(
            t, scalar(t, "SUBTRACT", 1.0, occlusion.outputs["AO"]), 0.05, 0.6
        )
        grime = scalar(t, "MULTIPLY", crevice, self.cover(self.noise(0.8, 1.0), 0.3))
        pooled = scalar(t, "MULTIPLY", _smooth(t, self.up, 0.6, 0.95), 0.5)
        pooled = scalar(t, "MULTIPLY", pooled, self.cover(self.noise(1.5, 4.0), 0.3))
        return scalar(t, "MAXIMUM", grime, pooled)

    def streaks(self) -> bpy.types.NodeSocket:
        """Along the edge is y on a side, x on a face looking fore or aft."""
        t = self.tree
        side = scalar(
            t,
            "GREATER_THAN",
            scalar(t, "ABSOLUTE", self.nx),
            scalar(t, "ABSOLUTE", self.ny),
        )
        along = scalar(
            t,
            "ADD",
            scalar(t, "MULTIPLY", side, self.y),
            scalar(t, "MULTIPLY", scalar(t, "SUBTRACT", 1.0, side), self.x),
        )
        # Judgement: a source every 1/30 of the length, 0.6-4 m apart.
        spacing_m = scalar(t, "DIVIDE", self.length, 30.0)
        spacing_m = scalar(t, "MINIMUM", scalar(t, "MAXIMUM", spacing_m, 0.6), 4.0)
        cells = _node(t, "ShaderNodeTexVoronoi", voronoi_dimensions="1D")
        feed(
            t,
            cells.inputs["W"],
            scalar(
                t,
                "ADD",
                scalar(t, "DIVIDE", along, spacing_m),
                scalar(t, "MULTIPLY", self.offset, 7.31),
            ),
        )
        draw = _node(t, "ShaderNodeSeparateColor")
        t.links.new(cells.outputs["Color"], draw.inputs["Color"])
        run, width, live = (draw.outputs[k] for k in ("Red", "Green", "Blue"))
        active = scalar(
            t, "GREATER_THAN", live, scalar(t, "MULTIPLY_ADD", self.amount, -0.7, 0.85)
        )
        run_m = scalar(
            t, "MULTIPLY", scalar(t, "MULTIPLY_ADD", run, 0.9, 0.1), self.length
        )
        run_m = scalar(
            t, "MULTIPLY", run_m, scalar(t, "MULTIPLY_ADD", self.amount, 0.05, 0.02)
        )
        width = scalar(t, "MULTIPLY_ADD", width, 0.12, 0.03)  # of a cell
        core = _smooth(
            t, cells.outputs["Distance"], scalar(t, "MULTIPLY", width, 0.4), width
        )
        falling = _smooth(t, self.below, scalar(t, "MULTIPLY", run_m, 0.2), run_m)
        wall = _smooth(t, scalar(t, "ABSOLUTE", self.up), 0.2, 0.5)
        streak = scalar(
            t,
            "MULTIPLY",
            scalar(t, "SUBTRACT", 1.0, core),
            scalar(
                t,
                "MULTIPLY",
                scalar(t, "SUBTRACT", 1.0, falling),
                scalar(t, "SUBTRACT", 1.0, wall),
            ),
        )
        broken = _smooth(t, self.noise(2.0, 17.0, 4.0), 0.25, 0.6)
        runoff = _attribute(t, "wear_runoff").outputs["Factor"]
        broken = scalar(t, "MULTIPLY", broken, runoff)
        return scalar(
            t,
            "MULTIPLY",
            scalar(t, "MULTIPLY", streak, active),
            scalar(t, "MULTIPLY", broken, self.region),
        )

    def chips(self) -> bpy.types.NodeSocket:
        t = self.tree
        bevel = _node(t, "ShaderNodeBevel")
        # Judgement: coating fails within 8 cm of an edge.
        bevel.inputs["Radius"].default_value = 0.08
        geometry = _node(t, "ShaderNodeNewGeometry")
        facing = vector(
            t, "DOT_PRODUCT", bevel.outputs["Normal"], geometry.outputs["Normal"]
        )
        edge = _smooth(t, scalar(t, "SUBTRACT", 1.0, facing), 0.01, 0.08)
        chipped = scalar(t, "MULTIPLY", edge, self.cover(self.noise(3.0, 7.0), 0.15))
        return scalar(t, "MULTIPLY", chipped, self.region)

    def stain(self) -> bpy.types.NodeSocket:
        t = self.tree
        bow = _smooth(t, scalar(t, "DIVIDE", self.y, self.length), 0.32, 0.45)
        # Judgement: a waterline stain 0.8% of the length tall on GRP, a hand's width
        # on a yacht; 1.8% on steel, capped at a 1.2 m boot top; higher at the bow.
        tall = scalar(t, "MULTIPLY_ADD", self.steel, 0.01, 0.008)
        top_m = scalar(t, "MULTIPLY", self.length, tall)
        top_m = scalar(
            t,
            "MULTIPLY",
            scalar(t, "MINIMUM", top_m, 1.2),
            scalar(t, "MULTIPLY_ADD", bow, 0.8, 1.0),
        )
        patchy = self.noise(0.5, 5.0, 2.0)
        top_m = scalar(
            t, "MULTIPLY", top_m, scalar(t, "MULTIPLY_ADD", patchy, 0.8, 0.6)
        )
        # Judgement: it reaches 0.3 m under the still waterline.
        band = scalar(
            t,
            "MULTIPLY",
            _smooth(t, self.z, -0.3, 0.0),
            scalar(
                t,
                "SUBTRACT",
                1.0,
                _smooth(t, self.z, scalar(t, "MULTIPLY", top_m, 0.5), top_m),
            ),
        )
        return scalar(t, "MULTIPLY", band, self.cover(patchy, 0.3, 0.3))

    def layers(self) -> tuple[bpy.types.NodeSocket, ...]:
        t = self.tree
        streak = self.streaks()
        not_steel = scalar(t, "SUBTRACT", 1.0, self.steel)
        grime = scalar(
            t, "MAXIMUM", self.grime(), scalar(t, "MULTIPLY", streak, not_steel)
        )
        washed = scalar(
            t,
            "MULTIPLY",
            self.shell,
            scalar(t, "SUBTRACT", 1.0, _smooth(t, self.height, 0.0, 0.5)),
        )
        grime = scalar(t, "MULTIPLY", grime, scalar(t, "SUBTRACT", 1.0, washed))
        rust = scalar(
            t, "MAXIMUM", self.chips(), scalar(t, "MULTIPLY", streak, self.steel)
        )
        soot = self.cover(self.noise(0.5, 13.0), 0.4, 0.3)
        sky = scalar(t, "MULTIPLY_ADD", _smooth(t, self.up, 0.0, 1.0), 0.5, 0.5)
        return grime, rust, self.stain(), soot, scalar(t, "MULTIPLY", self.amount, sky)


def masks() -> bpy.types.NodeTree:
    """Each layer's cover, 0-1."""
    if (group := bpy.data.node_groups.get("wear_masks")) is not None:
        return group
    group = bpy.data.node_groups.new("wear_masks", "ShaderNodeTree")
    for layer in LAYERS:
        group.interface.new_socket(
            layer, in_out="OUTPUT", socket_type="NodeSocketFloat"
        )
    g = _Masks(group)
    out = _node(group, "NodeGroupOutput")
    # An unworn copy shares its worn twin's import and materials: no cover at all.
    worn = scalar(group, "GREATER_THAN", g.amount, 0.0)
    for layer, mask in zip(LAYERS, g.layers(), strict=True):
        weight = _attribute(group, f"wear_{layer}").outputs["Factor"]
        weight = scalar(group, "MULTIPLY", weight, worn)
        feed(
            group,
            out.inputs[layer],
            clamped(group, scalar(group, "MULTIPLY", mask, weight)),
        )
    return group


def mark(anchor: bpy.types.Object, amount: float, rng: np.random.Generator) -> None:
    """Give a hull's meshes their wear and a noise offset of their own, so copies of
    one asset wear apart."""
    offset = float(rng.uniform(0.0, 1000.0))
    for part in anchor.children_recursive:
        if part.type == "MESH":
            part["wear"] = amount
            part["wear_offset"] = offset


# Judgement: linear albedo of each layer, from photographs of worn hulls.
GRIME = (0.045, 0.038, 0.030, 1.0)
RUST = (0.20, 0.065, 0.022, 1.0)
STAIN_STEEL = (0.06, 0.07, 0.035, 1.0)  # algae on a ship's boot top
STAIN = (0.22, 0.14, 0.05, 1.0)  # a yacht's: tannin, oil and algae
SOOT = (0.015, 0.014, 0.013, 1.0)


def dirty(parts: list[bpy.types.Object]) -> None:
    meshes = [o for p in parts for o in (p, *p.children_recursive) if o.type == "MESH"]
    for material in {s.material for o in meshes for s in o.material_slots} - {None}:
        _dirty(material)


def _source(
    socket: bpy.types.NodeSocket,
) -> bpy.types.NodeSocket | float | tuple[float, ...]:
    """What feeds `socket`: its link, or else its own value."""
    if socket.is_linked:
        return socket.links[0].from_socket
    value = socket.default_value
    return value if isinstance(value, float) else tuple(value)


def _dirty(material: bpy.types.Material) -> None:
    """A second dielectric BSDF, not a tinted one: tinting a metallic surface colours
    its mirror (rust reads as copper), and mixing the one BSDF's inputs would need
    Metallic, Roughness and Coat mixed in every material."""
    t = material.node_tree
    shader = next((n for n in t.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if shader is None:
        return
    group = _node(t, "ShaderNodeGroup", node_tree=masks())
    cover = {layer: group.outputs[layer] for layer in LAYERS}

    # Paint fades; metal does not.
    paint = scalar(t, "SUBTRACT", 1.0, 0.0)
    feed(t, paint.node.inputs[1], _source(shader.inputs["Metallic"]))
    fade = scalar(t, "MULTIPLY", cover["fade"], paint)
    faded = _node(t, "ShaderNodeHueSaturation")
    feed(t, faded.inputs["Color"], _source(shader.inputs["Base Color"]))
    # Judgement: full fade halves saturation and lifts value 15%.
    feed(t, faded.inputs["Saturation"], scalar(t, "MULTIPLY_ADD", fade, -0.5, 1.0))
    feed(t, faded.inputs["Value"], scalar(t, "MULTIPLY_ADD", fade, 0.15, 1.0))
    t.links.new(faded.outputs["Color"], shader.inputs["Base Color"])

    dirt = _node(t, "ShaderNodeBsdfPrincipled")
    stain = _node(t, "ShaderNodeMix", data_type="RGBA")
    feed(t, stain.inputs["Factor"], _attribute(t, "wear_steel").outputs["Factor"])
    stain.inputs["A"].default_value = STAIN
    stain.inputs["B"].default_value = STAIN_STEEL
    layered: bpy.types.NodeSocket | tuple[float, ...] = GRIME
    for layer, albedo in (
        ("rust", RUST),
        ("stain", stain.outputs["Result"]),
        ("soot", SOOT),
    ):
        mix = _node(t, "ShaderNodeMix", data_type="RGBA")
        feed(t, mix.inputs["Factor"], cover[layer])
        feed(t, mix.inputs["A"], layered)
        feed(t, mix.inputs["B"], albedo)
        layered = mix.outputs["Result"]
    feed(t, dirt.inputs["Base Color"], layered)
    # Judgement: dry grime and rust are matte.
    dirt.inputs["Roughness"].default_value = 0.9
    # Glass and decals stay as clear, and lamps as bright, under their dirt.
    for name in ("Alpha", "Normal", "Emission Color", "Emission Strength"):
        if name != "Normal" or shader.inputs[name].is_linked:
            feed(t, dirt.inputs[name], _source(shader.inputs[name]))

    total = scalar(
        t,
        "MAXIMUM",
        scalar(t, "MAXIMUM", cover["grime"], cover["rust"]),
        scalar(t, "MAXIMUM", cover["stain"], cover["soot"]),
    )
    consumers = [link.to_socket for link in shader.outputs["BSDF"].links]
    mix = _node(t, "ShaderNodeMixShader")
    feed(t, mix.inputs["Factor"], total)
    t.links.new(shader.outputs["BSDF"], mix.inputs["Shader"])
    t.links.new(dirt.outputs["BSDF"], mix.inputs["Shader_001"])
    for socket in consumers:
        t.links.new(mix.outputs["Shader"], socket)
