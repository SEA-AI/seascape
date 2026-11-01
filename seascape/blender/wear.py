"""Wear attributes written at import, in the fitted hull frame: metres, bow +Y,
waterline z = 0."""

from typing import NamedTuple

import bpy
import numpy as np

from seascape.assets import Material, manifest
from seascape.blender import faces
from seascape.blender.blend import clamped, scalar, vector
from seascape.config import substream
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
    patch -= (
        patch.min(axis=0) - 1
    )  # an empty border, so np.roll never wraps a patch into the plan
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
    below_m = below_edge_m(np.concatenate(corners_m), np.concatenate(corner_zones))
    for obj, at in zip(objects, np.cumsum([len(c) for c in corners_m]), strict=True):
        n = len(obj.data.loops)
        _write(obj.data, "wear_below_m", "CORNER", below_m[at - n : at])


def _write(mesh: bpy.types.Mesh, name: str, domain: str, values: np.ndarray) -> None:
    vector = values.ndim == 2
    kind = "FLOAT_VECTOR" if vector else "FLOAT"
    attribute = mesh.attributes.get(name)
    if attribute is not None and (attribute.data_type, attribute.domain) != (
        kind,
        domain,
    ):
        mesh.attributes.remove(attribute)
        attribute = None
    if attribute is None:
        attribute = mesh.attributes.new(name, kind, domain)
    attribute.data.foreach_set(
        "vector" if vector else "value", values.astype(np.float32).ravel()
    )


type Socket = bpy.types.NodeSocket | float


class _Graph:
    """Nodes on one tree, each input linked from a socket or set to a number."""

    def __init__(self, tree: bpy.types.NodeTree) -> None:
        self.tree = tree

    def node(self, kind: str, **props: object) -> bpy.types.Node:
        node = self.tree.nodes.new(kind)
        for key, value in props.items():
            setattr(node, key, value)
        return node

    def put(self, socket: bpy.types.NodeSocket, value: object) -> None:
        if isinstance(value, bpy.types.NodeSocket):
            self.tree.links.new(value, socket)
        else:
            socket.default_value = value

    def m(self, operation: str, *inputs: Socket) -> bpy.types.NodeSocket:
        return scalar(self.tree, operation, *inputs)

    def smooth(self, x: Socket, low: Socket, high: Socket) -> bpy.types.NodeSocket:
        """0 below `low`, 1 above `high`, smoothstep between."""
        node = self.node("ShaderNodeMapRange", interpolation_type="SMOOTHSTEP")
        for name, value in (("Value", x), ("From Min", low), ("From Max", high)):
            self.put(node.inputs[name], value)
        return node.outputs["Result"]

    def attribute(self, name: str, kind: str = "GEOMETRY") -> bpy.types.Node:
        return self.node(
            "ShaderNodeAttribute", attribute_type=kind, attribute_name=name
        )

    def xyz(self, vector: bpy.types.NodeSocket) -> list[bpy.types.NodeSocket]:
        node = self.node("ShaderNodeSeparateXYZ")
        self.tree.links.new(vector, node.inputs["Vector"])
        return [node.outputs[k] for k in "XYZ"]


class _Masks(_Graph):
    """The `wear_masks` group: what `tag` wrote and the object's own wear, and a
    method per layer. Judgement, every number: shapes eyeballed against photographs
    of worn hulls."""

    def __init__(self, tree: bpy.types.NodeTree) -> None:
        super().__init__(tree)
        self.position = self.attribute("wear_position_m").outputs["Vector"]
        self.x, self.y, self.z = self.xyz(self.position)
        normal = self.attribute("wear_normal").outputs["Vector"]
        self.nx, self.ny, self.up = self.xyz(normal)
        self.below = self.attribute("wear_below_m").outputs["Factor"]
        self.amount, self.seed, self.length, self.steel = (
            self.attribute(name, "OBJECT").outputs["Factor"]
            for name in ("wear", "wear_seed", "wear_length_m", "wear_steel")
        )
        m = self.m
        stain = self.attribute("wear_stain").outputs["Factor"]
        self.shell = m("GREATER_THAN", stain, 0.0)  # what the sea wets
        freeboard_m = m("MAXIMUM", m("ADD", self.z, self.below), 0.01)
        # 0 at the waterline, 1 at the edge above.
        self.height = m("DIVIDE", self.z, freeboard_m)
        # Some plates and panels are far worse than others.
        self.region = m(
            "MULTIPLY_ADD", self.smooth(self.noise(0.04, 21.0, 2.0), 0.3, 0.7), 1.3, 0.2
        )

    def noise(
        self,
        per_m: float,
        offset: float,
        detail: float = 6.0,
        stretch: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> Socket:
        """A fractal noise, its features `per_m` to a metre, `stretch` times as many
        along each axis."""
        node = self.node("ShaderNodeTexNoise", noise_dimensions="4D")
        self.put(
            node.inputs["Vector"], vector(self.tree, "MULTIPLY", self.position, stretch)
        )
        node.inputs["Scale"].default_value = per_m
        node.inputs["Detail"].default_value = detail
        self.put(node.inputs["W"], self.m("ADD", self.seed, offset))
        return node.outputs["Factor"]

    def cover(self, field: Socket, soft: float, boost: float = 0.0) -> Socket:
        """`field` past a threshold that falls as wear rises."""
        m = self.m
        threshold = m("MULTIPLY_ADD", m("ADD", self.amount, boost), -0.45, 0.72)
        return self.smooth(field, threshold, m("ADD", threshold, soft))

    def grime(self) -> Socket:
        """In crevices, a thin film, pooled on what faces up."""
        m, smooth = self.m, self.smooth
        occlusion = self.node("ShaderNodeAmbientOcclusion", only_local=True)
        occlusion.inputs["Distance"].default_value = 0.6
        crevice = smooth(m("SUBTRACT", 1.0, occlusion.outputs["AO"]), 0.05, 0.6)
        grime = m("MULTIPLY", crevice, self.cover(self.noise(0.8, 1.0), 0.3))
        film = m("MULTIPLY", self.cover(self.noise(0.15, 3.0), 0.3, -0.2), 0.35)
        pooled = m("MULTIPLY", smooth(self.up, 0.6, 0.95), 0.5)
        pooled = m("MULTIPLY", pooled, self.cover(self.noise(1.5, 4.0), 0.3))
        return m("MAXIMUM", m("MAXIMUM", grime, film), pooled)

    def streaks(self) -> Socket:
        """Runoff from the edge above: one source per cell along it, each its own
        length and width, most dormant on a lightly worn hull. Along the edge is y
        on a side, x on a face looking fore or aft."""
        m, smooth = self.m, self.smooth
        side = m("GREATER_THAN", m("ABSOLUTE", self.nx), m("ABSOLUTE", self.ny))
        along = m(
            "ADD",
            m("MULTIPLY", side, self.y),
            m("MULTIPLY", m("SUBTRACT", 1.0, side), self.x),
        )
        spacing_m = m("MINIMUM", m("MAXIMUM", m("DIVIDE", self.length, 30.0), 0.6), 4.0)
        cells = self.node("ShaderNodeTexVoronoi", voronoi_dimensions="1D")
        self.put(
            cells.inputs["W"],
            m("ADD", m("DIVIDE", along, spacing_m), m("MULTIPLY", self.seed, 7.31)),
        )
        draw = self.node("ShaderNodeSeparateColor")
        self.tree.links.new(cells.outputs["Color"], draw.inputs["Color"])
        run, width, live = (draw.outputs[k] for k in ("Red", "Green", "Blue"))
        active = m("GREATER_THAN", live, m("MULTIPLY_ADD", self.amount, -0.7, 0.85))
        run_m = m("MULTIPLY", m("MULTIPLY_ADD", run, 0.9, 0.1), self.length)
        run_m = m("MULTIPLY", run_m, m("MULTIPLY_ADD", self.amount, 0.05, 0.02))
        width = m("MULTIPLY_ADD", width, 0.12, 0.03)  # of a cell
        core = smooth(cells.outputs["Distance"], m("MULTIPLY", width, 0.4), width)
        falling = smooth(self.below, m("MULTIPLY", run_m, 0.2), run_m)
        wall = smooth(m("ABSOLUTE", self.up), 0.2, 0.5)
        streak = m(
            "MULTIPLY",
            m("SUBTRACT", 1.0, core),
            m("MULTIPLY", m("SUBTRACT", 1.0, falling), m("SUBTRACT", 1.0, wall)),
        )
        broken = smooth(self.noise(2.0, 17.0, 4.0), 0.25, 0.6)
        broken = m("MULTIPLY", broken, self.attribute("wear_runoff").outputs["Factor"])
        return m(
            "MULTIPLY",
            m("MULTIPLY", streak, active),
            m("MULTIPLY", broken, self.region),
        )

    def chips(self) -> Socket:
        """Coating failing at edges, worse in bad regions."""
        m = self.m
        bevel = self.node("ShaderNodeBevel")
        bevel.inputs["Radius"].default_value = 0.08
        geometry = self.node("ShaderNodeNewGeometry")
        facing = vector(
            self.tree,
            "DOT_PRODUCT",
            bevel.outputs["Normal"],
            geometry.outputs["Normal"],
        )
        edge = self.smooth(m("SUBTRACT", 1.0, facing), 0.01, 0.08)
        chipped = m("MULTIPLY", edge, self.cover(self.noise(3.0, 7.0), 0.15))
        return m("MULTIPLY", chipped, self.region)

    def scuffs(self) -> Socket:
        """Fenders and quays, amidships and halfway up the freeboard."""
        m, smooth = self.m, self.smooth
        ahead = m("ABSOLUTE", m("DIVIDE", self.y, self.length))
        amidships = m("SUBTRACT", 1.0, smooth(ahead, 0.2, 0.35))
        fendered = m(
            "MULTIPLY",
            smooth(self.height, 0.25, 0.4),
            m("SUBTRACT", 1.0, smooth(self.height, 0.6, 0.75)),
        )
        # Long and low: a fender rubs along the hull.
        scuff = self.cover(self.noise(1.5, 41.0, 3.0, (1.0, 0.25, 6.0)), 0.05, -0.35)
        return m(
            "MULTIPLY",
            m("MULTIPLY", scuff, fendered),
            m("MULTIPLY", amidships, m("MULTIPLY", self.shell, 0.6)),
        )

    def stain(self) -> Socket:
        """A yacht's waterline stain is a hand's width, a ship's boot top a metre or
        so, higher where the bow wave rides."""
        m, smooth = self.m, self.smooth
        bow = smooth(m("DIVIDE", self.y, self.length), 0.32, 0.45)
        top_m = m("MULTIPLY", self.length, m("MULTIPLY_ADD", self.steel, 0.01, 0.008))
        top_m = m(
            "MULTIPLY", m("MINIMUM", top_m, 1.2), m("MULTIPLY_ADD", bow, 0.8, 1.0)
        )
        top_m = m(
            "MULTIPLY", top_m, m("MULTIPLY_ADD", self.noise(0.5, 5.0, 2.0), 0.8, 0.6)
        )
        band = m(
            "MULTIPLY",
            smooth(self.z, -0.3, 0.0),
            m("SUBTRACT", 1.0, smooth(self.z, m("MULTIPLY", top_m, 0.5), top_m)),
        )
        return m("MULTIPLY", band, self.cover(self.noise(0.6, 5.0), 0.3, 0.3))

    def layers(self) -> tuple[Socket, ...]:
        """Grime, rust, stain, soot and fade, unweighted by zone."""
        m = self.m
        streak, scuff = self.streaks(), self.scuffs()
        not_steel = m("SUBTRACT", 1.0, self.steel)
        # Steel bleeds rust down a streak and at a scuff; anything else, grime.
        grime = m(
            "MAXIMUM",
            self.grime(),
            m("MULTIPLY", m("MAXIMUM", streak, scuff), not_steel),
        )
        washed = m(
            "MULTIPLY",
            self.shell,
            m("SUBTRACT", 1.0, self.smooth(self.height, 0.0, 0.5)),
        )
        grime = m("MULTIPLY", grime, m("SUBTRACT", 1.0, washed))
        rust = m(
            "MAXIMUM",
            self.chips(),
            m("MULTIPLY", m("MAXIMUM", streak, scuff), self.steel),
        )
        soot = self.cover(self.noise(0.5, 13.0), 0.4, 0.3)
        sky = m("MULTIPLY_ADD", self.smooth(self.up, 0.0, 1.0), 0.5, 0.5)
        return grime, rust, self.stain(), soot, m("MULTIPLY", self.amount, sky)


def masks() -> bpy.types.NodeTree:
    """Each layer's cover, 0-1, weighted by the face's zone and the object's wear.

    Reads what `tag` writes, and the object's `wear`, `wear_seed`, `wear_length_m`
    and `wear_steel`.
    """
    if (group := bpy.data.node_groups.get("wear_masks")) is not None:
        return group
    group = bpy.data.node_groups.new("wear_masks", "ShaderNodeTree")
    for layer in LAYERS:
        group.interface.new_socket(
            layer, in_out="OUTPUT", socket_type="NodeSocketFloat"
        )
    g = _Masks(group)
    out = g.node("NodeGroupOutput")
    # An unworn copy shares its worn twin's import and materials: no cover at all.
    worn = g.m("GREATER_THAN", g.amount, 0.0)
    for layer, mask in zip(LAYERS, g.layers(), strict=True):
        weight = g.attribute(f"wear_{layer}").outputs["Factor"]
        weight = g.m("MULTIPLY", weight, worn)
        g.put(out.inputs[layer], clamped(g.tree, g.m("MULTIPLY", mask, weight)))
    return group


# Judgement: linear albedo of each layer, from photographs of worn hulls.
GRIME = (0.045, 0.038, 0.030, 1.0)
RUST = (0.20, 0.065, 0.022, 1.0)
STAIN_STEEL = (0.06, 0.07, 0.035, 1.0)  # algae on a ship's boot top
STAIN = (0.22, 0.14, 0.05, 1.0)  # a yacht's: tannin, oil and algae
SOOT = (0.015, 0.014, 0.013, 1.0)


def dirty(parts: list[bpy.types.Object]) -> None:
    """Lay the wear over every material of an EO import's `parts`."""
    meshes = [o for p in parts for o in (p, *p.children_recursive) if o.type == "MESH"]
    for material in {s.material for o in meshes for s in o.material_slots} - {None}:
        _dirty(material)


def _dirty(material: bpy.types.Material) -> None:
    """A second, rough dielectric surface mixed in by the wear's cover, and the paint
    faded. Tinting a metallic surface instead would colour its mirror: rust on steel
    reads as copper."""
    g = _Graph(material.node_tree)
    shader = next((n for n in g.tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if shader is None:
        return
    group = g.node("ShaderNodeGroup", node_tree=masks())
    cover = {layer: group.outputs[layer] for layer in LAYERS}
    base = shader.inputs["Base Color"]
    if base.is_linked:
        colour = base.links[0].from_socket
    else:
        rgb = g.node("ShaderNodeRGB")
        rgb.outputs["Color"].default_value = base.default_value
        colour = rgb.outputs["Color"]
    faded = g.node("ShaderNodeHueSaturation")
    g.tree.links.new(colour, faded.inputs["Color"])
    g.tree.links.new(
        g.m("MULTIPLY_ADD", cover["fade"], -0.5, 1.0), faded.inputs["Saturation"]
    )
    g.tree.links.new(
        g.m("MULTIPLY_ADD", cover["fade"], 0.15, 1.0), faded.inputs["Value"]
    )
    g.tree.links.new(faded.outputs["Color"], base)

    total = g.m(
        "MAXIMUM",
        g.m("MAXIMUM", cover["grime"], cover["rust"]),
        g.m("MAXIMUM", cover["stain"], cover["soot"]),
    )
    stain = g.node("ShaderNodeMix", data_type="RGBA")
    g.tree.links.new(
        g.attribute("wear_steel", "OBJECT").outputs["Factor"], stain.inputs["Factor"]
    )
    stain.inputs["A"].default_value = STAIN
    stain.inputs["B"].default_value = STAIN_STEEL
    # Each layer's colour by its share of the cover, so rust is not muddied by grime.
    layered: Socket | tuple[float, ...] = GRIME
    for layer, colour in (
        ("rust", RUST),
        ("stain", stain.outputs["Result"]),
        ("soot", SOOT),
    ):
        mix = g.node("ShaderNodeMix", data_type="RGBA")
        share = clamped(
            g.tree, g.m("DIVIDE", cover[layer], g.m("MAXIMUM", total, 1e-3))
        )
        g.tree.links.new(share, mix.inputs["Factor"])
        g.put(mix.inputs["A"], layered)
        g.put(mix.inputs["B"], colour)
        layered = mix.outputs["Result"]
    dirt = g.node("ShaderNodeBsdfPrincipled")
    g.put(dirt.inputs["Base Color"], layered)
    dirt.inputs["Roughness"].default_value = 0.9
    if shader.inputs["Normal"].is_linked:
        g.tree.links.new(
            shader.inputs["Normal"].links[0].from_socket, dirt.inputs["Normal"]
        )
    consumers = [link.to_socket for link in shader.outputs["BSDF"].links]
    mix = g.node("ShaderNodeMixShader")
    g.tree.links.new(total, mix.inputs["Factor"])
    g.tree.links.new(shader.outputs["BSDF"], mix.inputs["Shader"])
    g.tree.links.new(dirt.outputs["BSDF"], mix.inputs["Shader_001"])
    for socket in consumers:
        g.tree.links.new(mix.outputs["Shader"], socket)


def vary(anchors: list[bpy.types.Object], seed: int) -> None:
    """Give each worn hull's meshes what the masks read of it, its noise drawn from
    a substream of its own so copies of one asset wear apart."""
    for anchor in anchors:
        if not anchor.get("wear"):
            continue
        offset = float(substream(seed, f"wear/{anchor.name}").uniform(0.0, 1000.0))
        steel = float(manifest()[anchor["asset"]].material == "steel")
        for part in anchor.children_recursive:
            if part.type == "MESH":
                part["wear"] = anchor["wear"]
                part["wear_seed"] = offset
                part["wear_length_m"] = anchor["length_m"]
                part["wear_steel"] = steel
