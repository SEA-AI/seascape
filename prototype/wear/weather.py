"""Prototype: procedural weathering for any asset with `zones`, in EO and LWIR.

One set of masks, driven by the asset's thermal zones, feeds both bands:
EO mixes colour and roughness, LWIR mixes zone skins with the absorptance and
emissivity of what the wear is made of. Hooks into a normal build, so nothing in
seascape changes.

Per face (mesh attributes, written at import):
    wear_p                      position in the fitted hull frame, metres, waterline z=0
    wear_grime/rust/scum/soot/fade  the zone's profile weight
Per object (custom properties, set after build, so copies of one asset differ):
    wear_amount  0..1, how worn
    wear_seed    noise offset, from substream(seed, "wear/<anchor>")
    wear_height_m
"""

import time
from dataclasses import replace
from typing import NamedTuple, get_args

import bpy
import numpy as np

from seascape.assets import manifest
from seascape.blender import faces, heat, meshes, scene, vessels
from seascape.blender.blend import curve_image, lookup, scalar
from seascape.config import substream
from seascape.physics import lwir, thermal

LAYERS = ("grime", "rust", "scum", "soot", "fade")


class Profile(NamedTuple):
    grime: float
    rust: float
    scum: float
    soot: float
    fade: float


# Judgement, all of it: how each zone weathers.
PROFILE: dict[str, Profile] = {
    "paint_hull": Profile(1.0, 1.0, 1.0, 0.0, 0.5),
    "paint_superstructure": Profile(0.8, 0.7, 0.0, 0.0, 0.5),
    "deck": Profile(1.0, 0.6, 0.0, 0.0, 0.3),
    "glass": Profile(0.35, 0.0, 0.0, 0.0, 0.0),  # salt film
    "bare_metal": Profile(0.6, 0.6, 0.0, 0.0, 0.0),
    "rubber": Profile(0.6, 0.0, 0.6, 0.0, 0.7),
    "fabric": Profile(0.8, 0.0, 0.0, 0.0, 0.7),
    "funnel": Profile(0.5, 0.5, 0.0, 1.0, 0.3),
    "exhaust": Profile(0.4, 0.0, 0.0, 1.0, 0.0),
    "engine_cowl": Profile(0.5, 0.0, 0.0, 0.0, 0.7),
    "wood_composite": Profile(0.8, 0.0, 1.0, 0.0, 0.6),  # GRP: no rust, scum line
}
ZONE_NAMES = list(get_args(thermal.ZoneName.__value__))
# Zones say how a surface heats, not what it is made of: a sloop's "deck" is teak and
# its coachroof GRP. Judgement: these categories have no steel to rust.
NOT_STEEL = frozenset({
    "motor yacht", "sailing yacht", "rigid inflatable boat", "uncrewed surface vessel",
    "log", "pallet",
})

# EO colours, linear. Judgement.
GRIME = (0.045, 0.038, 0.030, 1.0)
RUST = (0.20, 0.065, 0.022, 1.0)
SCUM = (0.06, 0.07, 0.035, 1.0)  # a ship's waterline: algae
MUSTACHE = (0.22, 0.14, 0.05, 1.0)  # a yacht's: tannin, oil and algae, yellow-brown
SOOT = (0.015, 0.014, 0.013, 1.0)

# LWIR. Unverified, recalled from the Neutrium compilation the zones cite: iron,
# rusted 0.61-0.85; lampblack 0.95. Absorptances are judgement: rust and dirt are
# dark in the visible; dirt raises a white paint's alpha, as cool-roof weathering does.
RUST_EMISSIVITY, RUST_ABSORPTANCE = 0.75, 0.75
GRIME_ABSORPTANCE = 0.6
SOOT_EMISSIVITY, SOOT_ABSORPTANCE = 0.95, 0.96

timings: dict[str, float] = {}


# --- tagging --------------------------------------------------------------------


def tag(name: str, band: str, parts: list[bpy.types.Object]) -> None:
    """Each face's zone profile and each vertex's place in the hull frame."""
    t0 = time.perf_counter()
    bpy.context.view_layer.update()  # matrix_world is stale after the fit
    objs = meshes.mesh_objects(parts)
    if band == "eo":
        zones = manifest()[name].zones
        found = faces.found(objs)
        per_face = (
            faces.face_zones(found, zones)
            if zones is not None
            else [np.full(len(o.data.polygons), ZONE_NAMES.index("paint_hull")) for o in found.objects]
        )
        pairs = list(zip(found.objects, per_face, strict=True))
        # Free the import's names, as the IR import does: the next asset's would
        # otherwise come in as "material.001" and miss its zone labels.
        for m in {s.material for o in objs for s in o.material_slots} - {None}:
            m.name = f"{name}/{m.name}"
        for o in [o for p in parts for o in (p, *p.children_recursive)]:
            o.name = f"{name}/{o.name}"
    else:
        pairs = []
        for o in objs:
            slots = np.array([ZONE_NAMES.index(z) for z in o["zones"]])
            pairs.append((o, slots[faces.ints(o.data.polygons, "material_index", len(o.data.polygons))]))
    table = np.array([PROFILE[z] for z in ZONE_NAMES], dtype=np.float32)
    if getattr(manifest()[name], "category", None) in NOT_STEEL:
        table[:, LAYERS.index("rust")] *= 0.15  # staining round fittings only
    worlds, vzones = [], []
    for obj, zone in pairs:
        if obj.data.users > 1:  # instanced parts each need their own wear_p
            obj.data = obj.data.copy()
        mesh = obj.data
        n = len(mesh.vertices)
        co = np.empty(n * 3, dtype=np.float32)
        # A skinned mesh renders posed, as the IR import bakes it: wear sits there.
        posed = obj.evaluated_get(bpy.context.evaluated_depsgraph_get()) if obj.modifiers else None
        src = posed.to_mesh() if posed is not None else mesh
        if len(src.vertices) != n or len(src.polygons) != len(mesh.polygons):
            src = mesh
        src.vertices.foreach_get("co", co)
        m = np.array(obj.matrix_world, dtype=np.float32)
        world = co.reshape(-1, 3) @ m[:3, :3].T + m[:3, 3]
        p = mesh.attributes.get("wear_p") or mesh.attributes.new("wear_p", "FLOAT_VECTOR", "POINT")
        p.data.foreach_set("vector", world.ravel())
        nf = len(mesh.polygons)
        normal = np.empty(nf * 3, dtype=np.float32)
        src.polygons.foreach_get("normal", normal)
        if posed is not None:
            posed.to_mesh_clear()
        normal = normal.reshape(-1, 3) @ np.linalg.inv(m[:3, :3]).T  # normals take the inverse transpose
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-9)
        hn = mesh.attributes.get("wear_n") or mesh.attributes.new("wear_n", "FLOAT_VECTOR", "FACE")
        hn.data.foreach_set("vector", normal.ravel())
        for i, layer in enumerate(LAYERS):
            a = mesh.attributes.get(f"wear_{layer}") or mesh.attributes.new(f"wear_{layer}", "FLOAT", "FACE")
            a.data.foreach_set("value", table[zone, i])
        # Each vertex takes the zone of a face it bounds.
        nl = len(mesh.loops)
        loop_vert = faces.ints(mesh.loops, "vertex_index", nl)
        starts = faces.ints(mesh.polygons, "loop_start", len(mesh.polygons))
        loop_face = np.repeat(np.arange(len(starts)), np.diff(np.append(starts, nl)))
        vz = np.zeros(n, dtype=np.int64)
        vz[loop_vert] = zone[loop_face]
        worlds.append(world)
        vzones.append(vz)
    _below(pairs, worlds, vzones)
    timings["tag"] = timings.get("tag", 0.0) + time.perf_counter() - t0


def _below(pairs, worlds, vzones) -> None:
    """Each vertex's depth below the highest point of its zone in the same plan cell:
    for a hull side, the deck edge above it; for a house wall, its roof edge. Runoff
    and rust streaks start there and run down."""
    world = np.concatenate(worlds)
    zone = np.concatenate(vzones)
    length = float(np.ptp(world[:, 1])) or 1.0
    cell = max(0.2, length / 200.0)
    ij = np.floor(world[:, :2] / cell).astype(np.int64)
    ij -= ij.min(axis=0)
    key = (zone * (ij[:, 0].max() + 1) + ij[:, 0]) * (ij[:, 1].max() + 1) + ij[:, 1]
    uniq, inverse = np.unique(key, return_inverse=True)
    top = np.full(len(uniq), -np.inf, dtype=np.float32)
    np.maximum.at(top, inverse, world[:, 2])
    below = top[inverse] - world[:, 2]
    i = 0
    for (obj, _), w in zip(pairs, worlds, strict=True):
        a = obj.data.attributes.get("wear_below_m") or obj.data.attributes.new("wear_below_m", "FLOAT", "POINT")
        a.data.foreach_set("value", below[i : i + len(w)])
        i += len(w)


def per_vessel(scenario, amounts: dict[str, float]) -> None:
    for anchor in bpy.data.objects:
        if anchor.type != "EMPTY" or "engine_off_min" not in anchor:
            continue
        rng = substream(scenario.seed, f"wear/{anchor.name}")
        seed = float(rng.uniform(0.0, 100.0))
        amount = amounts.get(anchor.name.split(".")[0], float(rng.uniform(0.2, 0.9)))
        for obj in anchor.children_recursive:
            if obj.type == "MESH":
                obj["wear_amount"] = amount
                obj["wear_seed"] = seed
                obj["wear_height_m"] = anchor["height_m"]
                obj["wear_length_m"] = anchor["length_m"]
                category = getattr(manifest().get(anchor.name.split(".")[0]), "category", None)
                obj["wear_steel"] = 0.0 if category in NOT_STEEL else 1.0


# --- masks ----------------------------------------------------------------------


class _G:
    """Node helpers on one tree."""

    def __init__(self, tree: bpy.types.NodeTree) -> None:
        self.tree = tree

    def node(self, kind: str, **props) -> bpy.types.Node:
        n = self.tree.nodes.new(kind)
        for k, v in props.items():
            setattr(n, k, v)
        return n

    def link(self, a, b) -> None:
        self.tree.links.new(a, b)

    def put(self, socket, v) -> None:
        if isinstance(v, (int, float, tuple)):
            socket.default_value = v
        else:
            self.link(v, socket)

    def math(self, op: str, *xs, clamp: bool = False):
        n = self.node("ShaderNodeMath", operation=op, use_clamp=clamp)
        for i, x in enumerate(xs):
            self.put(n.inputs[i], x)
        return n.outputs["Value"]

    def smooth(self, x, lo, hi):
        n = self.node("ShaderNodeMapRange", interpolation_type="SMOOTHSTEP")
        self.put(n.inputs["Value"], x)
        self.put(n.inputs["From Min"], lo)
        self.put(n.inputs["From Max"], hi)
        return n.outputs["Result"]

    def attr(self, name: str, kind: str = "GEOMETRY", out: str = "Fac"):
        return self.node("ShaderNodeAttribute", attribute_type=kind, attribute_name=name).outputs[out]


def masks_group() -> bpy.types.NodeTree:
    """Outputs each layer's coverage, 0..1, already weighted by zone and amount.

    Where each layer sits follows what boat-care and coating sources report: grime
    and rust run down from deck and roof edges and from discrete sources (scuppers,
    the hawse pipe at the bow), the sea washes the lower topsides, a waterline stain
    rises at the bow, fenders scuff the sides amidships, coatings fail at edges.
    """
    if (g := bpy.data.node_groups.get("wear_masks")) is not None:
        return g
    g = bpy.data.node_groups.new("wear_masks", "ShaderNodeTree")
    for layer in LAYERS:
        g.interface.new_socket(layer, in_out="OUTPUT", socket_type="NodeSocketFloat")
    b = _G(g)
    out = b.node("NodeGroupOutput")
    M, A, S_ = b.math, b.attr, b.smooth

    p = A("wear_p", out="Vector")
    amount = A("wear_amount", "OBJECT")
    seed = A("wear_seed", "OBJECT")
    height = A("wear_height_m", "OBJECT")
    length = A("wear_length_m", "OBJECT")
    steel = A("wear_steel", "OBJECT")
    below = A("wear_below_m")
    shell = M("GREATER_THAN", A("wear_scum"), 0.0)  # zones the sea wets: the hull shell
    xyz = b.node("ShaderNodeSeparateXYZ")
    b.link(p, xyz.inputs["Vector"])
    x, y, z = (xyz.outputs[k] for k in "XYZ")
    yn = M("DIVIDE", y, length)  # -0.5 stern .. 0.5 bow
    midships = M("SUBTRACT", 1.0, S_(M("ABSOLUTE", yn), 0.2, 0.35))
    bow = S_(yn, 0.32, 0.45)
    geo = b.node("ShaderNodeNewGeometry")
    nz = b.node("ShaderNodeSeparateXYZ")
    b.link(geo.outputs["Normal"], nz.inputs["Vector"])
    up = nz.outputs["Z"]
    vertical = M("SUBTRACT", 1.0, S_(M("ABSOLUTE", up), 0.2, 0.5))

    def noise(scale, w, detail=6.0, stretch=(1.0, 1.0, 1.0)):
        m = b.node("ShaderNodeVectorMath", operation="MULTIPLY")
        b.link(p, m.inputs[0])
        m.inputs[1].default_value = stretch
        n = b.node("ShaderNodeTexNoise", noise_dimensions="4D")
        b.link(m.outputs["Vector"], n.inputs["Vector"])
        n.inputs["Scale"].default_value = scale
        n.inputs["Detail"].default_value = detail
        n.inputs["Roughness"].default_value = 0.6
        b.link(M("ADD", seed, w), n.inputs["W"])
        return n.outputs["Fac"]

    def cover(x_, soft, boost=0.0):
        t = M("MULTIPLY_ADD", M("ADD", amount, boost), -0.45, 0.72)
        return S_(x_, t, M("ADD", t, soft))

    # How bad each region is: some plates and panels are far worse than others.
    health = S_(noise(0.04, 21.0, detail=2.0), 0.3, 0.7)
    region = M("MULTIPLY_ADD", health, 1.3, 0.2)

    # Freeboard washed by the sea: grime fades over the lowest ~6% of length.
    wash = S_(z, 0.0, M("MULTIPLY", length, 0.06))
    clean_low = M("SUBTRACT", 1.0, M("MULTIPLY", shell, M("SUBTRACT", 1.0, wash)))

    # Grime: crevices, a film, pooling on upward faces; none where the sea washes.
    ao = b.node("ShaderNodeAmbientOcclusion", only_local=True, samples=8)
    ao.inputs["Distance"].default_value = 0.6
    crevice = S_(M("SUBTRACT", 1.0, ao.outputs["AO"]), 0.05, 0.6)
    grime = M("MULTIPLY", crevice, cover(noise(0.8, 1.0), 0.3), clamp=True)
    film = M("MULTIPLY", cover(noise(0.15, 3.0), 0.3, boost=-0.2), 0.35)
    pool = M("MULTIPLY", S_(up, 0.6, 0.95), M("MULTIPLY", cover(noise(1.5, 4.0), 0.3), 0.5))
    grime = M("MULTIPLY", M("MAXIMUM", M("MAXIMUM", grime, film), pool), clean_low)

    # Streaks: one source per Voronoi cell along the edges, each with its own length,
    # width and strength; most of them dormant on a lightly worn hull. Denser at the
    # bow, where the hawse pipe and anchor bleed.
    spacing = M("MINIMUM", M("MAXIMUM", M("DIVIDE", length, 30.0), 0.6), 4.0)
    spacing = M("MULTIPLY", spacing, M("MULTIPLY_ADD", bow, -0.6, 1.0))
    # Along the surface: y on a side, x on a bulkhead facing fore or aft.
    hn = b.node("ShaderNodeSeparateXYZ")
    b.link(A("wear_n", out="Vector"), hn.inputs["Vector"])
    side = M("GREATER_THAN", M("ABSOLUTE", hn.outputs["X"]), M("ABSOLUTE", hn.outputs["Y"]))
    u = M("ADD", M("MULTIPLY", side, y), M("MULTIPLY", M("SUBTRACT", 1.0, side), x))
    cells = b.node("ShaderNodeTexVoronoi", voronoi_dimensions="1D", feature="F1")
    b.link(M("ADD", M("DIVIDE", u, spacing), M("MULTIPLY", seed, 7.31)), cells.inputs["W"])
    rnd = b.node("ShaderNodeSeparateColor")
    b.link(cells.outputs["Color"], rnd.inputs["Color"])
    r_len, r_width, r_on = rnd.outputs["Red"], rnd.outputs["Green"], rnd.outputs["Blue"]
    on = M("GREATER_THAN", r_on, M("MULTIPLY_ADD", amount, -0.7, 0.85))
    run_m = M("MULTIPLY", M("MULTIPLY_ADD", r_len, 0.9, 0.1), M("MULTIPLY", length, 0.05))
    run_m = M("MULTIPLY", run_m, M("MULTIPLY_ADD", amount, 1.0, 0.4))
    width = M("MULTIPLY_ADD", r_width, 0.12, 0.03)  # in cell units
    core = M("SUBTRACT", 1.0, S_(cells.outputs["Distance"], M("MULTIPLY", width, 0.4), width))
    fade_down = M("SUBTRACT", 1.0, S_(below, M("MULTIPLY", run_m, 0.2), run_m))
    breakup = S_(noise(2.0, 17.0, detail=4.0, stretch=(4.0, 4.0, 0.5)), 0.25, 0.6)
    streak = M("MULTIPLY", M("MULTIPLY", core, fade_down), M("MULTIPLY", on, vertical))
    streak = M("MULTIPLY", M("MULTIPLY", streak, breakup), region, clamp=True)
    hard = M("MAXIMUM", shell, M("GREATER_THAN", A("wear_rust"), 0.0))  # not sails or canvas
    streak = M("MULTIPLY", streak, hard)
    # A steel hull bleeds rust; on GRP the same runoff is black deck grime.
    rust_streak = M("MULTIPLY", streak, steel)
    grime = M("MAXIMUM", grime, M("MULTIPLY", streak, M("SUBTRACT", 1.0, steel)))

    # Coating failure at edges, worse in bad regions.
    bevel = b.node("ShaderNodeBevel", samples=8)
    bevel.inputs["Radius"].default_value = 0.08
    dot = b.node("ShaderNodeVectorMath", operation="DOT_PRODUCT")
    b.link(bevel.outputs["Normal"], dot.inputs[0])
    b.link(geo.outputs["Normal"], dot.inputs[1])
    edge = S_(M("SUBTRACT", 1.0, dot.outputs["Value"]), 0.01, 0.08)
    chips = M("MULTIPLY", M("MULTIPLY", edge, cover(noise(3.0, 7.0), 0.15)), region, clamp=True)

    # Fender and quay scuffs: horizontal, amidships, mid-freeboard, on the shell.
    freeboard = M("ADD", z, below)
    mid = M("DIVIDE", z, M("MAXIMUM", freeboard, 0.01))
    band = M("MULTIPLY", S_(mid, 0.25, 0.4), M("SUBTRACT", 1.0, S_(mid, 0.6, 0.75)))
    scuff = cover(noise(1.5, 41.0, detail=3.0, stretch=(1.0, 0.25, 6.0)), 0.05, boost=-0.35)
    scuff = M("MULTIPLY", M("MULTIPLY", scuff, band), M("MULTIPLY", midships, M("MULTIPLY", shell, 0.6)))
    rust = M("MAXIMUM", M("MAXIMUM", chips, rust_streak), M("MULTIPLY", scuff, steel))
    grime = M("MAXIMUM", grime, M("MULTIPLY", scuff, M("SUBTRACT", 1.0, steel)))

    # Waterline stain: thin on a yacht, wider on a ship; higher at the bow wave.
    # Capped: a big ship's boot-top stain is a metre or two, not a fiftieth of 300 m.
    top = M("MINIMUM", M("MULTIPLY", length, M("MULTIPLY_ADD", steel, 0.01, 0.008)), 1.2)
    top = M("MULTIPLY", top, M("MULTIPLY_ADD", bow, 0.8, 1.0))
    top = M("MULTIPLY", top, M("MULTIPLY_ADD", noise(0.5, 5.0, detail=2.0), 0.8, 0.6))
    stain = M("MULTIPLY", S_(z, -0.3, 0.0), M("SUBTRACT", 1.0, S_(z, M("MULTIPLY", top, 0.5), top)))
    scum = M("MULTIPLY", stain, cover(noise(0.6, 5.0), 0.3, boost=0.3), clamp=True)

    # Soot: heavier towards the top of the vessel.
    high = S_(M("DIVIDE", z, height), 0.5, 1.0)
    soot = M("MULTIPLY", cover(noise(0.5, 13.0), 0.4, boost=0.3), M("MULTIPLY_ADD", high, 0.6, 0.4))

    # Fade and chalking: most on what faces up.
    fade = M("MULTIPLY", amount, M("MULTIPLY_ADD", S_(up, 0.0, 1.0), 0.5, 0.5))

    for layer, mask in zip(LAYERS, (grime, rust, scum, soot, fade), strict=True):
        b.link(M("MULTIPLY", mask, A(f"wear_{layer}"), clamp=True), out.inputs[layer])
    return g


def _masks(b: _G) -> dict[str, bpy.types.NodeSocket]:
    n = b.node("ShaderNodeGroup")
    n.node_tree = masks_group()
    return {layer: n.outputs[layer] for layer in LAYERS}


# --- EO -------------------------------------------------------------------------


def weather_eo(material: bpy.types.Material) -> None:
    b = _G(material.node_tree)
    bsdf = next((n for n in b.tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is None:
        return
    base_in, rough_in, metal_in = (bsdf.inputs[k] for k in ("Base Color", "Roughness", "Metallic"))
    if base_in.is_linked:
        base = base_in.links[0].from_socket
    else:
        rgb = b.node("ShaderNodeRGB")
        rgb.outputs["Color"].default_value = base_in.default_value
        base = rgb.outputs["Color"]
    m = _masks(b)

    hsv = b.node("ShaderNodeHueSaturation")
    b.link(base, hsv.inputs["Color"])
    b.link(b.math("MULTIPLY_ADD", m["fade"], -0.5, 1.0), hsv.inputs["Saturation"])
    b.link(b.math("MULTIPLY_ADD", m["fade"], 0.15, 1.0), hsv.inputs["Value"])
    b.link(hsv.outputs["Color"], base_in)

    # Wear is a second, rough dielectric surface over a fraction of the clean one.
    # Mixing its colour into a metallic BSDF instead tints the mirror: copper.
    cover = b.math(
        "MAXIMUM", b.math("MAXIMUM", m["grime"], m["rust"]), b.math("MAXIMUM", m["scum"], m["soot"])
    )
    steel = b.attr("wear_steel", "OBJECT")
    scum_colour = b.node("ShaderNodeMix", data_type="RGBA")
    b.link(steel, scum_colour.inputs["Factor"])
    scum_colour.inputs["A"].default_value = MUSTACHE
    scum_colour.inputs["B"].default_value = SCUM
    # Each layer's colour by its share of the cover, so rust is not muddied by grime.
    col = GRIME
    for layer, colour in (("rust", RUST), ("scum", scum_colour.outputs["Result"]), ("soot", SOOT)):
        mix = b.node("ShaderNodeMix", data_type="RGBA")
        b.link(b.math("DIVIDE", m[layer], b.math("MAXIMUM", cover, 1e-3), clamp=True), mix.inputs["Factor"])
        b.put(mix.inputs["A"], col)
        b.put(mix.inputs["B"], colour)
        col = mix.outputs["Result"]
    dirt = b.node("ShaderNodeBsdfPrincipled")
    b.put(dirt.inputs["Base Color"], col)
    dirt.inputs["Roughness"].default_value = 0.9
    dirt.inputs["Metallic"].default_value = 0.0
    if bsdf.inputs["Normal"].is_linked:
        b.link(bsdf.inputs["Normal"].links[0].from_socket, dirt.inputs["Normal"])
    consumers = [link.to_socket for link in bsdf.outputs["BSDF"].links]
    mix = b.node("ShaderNodeMixShader")
    b.link(cover, mix.inputs["Factor"])
    b.link(bsdf.outputs["BSDF"], mix.inputs[1])
    b.link(dirt.outputs["BSDF"], mix.inputs[2])
    for to in consumers:
        b.link(mix.outputs["Shader"], to)


def weather_eo_all() -> None:
    t0 = time.perf_counter()
    done = set()
    for obj in bpy.data.objects:
        if obj.type != "MESH" or "wear_p" not in obj.data.attributes:
            continue
        for slot in obj.material_slots:
            if slot.material and slot.material.name not in done:
                done.add(slot.material.name)
                weather_eo(slot.material)
    timings["eo_materials"] = len(done)
    timings["eo_weather_s"] = time.perf_counter() - t0


# --- LWIR -----------------------------------------------------------------------


def _skin(tree, zone: thermal.Zone, h: heat.Heat, name: str, zone_name: str | None):
    """heat.zone_skin's emission and reflection, as a socket on `tree`."""
    link = tree.links.new
    sun_cos, up, table = thermal.temperature_table_k(zone, h.env)
    image = curve_image(name, np.vectorize(lwir.band_radiance)(table))
    geometry = tree.nodes.new("ShaderNodeNewGeometry")
    normal = tree.nodes.new("ShaderNodeSeparateXYZ")
    link(geometry.outputs["Normal"], normal.inputs["Vector"])
    y = scalar(
        tree, "MULTIPLY_ADD",
        scalar(tree, "MULTIPLY_ADD", normal.outputs["Z"], 0.5, 0.5),
        (len(up) - 1) / len(up), 0.5 / len(up),
    )
    if h.sun is None:
        x = tree.nodes.new("ShaderNodeValue")
        x.outputs["Value"].default_value = 0.5 / len(sun_cos)
        at = x.outputs["Value"]
    else:
        facing = tree.nodes.new("ShaderNodeVectorMath")
        facing.operation = "DOT_PRODUCT"
        facing.inputs["Vector_001"].default_value = h.sun
        link(geometry.outputs["Normal"], facing.inputs["Vector"])
        lit = scalar(tree, "MAXIMUM", facing.outputs["Value"], 0.0)
        at = scalar(tree, "MULTIPLY_ADD", lit, (len(sun_cos) - 1) / len(sun_cos), 0.5 / len(sun_cos))
    emission = tree.nodes.new("ShaderNodeEmission")
    link(lookup(tree, image, at, y), emission.inputs["Color"])
    return heat._reflecting(tree, emission.outputs["Emission"], zone.emissivity, zone_name)


def _as(zone: thermal.Zone, emissivity: float | None = None, absorptance: float | None = None) -> thermal.Zone:
    a = zone.absorptance if absorptance is None else absorptance
    return replace(
        zone,
        emissivity=zone.emissivity if emissivity is None else emissivity,
        absorptance=a,
        absorptance_range=(a, a),
    )


def worn_zone_skin(zone_name, h: heat.Heat, paint_absorptance=None, wet=True) -> bpy.types.Material:
    """heat.zone_skin, mixed with the skins of the grime, rust, scum and soot on it."""
    clean = heat.zone_skin(zone_name, h, paint_absorptance, wet)  # for its name
    name = f"{clean.name}_worn"
    if (found := bpy.data.materials.get(name)) is not None:
        return found
    t0 = time.perf_counter()
    zone = thermal.ZONES[zone_name]
    if paint_absorptance is not None and zone_name in thermal.PAINTED:
        zone = _as(zone, absorptance=round(paint_absorptance, 2))
    material = bpy.data.materials.new(name)
    tree = material.node_tree
    tree.nodes.clear()
    b = _G(tree)
    m = _masks(b)

    s = _skin(tree, zone, h, name, zone_name)
    variants = {
        "grime": _skin(tree, _as(zone, absorptance=max(zone.absorptance, GRIME_ABSORPTANCE)), h, f"{name}_grime", None),
        "rust": _skin(tree, _as(zone, RUST_EMISSIVITY, RUST_ABSORPTANCE), h, f"{name}_rust", None),
        "soot": _skin(tree, _as(zone, SOOT_EMISSIVITY, SOOT_ABSORPTANCE), h, f"{name}_soot", None),
    }
    # Judgement: the splash zone stays wet, near the bulk water.
    water = b.node("ShaderNodeEmission")
    water.inputs["Strength"].default_value = lwir.band_radiance(h.t_water_k)
    variants["scum"] = heat._reflecting(tree, water.outputs["Emission"], thermal.SEA_EMISSIVITY, None)
    for layer in ("grime", "rust", "scum", "soot"):
        mix = b.node("ShaderNodeMixShader")
        b.link(m[layer], mix.inputs["Factor"])
        b.link(s, mix.inputs[1])
        b.link(variants[layer], mix.inputs[2])
        s = mix.outputs["Shader"]
    out = b.node("ShaderNodeOutputMaterial")
    b.link(heat._wetted(tree, s, h) if wet else s, out.inputs["Surface"])
    timings["ir_weather_s"] = timings.get("ir_weather_s", 0.0) + time.perf_counter() - t0
    return material


# --- hooks ----------------------------------------------------------------------


def install(scenario, amounts: dict[str, float] | None = None) -> None:
    amounts = amounts or {}
    original_import = vessels.import_asset

    def import_and_tag(name, band):
        parts = original_import(name, band)
        tag(name, band, parts)
        return parts

    vessels.import_asset = import_and_tag
    vessels.zone_skin = worn_zone_skin
    original_build = scene.build

    def build(scn, band="eo", cull=False):
        t0 = time.perf_counter()
        built = original_build(scn, band, cull)
        per_vessel(scn, amounts)
        if band == "eo":
            weather_eo_all()
        timings[f"build_{band}_s"] = time.perf_counter() - t0
        return built

    scene.build = build
