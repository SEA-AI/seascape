"""The sea's grid, and the waves it draws: tall ones displaced, the rest as normals."""

import math
from collections.abc import Callable

import bpy

from seascape.blend import (
    _math,
    _vector,
    place,
)
from seascape.blender.sea.pixel import _instance, _Pixel, _pixel
from seascape.config import Scenario, Sea
from seascape.physics.waves import (
    Wave,
    earth_radius_m,
    horizon_m,
    sea_z_m,
    tallest,
)

# A judgement: the waves carrying this share of their own sea's height variance are
# drawn as geometry, the rest as shading.
DISPLACED_HEIGHT_SHARE = 0.999
# Measured: the shortest wave drawn as geometry; 1 m overflows Cycles' SVM stack, which
# surface and displacement share, at 7 m/s in eo, 2 m fits.
DISPLACED_MIN_M = 4.0
# A judgement: the micropolygon edge adaptive subdivision dices the sea to, in pixels.
DICING_PX = 2.0
# A judgement: the longest edge it dices to near the camera, an eighth of the shortest
# displaced wave, where a straight edge strays from a cosine by at most 1 - cos(pi / 8),
# 8 % of its amplitude.
DICED_EDGE_M = DISPLACED_MIN_M / 8
# A judgement: off-screen sea, behind and beside the camera, diced this much coarser;
# Blender's default of 4 dices it to 10 GB at a 6 m eye.
OFFSCREEN_DICING = 128.0


# Cells per side: enough for the tangent point to land on a face, not an accuracy
# knob. A cell's sagitta, width^2 / 8R, is far under a pixel at the horizon.
SEA_CELLS = 128

# Margin on the horizon, or the grid's own edge becomes the horizon.
SEA_MARGIN = 1.5


def sea_reach_m(scenario: Scenario) -> float:
    """Half-width of the sea, a margin past the highest rig's horizon."""
    height_m = max(rig.height_m for rig in scenario.rigs.values())
    return SEA_MARGIN * horizon_m(height_m, scenario.sea.refraction_k)


# Waves per lane-vector: a Vector Math node costs Cycles about what a Math node does.
LANES = 3

type Triple = tuple[Wave | None, ...]  # LANES long, None an empty lane


def _triples(field: tuple[Wave, ...]) -> list[Triple]:
    """`field` in consecutive threes, in field order, the last padded."""
    return [
        tuple(field[i : i + LANES]) + (None,) * (LANES - len(field[i : i + LANES]))
        for i in range(0, len(field), LANES)
    ]


def _group(
    name: str,
    inputs: tuple[tuple[str, str], ...],
    outputs: tuple[tuple[str, str], ...],
) -> tuple[bpy.types.NodeTree, bpy.types.NodeOutputs, bpy.types.NodeInputs]:
    """A node group with these (name, socket kind) inputs and outputs; its Group Input's
    outputs and Group Output's inputs."""
    group = bpy.data.node_groups.new(name, "ShaderNodeTree")
    for sockets, in_out in ((inputs, "INPUT"), (outputs, "OUTPUT")):
        for socket, kind in sockets:
            group.interface.new_socket(
                socket, in_out=in_out, socket_type=f"NodeSocket{kind}"
            )
    given = group.nodes.new("NodeGroupInput").outputs
    return group, given, group.nodes.new("NodeGroupOutput").inputs


def _spread(
    group: bpy.types.NodeTree, value: bpy.types.NodeSocket
) -> bpy.types.NodeSocket:
    """A scalar in every lane. Cycles merges identical nodes, so each spread runs once
    per sample however many groups make it."""
    node = group.nodes.new("ShaderNodeCombineXYZ")
    for lane in node.inputs:
        group.links.new(value, lane)
    return node.outputs["Vector"]


def _total(
    tree: bpy.types.NodeTree, lanes: bpy.types.NodeSocket
) -> bpy.types.NodeSocket:
    return _vector(tree, "DOT_PRODUCT", lanes, (1.0, 1.0, 1.0))


# What the pixel resolves: shared by every triple.
_FOOTPRINT = (("Along", "Vector"), ("Across Sq", "Float"), ("Stretch", "Float"))
# How each lane fades and calms.
_FADE = (
    ("Tx", "Vector"),
    ("Ty", "Vector"),
    ("Half K Sq", "Vector"),
    ("Damped", "Vector"),
)
_SLOPE = (("Slope X", "Vector"), ("Slope Y", "Vector"))
_SUMS = (("Sum X", "Vector"), ("Sum Y", "Vector"), ("Sum Z", "Vector"))


def _fade(
    group: bpy.types.NodeTree, given: bpy.types.NodeOutputs, calm: bool
) -> tuple[bpy.types.NodeSocket, tuple[bpy.types.NodeSocket, bpy.types.NodeSocket]]:
    """How much of each lane the pixel draws, `waves.visibility` at its footprint
    along the wave, footprint^2 = across^2 + (along^2 - across^2) cos^2: sin(y) / y
    at y = footprint k / 2, held at 0 past pi; scaled by `Calm` in a `Damped` lane.
    And `Along`, (x, y), spread.

    Built alike in every group, so Cycles merges a triple's fade across them."""
    along = group.nodes.new("ShaderNodeSeparateXYZ")
    group.links.new(given["Along"], along.inputs["Vector"])
    along_xy = (_spread(group, along.outputs["X"]), _spread(group, along.outputs["Y"]))
    heading = _vector(
        group,
        "MULTIPLY_ADD",
        given["Ty"],
        along_xy[1],
        _vector(group, "MULTIPLY", given["Tx"], along_xy[0]),
    )
    footprint_sq = _vector(
        group,
        "MULTIPLY_ADD",
        _vector(group, "MULTIPLY", heading, heading),
        _spread(group, given["Stretch"]),
        _spread(group, given["Across Sq"]),
    )
    y = _vector(
        group,
        "POWER",
        _vector(group, "MULTIPLY", footprint_sq, given["Half K Sq"]),
        (0.5, 0.5, 0.5),
    )
    # An empty lane's y is 0, and Vector Math divides by 0 to 0.
    seen = _vector(
        group,
        "DIVIDE",
        _vector(group, "SINE", _vector(group, "MINIMUM", y, (math.pi,) * 3)),
        y,
    )
    if calm:
        less = _spread(group, _math(group, "SUBTRACT", given["Calm"], 1.0))
        scale = _vector(group, "MULTIPLY_ADD", given["Damped"], less, (1.0, 1.0, 1.0))
        seen = _vector(group, "MULTIPLY", seen, scale)
    return seen, along_xy


def _wave_group(calm: bool) -> bpy.types.NodeTree:
    """Three waves, each as far as `_fade` says the pixel resolves it. Each adds its
    slope sin(phase) to `Sum X` and `Sum Y` and `Lift` cos(phase) to `Sum Z`, per lane:
    a downward acceleration in g when `Lift` is a k, a height when it is a."""
    name = "waves_calm" if calm else "waves"
    if name in bpy.data.node_groups:
        return bpy.data.node_groups[name]
    group, given, out = _group(
        name,
        (
            ("Position", "Vector"),
            *_FOOTPRINT,
            ("Calm", "Float"),
            *_SUMS,
            ("Kx", "Vector"),
            ("Ky", "Vector"),
            ("W", "Vector"),
            ("Phase", "Vector"),
            *_FADE,
            *_SLOPE,
            ("Lift", "Vector"),
        ),
        _SUMS,
    )
    position = group.nodes.new("ShaderNodeSeparateXYZ")
    group.links.new(given["Position"], position.inputs["Vector"])
    east, north, time_s = (_spread(group, position.outputs[a]) for a in "XYZ")
    phase = _vector(
        group,
        "MULTIPLY_ADD",
        given["Kx"],
        east,
        _vector(
            group,
            "MULTIPLY_ADD",
            given["Ky"],
            north,
            _vector(group, "MULTIPLY_ADD", given["W"], time_s, given["Phase"]),
        ),
    )
    seen, _ = _fade(group, given, calm)

    def shown(trig: str) -> bpy.types.NodeSocket:
        return _vector(group, "MULTIPLY", _vector(group, trig, phase), seen)

    sine = shown("SINE")
    # -d height / dx of a cos(phase) is a k_x sin(phase).
    for total, factor, wave in (
        ("Sum X", "Slope X", sine),
        ("Sum Y", "Slope Y", sine),
        ("Sum Z", "Lift", shown("COSINE")),
    ):
        group.links.new(
            _vector(group, "MULTIPLY_ADD", wave, given[factor], given[total]),
            out[total],
        )
    return group


def _lanes(triple: Triple, value: Callable[[Wave], float], pad: float = 0.0) -> tuple:
    return tuple(pad if wave is None else value(wave) for wave in triple)


def _set_triple(
    inputs: bpy.types.NodeInputs, triple: Triple, damped: frozenset[Wave]
) -> None:
    """The inputs of `_fade` and the slopes that place `triple`. An empty lane has no
    amplitude, so whatever it fades adds 0."""

    sets = {
        "Tx": _lanes(triple, lambda w: math.sin(w.toward_rad)),
        "Ty": _lanes(triple, lambda w: math.cos(w.toward_rad)),
        "Half K Sq": _lanes(triple, lambda w: (w.k_rad_m / 2) ** 2),
        "Damped": _lanes(triple, lambda w: float(w in damped)),
        "Slope X": _lanes(triple, lambda w: w.amplitude_m * w.k_east_rad_m),
        "Slope Y": _lanes(triple, lambda w: w.amplitude_m * w.k_north_rad_m),
        "Kx": _lanes(triple, lambda w: w.k_east_rad_m),
        "Ky": _lanes(triple, lambda w: w.k_north_rad_m),
        "W": _lanes(triple, lambda w: -w.omega_rad_s),
        "Phase": _lanes(triple, lambda w: w.phase_rad),
    }
    for name, value in sets.items():
        if name in inputs:
            inputs[name].default_value = value


def _feed(
    group: bpy.types.NodeTree,
    given: bpy.types.NodeOutputs,
    node: bpy.types.Node,
    calmed: bool,
) -> None:
    """The footprint and, where `calmed`, the calm of a group around a triple."""
    for name, _ in _FOOTPRINT:
        group.links.new(given[name], node.inputs[name])
    if calmed:
        group.links.new(given["Calm"], node.inputs["Calm"])


def _normal_group(
    triples: list[Triple], wind_triples: int, damped: frozenset[Wave]
) -> bpy.types.NodeTree:
    """The slope the pixel draws at `Position`, and the downward acceleration in g of
    the first `wind_triples`. A group, as each node in a tree slows every edit to it."""
    group, given, out = _group(
        "sea_waves",
        (("Position", "Vector"), *_FOOTPRINT, ("Calm", "Float")),
        (("Gradient", "Vector"), ("Acceleration", "Float")),
    )
    sums = None
    for i, triple in enumerate(triples):
        calmed = not damped.isdisjoint(triple)
        node = group.nodes.new("ShaderNodeGroup")
        node.node_tree = _wave_group(calmed)
        node.name = f"waves_{i}"
        group.links.new(given["Position"], node.inputs["Position"])
        _feed(group, given, node, calmed)
        if sums is not None:
            for name, _ in _SUMS:
                group.links.new(sums[name], node.inputs[name])
        _set_triple(node.inputs, triple, damped)
        node.inputs["Lift"].default_value = _lanes(
            triple, lambda w: w.amplitude_m * w.k_rad_m
        )
        sums = node.outputs
        if i == wind_triples - 1:
            group.links.new(_total(group, sums["Sum Z"]), out["Acceleration"])
    if sums is not None:
        gradient = group.nodes.new("ShaderNodeCombineXYZ")
        group.links.new(_total(group, sums["Sum X"]), gradient.inputs["X"])
        group.links.new(_total(group, sums["Sum Y"]), gradient.inputs["Y"])
        group.links.new(gradient.outputs["Vector"], out["Gradient"])
    return group


def _displacement(
    tree: bpy.types.NodeTree, tall: tuple[Wave, ...], position: bpy.types.NodeSocket
) -> bpy.types.NodeSocket:
    """The offset of `tall`'s Gerstner surface from the flat sea at `position`, each
    wave faded by the footprint of the micropolygon there: up a cos(phase), and
    -k_hat a sin(phase) along the sea (Gerstner 1802; Tessendorf 2001, eq. 29, with
    lambda = 1). Adaptive subdivision dices at about a pixel, so the micropolygon's
    footprint is the pixel's."""
    camera = tree.nodes.new("ShaderNodeCameraData")
    # From the camera, in world space: the Geometry node's Incoming is unset here.
    toward = tree.nodes.new("ShaderNodeVectorTransform")
    toward.vector_type, toward.convert_from, toward.convert_to = (
        "VECTOR",
        "CAMERA",
        "WORLD",
    )
    tree.links.new(camera.outputs["View Vector"], toward.inputs["Vector"])
    pixel = _pixel(tree, toward.outputs["Vector"])
    sums = None
    for triple in _triples(tall):
        node = _instance(tree, _wave_group(calm=False), pixel, None)
        tree.links.new(position, node.inputs["Position"])
        if sums is not None:
            for name, _ in _SUMS:
                tree.links.new(sums[name], node.inputs[name])
        _set_triple(node.inputs, triple, frozenset())
        node.inputs["Slope X"].default_value = _lanes(
            triple, lambda w: w.amplitude_m * w.k_east_rad_m / w.k_rad_m
        )
        node.inputs["Slope Y"].default_value = _lanes(
            triple, lambda w: w.amplitude_m * w.k_north_rad_m / w.k_rad_m
        )
        node.inputs["Lift"].default_value = _lanes(triple, lambda w: w.amplitude_m)
        sums = node.outputs
    offset = tree.nodes.new("ShaderNodeCombineXYZ")
    if sums is not None:
        for axis, name, sign in (("X", "Sum X", -1.0), ("Y", "Sum Y", -1.0)):
            tree.links.new(
                _math(tree, "MULTIPLY", _total(tree, sums[name]), sign),
                offset.inputs[axis],
            )
        tree.links.new(_total(tree, sums["Sum Z"]), offset.inputs["Z"])
    return offset.outputs["Vector"]


def displaced(
    drawn: tuple[tuple[Wave, ...], tuple[Wave, ...]],
) -> tuple[tuple[Wave, ...], tuple[tuple[Wave, ...], tuple[Wave, ...]]]:
    """(the waves drawn as geometry, each sea's waves drawn as shading): the tallest
    of each sea on its own, or a swell's height would decide which wind waves count,
    down to `DISPLACED_MIN_M`."""
    tall = tuple(
        w
        for field in drawn
        for w in tallest(field, DISPLACED_HEIGHT_SHARE)
        if 2 * math.pi / w.k_rad_m >= DISPLACED_MIN_M
    )
    shaded = (
        tuple(w for w in drawn[0] if w not in tall),
        tuple(w for w in drawn[1] if w not in tall),
    )
    return tall, shaded


def _displace(
    material: bpy.types.Material,
    output: bpy.types.Node,
    offset_m: bpy.types.NodeSocket,
) -> None:
    """Move `material`'s surface by `offset_m`, in the object's frame."""
    tree = material.node_tree
    offset = tree.nodes.new("ShaderNodeVectorDisplacement")
    offset.space = "OBJECT"
    # Both default to a fraction: Midlevel 0.5 on some nodes, Scale 0.01 on all.
    offset.inputs["Midlevel"].default_value = 0.0
    offset.inputs["Scale"].default_value = 1.0
    tree.links.new(offset_m, offset.inputs["Vector"])
    tree.links.new(offset.outputs["Displacement"], output.inputs["Displacement"])
    material.displacement_method = "DISPLACEMENT"


def _flat(
    tree: bpy.types.NodeTree, time_s: bpy.types.NodeSocket
) -> bpy.types.NodeSocket:
    """(east, north, time) of the flat sea a point was displaced from: the UV map
    `water` writes, which displacement leaves where it was."""
    uv = tree.nodes.new("ShaderNodeUVMap")
    uv.uv_map = "UVMap"
    flat = tree.nodes.new("ShaderNodeSeparateXYZ")
    tree.links.new(uv.outputs["UV"], flat.inputs["Vector"])
    xyt = tree.nodes.new("ShaderNodeCombineXYZ")
    xyt.name = "sea_rest"
    tree.links.new(flat.outputs["X"], xyt.inputs["X"])
    tree.links.new(flat.outputs["Y"], xyt.inputs["Y"])
    tree.links.new(time_s, xyt.inputs["Z"])
    return xyt.outputs["Vector"]


def _normal(
    tree: bpy.types.NodeTree,
    wind: tuple[Wave, ...],
    swell: tuple[Wave, ...],
    position: bpy.types.NodeSocket,
    pixel: _Pixel,
    calm: bpy.types.NodeSocket | None = None,
    damped: frozenset[Wave] = frozenset(),
) -> bpy.types.NodeSocket:
    """The normal of what the pixel resolves at `position`; `calm` scales the `damped`
    waves."""
    geometry = tree.nodes.new("ShaderNodeNewGeometry")
    # Each sea in triples of its own, so the wind's acceleration ends with its last.
    triples = _triples(wind) + _triples(swell)
    drawn = _instance(
        tree, _normal_group(triples, len(_triples(wind)), damped), pixel, calm
    )
    tree.links.new(position, drawn.inputs["Position"])
    tilted = _vector(tree, "ADD", geometry.outputs["Normal"], drawn.outputs["Gradient"])
    return _vector(tree, "NORMALIZE", tilted, name="wave_normal")


def water(sea: Sea, reach_m: float, material: bpy.types.Material) -> bpy.types.Object:
    """A grid curved to the earth, out to `reach_m`. The waves are in `material`."""
    bpy.ops.mesh.primitive_grid_add(
        x_subdivisions=SEA_CELLS, y_subdivisions=SEA_CELLS, size=2 * reach_m
    )
    water = bpy.context.object
    water.name = "sea"
    place(water, 0.0, 0.0, 0.0)
    radius_m = earth_radius_m(sea.refraction_k)
    for vertex in water.data.vertices:
        vertex.co.z = sea_z_m(vertex.co.x, vertex.co.y, radius_m)
    # The flat sea's east and north, which the waves are a function of.
    uv = water.data.uv_layers["UVMap"].data
    for loop in water.data.loops:
        co = water.data.vertices[loop.vertex_index].co
        uv[loop.index].uv = (co.x, co.y)
    dice = water.modifiers.new("dice", "SUBSURF")
    dice.subdivision_type = "SIMPLE"
    dice.levels = dice.render_levels = 0
    dice.use_adaptive_subdivision = True
    dice.adaptive_pixel_size = DICING_PX
    # Near the camera a pixel is millimetres, and dicing to it is most of the render.
    # Cycles caps the dicing in halvings of a cell.
    bpy.context.scene.cycles.max_subdivisions = math.ceil(
        math.log2(2 * reach_m / SEA_CELLS / DICED_EDGE_M)
    )
    bpy.context.scene.cycles.offscreen_dicing_scale = OFFSCREEN_DICING
    # Flat faces would show their edges in the specular.
    for face in water.data.polygons:
        face.use_smooth = True
    water.data.materials.append(material)
    return water
