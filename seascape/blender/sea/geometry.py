"""The sea's grid, and the waves it draws: tall ones displaced, the rest as normals."""

import math
from collections.abc import Callable

import bpy

from seascape.blender.blend import (
    combine,
    feed,
    node,
    place,
    scalar,
    separate,
    vector,
)
from seascape.blender.sea.pixel import _instance, _Pixel, _pixel
from seascape.config import Scenario, Sea
from seascape.physics.earth import earth_radius_m, horizon_m, sea_z_m
from seascape.physics.waves import Wave, fade_axis, tallest

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

# Judgement: a margin on the horizon, or the grid's own edge becomes the horizon.
SEA_MARGIN = 1.5


def sea_reach_m(scenario: Scenario) -> float:
    """Half-width of the sea, a margin past the highest rig's horizon from as far
    from the origin as the ownship sails: a turn only shortens the run.
    """
    height_m = max(rig.height_m for rig in scenario.rigs.values())
    run_m = scenario.ownship.speed_mps * scenario.outputs.duration_s
    return SEA_MARGIN * horizon_m(height_m, scenario.sea.refraction_k) + run_m


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
    outputs and Group Output's inputs.
    """
    group = bpy.data.node_groups.new(name, "ShaderNodeTree")
    for sockets, in_out in ((inputs, "INPUT"), (outputs, "OUTPUT")):
        for socket, kind in sockets:
            group.interface.new_socket(
                socket, in_out=in_out, socket_type=f"NodeSocket{kind}"
            )
    given = group.nodes.new("NodeGroupInput").outputs
    return group, given, group.nodes.new("NodeGroupOutput").inputs


def _total(
    tree: bpy.types.NodeTree, lanes: bpy.types.NodeSocket
) -> bpy.types.NodeSocket:
    return vector(tree, "DOT_PRODUCT", lanes, (1.0, 1.0, 1.0))


# What the pixel resolves: shared by every triple.
_FOOTPRINT = (("Along", "Vector"), ("Across Sq", "Float"), ("Stretch", "Float"))
# How each lane fades and calms.
_FADE = (
    ("Tx", "Vector"),
    ("Ty", "Vector"),
    ("Band", "Vector"),
    ("Half K Sq", "Vector"),
    ("Damped", "Vector"),
)
_SLOPE = (("Slope X", "Vector"), ("Slope Y", "Vector"))
_SUMS = (("Sum X", "Vector"), ("Sum Y", "Vector"), ("Sum Z", "Vector"))


def _fade(
    group: bpy.types.NodeTree, given: bpy.types.NodeOutputs, calm: bool
) -> bpy.types.NodeSocket:
    """How much of each lane the pixel draws, `waves.visibility` at its footprint
    along the wave, footprint^2 = across^2 + (along^2 - across^2) cos^2: sin(y) / y
    at y = footprint k / 2, held at 0 past pi; scaled by `Calm` in a `Damped` lane.

    cos^2 is `fade_axis`'s. A judgement: the wave fades at the band's RMS footprint,
    not by the band's mean visibility.

    Built alike in every group, so Cycles merges a triple's fade across them.
    """
    along_x, along_y, _ = separate(group, given["Along"])
    heading = vector(
        group,
        "MULTIPLY_ADD",
        given["Ty"],
        along_y,
        vector(group, "MULTIPLY", given["Tx"], along_x),
    )
    footprint_sq = vector(
        group,
        "MULTIPLY_ADD",
        vector(group, "MULTIPLY_ADD", heading, heading, given["Band"]),
        given["Stretch"],
        given["Across Sq"],
    )
    y = vector(
        group,
        "POWER",
        vector(group, "MULTIPLY", footprint_sq, given["Half K Sq"]),
        (0.5, 0.5, 0.5),
    )
    # An empty lane's y is 0, and Vector Math divides by 0 to 0.
    seen = vector(
        group,
        "DIVIDE",
        vector(group, "SINE", vector(group, "MINIMUM", y, (math.pi,) * 3)),
        y,
    )
    if calm:
        less = scalar(group, "SUBTRACT", given["Calm"], 1.0)
        scale = vector(group, "MULTIPLY_ADD", given["Damped"], less, (1.0, 1.0, 1.0))
        seen = vector(group, "MULTIPLY", seen, scale)
    return seen


def _wave_group(calm: bool) -> bpy.types.NodeTree:
    """Three waves, each as far as `_fade` says the pixel resolves it. Each adds its
    slope sin(phase) to `Sum X` and `Sum Y` and `Lift` cos(phase) to `Sum Z`, per lane:
    a downward acceleration in g when `Lift` is a k, a height when it is a.
    """
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
    east, north, time_s = separate(group, given["Position"])
    phase = vector(
        group,
        "MULTIPLY_ADD",
        given["Kx"],
        east,
        vector(
            group,
            "MULTIPLY_ADD",
            given["Ky"],
            north,
            vector(group, "MULTIPLY_ADD", given["W"], time_s, given["Phase"]),
        ),
    )
    seen = _fade(group, given, calm)

    def shown(trig: str) -> bpy.types.NodeSocket:
        return vector(group, "MULTIPLY", vector(group, trig, phase), seen)

    sine = shown("SINE")
    # -d height / dx of a cos(phase) is a k_x sin(phase).
    for total, factor, wave in (
        ("Sum X", "Slope X", sine),
        ("Sum Y", "Slope Y", sine),
        ("Sum Z", "Lift", shown("COSINE")),
    ):
        group.links.new(
            vector(group, "MULTIPLY_ADD", wave, given[factor], given[total]),
            out[total],
        )
    return group


def _lanes(triple: Triple, value: Callable[[Wave], float]) -> tuple:
    return tuple(0.0 if wave is None else value(wave) for wave in triple)


def _set_triple(
    inputs: bpy.types.NodeInputs,
    triple: Triple,
    damped: frozenset[Wave],
    lift: Callable[[Wave], float],
) -> None:
    """Set the inputs of `_fade`, the phases that place three waves, and each wave's
    lift, split along its heading for the slopes. An empty lane has no amplitude, so
    whatever it fades adds 0.

    Args:
        inputs: A wave group node's inputs, set as constants.
        triple: The waves, one a lane; None leaves a lane empty.
        damped: The waves a slick calms.
        lift: What each wave adds up its lane: its amplitude for the surface, its
            slope for the normal.
    """
    axes = {w: fade_axis(w) for w in triple if w is not None}
    sets = {
        "Tx": _lanes(triple, lambda w: axes[w].tx),
        "Ty": _lanes(triple, lambda w: axes[w].ty),
        "Band": _lanes(triple, lambda w: axes[w].band),
        "Half K Sq": _lanes(triple, lambda w: (w.k_rad_m / 2) ** 2),
        "Damped": _lanes(triple, lambda w: float(w in damped)),
        "Kx": _lanes(triple, lambda w: w.k_east_rad_m),
        "Ky": _lanes(triple, lambda w: w.k_north_rad_m),
        "W": _lanes(triple, lambda w: -w.omega_rad_s),
        "Phase": _lanes(triple, lambda w: w.phase_rad),
        "Lift": _lanes(triple, lift),
        "Slope X": _lanes(triple, lambda w: lift(w) * math.sin(w.toward_rad)),
        "Slope Y": _lanes(triple, lambda w: lift(w) * math.cos(w.toward_rad)),
    }
    for name, value in sets.items():
        inputs[name].default_value = value


def _normal_group(
    triples: list[Triple], wind_triples: int, damped: frozenset[Wave]
) -> bpy.types.NodeTree:
    """The slope the pixel draws at `Position`, and the wind sea's downward
    acceleration in g. A group, as each node in a tree slows every edit to it.

    Args:
        triples: The waves, three a group, the wind sea's first.
        wind_triples: How many of the triples are the wind sea's.
        damped: The waves `Calm` scales.
    """
    group, given, out = _group(
        "sea_waves",
        (("Position", "Vector"), *_FOOTPRINT, ("Calm", "Float")),
        (("Gradient", "Vector"), ("Acceleration", "Float")),
    )
    sums = None
    for i, triple in enumerate(triples):
        calmed = not damped.isdisjoint(triple)
        waves = group.nodes.new("ShaderNodeGroup")
        waves.node_tree = _wave_group(calmed)
        waves.name = f"waves_{i}"
        feed(group, waves.inputs["Position"], given["Position"])
        for name, _ in _FOOTPRINT:
            feed(group, waves.inputs[name], given[name])
        if calmed:
            feed(group, waves.inputs["Calm"], given["Calm"])
        if sums is not None:
            for name, _ in _SUMS:
                feed(group, waves.inputs[name], sums[name])
        _set_triple(waves.inputs, triple, damped, lambda w: w.amplitude_m * w.k_rad_m)
        sums = waves.outputs
        if i == wind_triples - 1:
            group.links.new(_total(group, sums["Sum Z"]), out["Acceleration"])
    if sums is not None:
        gradient = combine(
            group, _total(group, sums["Sum X"]), _total(group, sums["Sum Y"])
        )
        group.links.new(gradient, out["Gradient"])
    return group


def _displacement(
    tree: bpy.types.NodeTree, tall: tuple[Wave, ...], position: bpy.types.NodeSocket
) -> bpy.types.NodeSocket:
    """The offset of a Gerstner surface from the flat sea, each wave faded by the
    footprint of the micropolygon there: up a cos(phase), and -k_hat a sin(phase)
    along the sea (Gerstner 1802; Tessendorf 2001, eq. 29, with lambda = 1). Adaptive
    subdivision dices at about a pixel, so the micropolygon's footprint is the pixel's.

    Args:
        tree: The sea material's shader tree.
        tall: The waves drawn as geometry.
        position: The flat sea's (east, north, time) the offset is taken at.
    """
    camera = tree.nodes.new("ShaderNodeCameraData")
    # From the camera, in world space: the Geometry node's Incoming is unset here.
    toward = node(
        tree,
        "ShaderNodeVectorTransform",
        {"Vector": camera.outputs["View Vector"]},
        vector_type="VECTOR",
        convert_from="CAMERA",
        convert_to="WORLD",
    )
    pixel = _pixel(tree, toward.outputs["Vector"])
    sums = None
    for triple in _triples(tall):
        group_node = _instance(tree, _wave_group(calm=False), pixel, None)
        tree.links.new(position, group_node.inputs["Position"])
        if sums is not None:
            for name, _ in _SUMS:
                tree.links.new(sums[name], group_node.inputs[name])
        _set_triple(group_node.inputs, triple, frozenset(), lambda w: w.amplitude_m)
        sums = group_node.outputs
    if sums is None:
        return combine(tree)
    x, y, z = (_total(tree, sums[name]) for name in ("Sum X", "Sum Y", "Sum Z"))
    return combine(
        tree, scalar(tree, "MULTIPLY", x, -1.0), scalar(tree, "MULTIPLY", y, -1.0), z
    )


def displaced(
    drawn: tuple[tuple[Wave, ...], tuple[Wave, ...]],
) -> tuple[tuple[Wave, ...], tuple[tuple[Wave, ...], tuple[Wave, ...]]]:
    """(the waves drawn as geometry, each sea's waves drawn as shading): the tallest
    of each sea on its own, or a swell's height would decide which wind waves count,
    down to `DISPLACED_MIN_M`.
    """
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
    # Midlevel and Scale default to a fraction: 0.5 on some nodes, 0.01 on all.
    offset = node(
        tree,
        "ShaderNodeVectorDisplacement",
        {"Midlevel": 0.0, "Scale": 1.0, "Vector": offset_m},
        space="OBJECT",
    )
    tree.links.new(offset.outputs["Displacement"], output.inputs["Displacement"])
    material.displacement_method = "DISPLACEMENT"


def _flat(
    tree: bpy.types.NodeTree, time_s: bpy.types.NodeSocket
) -> bpy.types.NodeSocket:
    """(east, north, time) of the flat sea a point was displaced from: the UV map
    `water` writes, which displacement leaves where it was.
    """
    uv = node(tree, "ShaderNodeUVMap", uv_map="UVMap")
    east, north, _ = separate(tree, uv.outputs["UV"])
    xyt = combine(tree, east, north, time_s)
    xyt.node.name = "sea_rest"
    return xyt


def _normal(
    tree: bpy.types.NodeTree,
    wind: tuple[Wave, ...],
    swell: tuple[Wave, ...],
    position: bpy.types.NodeSocket,
    pixel: _Pixel,
    calm: bpy.types.NodeSocket | None,
    damped: frozenset[Wave],
) -> bpy.types.NodeSocket:
    """The normal of what the pixel resolves.

    Args:
        tree: The sea material's shader tree.
        wind: The wind sea's shaded waves.
        swell: The swell's shaded waves.
        position: The flat sea's (east, north, time) the normal is taken at.
        pixel: The footprint the waves fade by.
        calm: Scales the damped waves; None for no slick.
        damped: The waves a slick calms.
    """
    geometry = tree.nodes.new("ShaderNodeNewGeometry")
    # Each sea in triples of its own, so the wind's acceleration ends with its last.
    triples = _triples(wind) + _triples(swell)
    drawn = _instance(
        tree, _normal_group(triples, len(_triples(wind)), damped), pixel, calm
    )
    tree.links.new(position, drawn.inputs["Position"])
    tilted = vector(tree, "ADD", geometry.outputs["Normal"], drawn.outputs["Gradient"])
    return vector(tree, "NORMALIZE", tilted, name="wave_normal")


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
