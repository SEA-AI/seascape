"""The sea's mesh and material: waves as shading on a grid curved to the earth.

Sources
-------
Microfacet lobe: Walter, Marschner, Li & Torrance, "Microfacet models for refraction
through rough surfaces", EGSR 2007 (doi:10.2312/EGWR/EGSR07/195-206) for Beckmann's
distribution and its Smith shadowing; Burley, "Physically-based shading at Disney",
SIGGRAPH 2012 course notes, for the alpha = roughness^2 convention Cycles follows.

Parallax: Kaneko et al., "Detailed shape representation with parallax mapping", ICAT
2001 205-208; its offset capped as in Welsh, "Parallax mapping with offset limiting: a
per-pixel approximation of uneven surfaces", Infiscape 2004.

Whitecaps: Koepke, "Effective reflectance of oceanic whitecaps", Applied Optics 23(11)
1816, 1984 (doi:10.1364/AO.23.001816), for the Monahan coverage it multiplies.

Seawater: Quan & Fry, "Empirical equation for the index of refraction of seawater",
Applied Optics 34(18) 3477, 1995 (doi:10.1364/AO.34.003477); Morel & Maritorena, "Bio-
optical properties of oceanic waters: a reappraisal", JGR 106(C4) 7163, 2001
(doi:10.1029/2000JC000319); Lee, Carder & Arnone, "Deriving inherent optical properties
from water color: a multiband quasi-analytical algorithm for optically deep waters",
Applied Optics 41(27) 5755, 2002 (doi:10.1364/AO.41.005755).
"""

import math
from collections.abc import Callable
from functools import cache
from pathlib import Path
from statistics import NormalDist
from typing import NamedTuple

import bpy
import numpy as np

from seascape import lwir
from seascape.blend import (
    CURVE_SAMPLES,
    SUN,
    animate,
    curve_image,
    drive,
    lookup,
    place,
)
from seascape.config import Outputs, Scenario, Sea
from seascape.wakes import (
    BUBBLE_EFOLD_S,
    BUBBLE_GAIN,
    FOAM_EFOLD_S,
    FRESH_FOAM_REFLECTANCE,
    KELVIN_HALF_ANGLE_RAD,
    MAX_STEEPNESS,
    NARROWING_FROUDE,
    SEEN_SLOPE_VARIANCE,
    Wake,
)
from seascape.water import body_color
from seascape.waves import (
    CDF_SIGMAS,
    FADE_FOOTPRINTS,
    GRAVITY_MS2,
    GUST_LENGTH_M,
    SLICK_DRIFT,
    WHITECAP_AREA_EFOLD_S,
    WINDROW_ASPECT,
    WINDROW_MIN_WIND_MPS,
    WINDROW_SPACING_S,
    Wave,
    breakers,
    cox_munk_slick_variances,
    cox_munk_slope,
    cox_munk_variances,
    earth_radius_m,
    fade_footprints_m,
    foam_ages_s,
    foam_edge_g,
    foam_fade_efold_s,
    gust_slope_variance,
    gusty_whitecap_fraction,
    horizon_m,
    resolvable,
    sea_z_m,
    slick_survivors,
    specular_cell_m2,
    tallest,
    turbulence_intensity,
    twinkle_hz,
    unresolved_acceleration_variance,
    unresolved_slope_variances,
    von_karman_field,
    whitecap_fraction,
    whitecap_thresholds_g,
)

# Quan & Fry 1995 at 550 nm, salinity 35, 15 C.
SEAWATER_IOR = 1.341

# The Sky Texture's default sun_size, a diameter. A facet tilted by d turns the
# reflection by 2 d, so the sun spans half its radius in slope.
SUN_SLOPE_RADIUS = math.radians(0.545) / 4

# A judgement: the span of d^2 / v over which the glitter's tilt fades out, d the tilt
# off the drawn normal of the facet that mirrors the sun, v the unresolved variance. The
# slope density there is exp(-d^2 / v) of its peak.
GLINT_REACH = (9.0, 16.0)

# A judgement: fixed-point steps to where the view ray meets the drawn height.
PARALLAX_STEPS = 3
# A judgement: the share of facets on which each fixed-point step contracts. A step
# contracts where the slope along the run times the run is under 1 (Banach); with
# slopes Gaussian per axis (Cox & Munk 1954), capping the run at 1 / z the rms slope
# along it covers facets within z sigma, two-sided.
PARALLAX_CONVERGED = 0.996
# A judgement: parallax reads the fewest waves carrying this share of their own sea's
# height variance.
PARALLAX_HEIGHT_SHARE = 0.999


# A judgement: the footprints a sea pixel spans.
FOOTPRINT_RANGE_M = (1e-3, 1e4)

# A judgement: emissivity and the reflected sky move slowly with slope.
SLOPE_ROWS = 16

# Elevation (rad, ascending) and the IR sky's radiance there.
type SkyCurve = tuple[np.ndarray, np.ndarray]


class Light(NamedTuple):
    """The ir sky round the horizon, or the unit vector an eo sea glints towards. A sky
    makes the sea ir."""

    sky: SkyCurve | None = None
    sun: tuple[float, float, float] | None = None


# Cells per side: enough for the tangent point to land on a face, not an accuracy
# knob. A cell's sagitta, width^2 / 8R, is far under a pixel at the horizon.
SEA_CELLS = 128

# Margin on the horizon, or the grid's own edge becomes the horizon.
SEA_MARGIN = 1.5

# A judgement: gusts to about ten metres, and a tile of dozens of integral scales, so
# its repeat is lost in the distance. Blender's Noise Texture would not repeat, but
# carries no Kaimal spectrum.
GUST_CELLS = 512
GUST_SPACING_M = 4.0
GUST_TILE_M = GUST_CELLS * GUST_SPACING_M
# A judgement: whitecap foam's finer texture, the tile again this much finer and
# weighted so.
LACE_OCTAVE = 6.0
LACE_FINE_WEIGHT = 0.8
# A judgement: the cells foam tears into, and how far Blender's Noise bends their
# edges, in cells.
FOAM_CELL_M = 0.3
FOAM_WARP = 0.6
# A judgement: a line's soft edge, in cells, and how sharply a segment of it goes.
FOAM_LINE_SOFT = 0.02
FOAM_POP_GAIN = 20.0
_NETWORK_CSV = Path(__file__).parent / "data" / "foam_network.csv"

# A judgement: foam patches a metre or two across, streaked 4:1 along the wake.
FOAM_CELLS = 512
FOAM_SPACING_M = 0.25
FOAM_LENGTH_M = 1.0
FOAM_TILE_M = FOAM_CELLS * FOAM_SPACING_M
FOAM_STREAK = 4.0


def sea_reach_m(scenario: Scenario) -> float:
    """Half-width of the sea, a margin past the highest rig's horizon."""
    height_m = max(rig.height_m for rig in scenario.rigs.values())
    return SEA_MARGIN * horizon_m(height_m, scenario.sea.refraction_k)


def _row_sigmas(slope_max: float) -> np.ndarray:
    """Per-axis unresolved RMS slope up a table's rows, to the total `slope_max`, at
    texel centres."""
    return (np.arange(SLOPE_ROWS) + 0.5) / SLOPE_ROWS * slope_max / math.sqrt(2)


def _emissivity_image(t_sea_k: float, slope_max: float) -> bpy.types.Image:
    """`lwir.emissivity_curve` baked against cos(theta) along a row and the unresolved
    RMS slope up the rows, both at texel centres, which is what the shader samples.

    The curve is sampled uniformly in angle; the shader's dot product is uniform in its
    cosine, so it is resampled here rather than corrected in nodes.
    """
    mu = (np.arange(CURVE_SAMPLES) + 0.5) / CURVE_SAMPLES
    rows = []
    for sigma in _row_sigmas(slope_max):
        theta, eps = lwir.emissivity_curve(t_sea_k=t_sea_k, slope_sigma=sigma)
        rows.append(np.interp(mu, np.cos(theta)[::-1], eps[::-1]))
    return curve_image("sea_emissivity", np.array(rows))


def _reflection_image(
    t_sea_k: float, sky: SkyCurve, slope_max: float
) -> bpy.types.Image:
    """`lwir.reflected_sky` baked against sin(mirror elevation) along a row and the
    unresolved RMS slope up the rows, both at texel centres."""
    mirror = np.arcsin((np.arange(CURVE_SAMPLES) + 0.5) / CURVE_SAMPLES)
    rows = [
        lwir.reflected_sky(mirror, *sky, t_sea_k=t_sea_k, slope_sigma=sigma)
        for sigma in _row_sigmas(slope_max)
    ]
    return curve_image("sea_reflection", np.array(rows))


def _sea_time(tree: bpy.types.NodeTree, outputs: Outputs) -> bpy.types.NodeSocket:
    """Seconds since the first frame, modulo the span in a loop."""
    times_s = outputs.times_s
    if not outputs.loop:
        clock = tree.nodes.new("ShaderNodeValue")
        clock.name = "sea_time"
        animate(clock.outputs["Value"], "default_value", times_s, lambda t: t)
        return clock.outputs["Value"]
    # A keyed t jumps at the wrap; every omega is a multiple of 2 pi / span, so t modulo
    # the span is exact.
    span_s = outputs.span_s
    turn = [tree.nodes.new("ShaderNodeValue") for _ in range(2)]
    names, trigs = ("sea_cos", "sea_sin"), (math.cos, math.sin)
    for node, name, trig in zip(turn, names, trigs, strict=True):
        node.name = name
        animate(
            node.outputs["Value"],
            "default_value",
            times_s,
            lambda t, trig=trig: trig(2 * math.pi * t / span_s),
        )
    angle = tree.nodes.new("ShaderNodeMath")
    angle.operation = "ARCTAN2"
    clock = tree.nodes.new("ShaderNodeMath")
    clock.operation = "MULTIPLY"
    clock.name = "sea_time"
    clock.inputs["Value_001"].default_value = span_s / (2 * math.pi)
    link = tree.links.new
    link(turn[1].outputs["Value"], angle.inputs["Value"])
    link(turn[0].outputs["Value"], angle.inputs["Value_001"])
    link(angle.outputs["Value"], clock.inputs["Value"])
    return clock.outputs["Value"]


def _vector(
    tree: bpy.types.NodeTree,
    operation: str,
    *inputs: bpy.types.NodeSocket | float,
    name: str | None = None,
) -> bpy.types.NodeSocket:
    node = tree.nodes.new("ShaderNodeVectorMath")
    node.operation = operation
    if name:
        node.name = name
    sockets = list(node.inputs)
    if operation == "SCALE":
        sockets = [node.inputs["Vector"], node.inputs["Scale"]]
    for socket, value in zip(sockets, inputs, strict=False):
        if isinstance(value, bpy.types.NodeSocket):
            tree.links.new(value, socket)
        else:
            socket.default_value = value
    scalar = operation in {"DOT_PRODUCT", "LENGTH", "DISTANCE"}
    return node.outputs["Value" if scalar else "Vector"]


def _math(
    tree: bpy.types.NodeTree,
    operation: str,
    *inputs: float | bpy.types.NodeSocket,
    name: str | None = None,
) -> bpy.types.NodeSocket:
    """A Math node. Unset inputs read 0.5."""
    node = tree.nodes.new("ShaderNodeMath")
    node.operation = operation
    if name:
        node.name = name
    for socket, value in zip(node.inputs, inputs, strict=False):
        if isinstance(value, bpy.types.NodeSocket):
            tree.links.new(value, socket)
        else:
            socket.default_value = value
    return node.outputs["Value"]


class _Pixel(NamedTuple):
    across_m: bpy.types.NodeSocket
    along_m: bpy.types.NodeSocket
    along_dir: bpy.types.NodeSocket  # unit, horizontal


def _pixel(tree: bpy.types.NodeTree) -> _Pixel:
    """The footprint of a pixel of the scene's perspective camera at the render's
    resolution, Resolution % included."""
    camera = tree.nodes.new("ShaderNodeCameraData")
    geometry = tree.nodes.new("ShaderNodeNewGeometry")
    facing = _vector(
        tree, "DOT_PRODUCT", geometry.outputs["Incoming"], geometry.outputs["Normal"]
    )
    across = _math(tree, "MULTIPLY", camera.outputs["View Distance"], 0.0)
    # ponytail: the build's cameras fit their angle to the width; a camera added in
    # Blender with another sensor fit gets the wrong pixel. Read `sensor_fit` when
    # such a camera must render.
    drive(
        across.node.inputs[1],
        "default_value",
        "angle / (width * percent / 100)",
        bpy.context.scene,
        angle="camera.data.angle_x",
        width="render.resolution_x",
        percent="render.resolution_percentage",
    )
    # Grazing stretches the pixel by 1 / cos along the view; the floor keeps it finite.
    cosine = _math(tree, "MAXIMUM", _math(tree, "ABSOLUTE", facing), 1e-3)
    tilt = _vector(tree, "SCALE", geometry.outputs["Normal"])
    tree.links.new(facing, tilt.node.inputs["Scale"])
    level = _vector(tree, "SUBTRACT", geometry.outputs["Incoming"], tilt)
    along = _vector(tree, "NORMALIZE", level)
    return _Pixel(across, _math(tree, "DIVIDE", across, cosine), along)


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
    ("Gone Sq", "Vector"),
    ("Whole Sq", "Vector"),
    ("Damped", "Vector"),
)
_SLOPE = (("Slope X", "Vector"), ("Slope Y", "Vector"))
_SUMS = (("Sum X", "Vector"), ("Sum Y", "Vector"), ("Sum Z", "Vector"))


def _fade(
    group: bpy.types.NodeTree, given: bpy.types.NodeOutputs, calm: bool
) -> tuple[bpy.types.NodeSocket, tuple[bpy.types.NodeSocket, bpy.types.NodeSocket]]:
    """How much of each lane the pixel draws: 1 to 0 as its footprint along the wave,
    footprint^2 = across^2 + (along^2 - across^2) cos^2, widens from `Whole Sq` to
    `Gone Sq`; scaled by `Calm` in a `Damped` lane. And `Along`, (x, y), spread.

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
    fade = group.nodes.new("ShaderNodeMapRange")
    # First: a name finds the enabled socket, and the float ones are until then.
    fade.data_type = "FLOAT_VECTOR"
    fade.interpolation_type = "SMOOTHSTEP"
    # From Min above From Max: a wider footprint fades the wave out.
    group.links.new(given["Gone Sq"], fade.inputs["From Min"])
    group.links.new(given["Whole Sq"], fade.inputs["From Max"])
    group.links.new(
        _vector(
            group,
            "MULTIPLY_ADD",
            _vector(group, "MULTIPLY", heading, heading),
            _spread(group, given["Stretch"]),
            _spread(group, given["Across Sq"]),
        ),
        fade.inputs["Vector"],
    )
    seen = fade.outputs["Vector"]
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


def _run_variance_group() -> bpy.types.NodeTree:
    """Three waves' slope variance along `Along`, faded as `_wave_group` draws them,
    added to `Variance` per lane, twice over: (Slope . Along)^2."""
    name = "waves_run"
    if name in bpy.data.node_groups:
        return bpy.data.node_groups[name]
    group, given, out = _group(
        name,
        (*_FOOTPRINT, ("Variance", "Vector"), *_FADE, *_SLOPE),
        (("Variance", "Vector"),),
    )
    seen, (along_x, along_y) = _fade(group, given, calm=False)
    run = _vector(
        group,
        "MULTIPLY",
        _vector(
            group,
            "MULTIPLY_ADD",
            given["Slope Y"],
            along_y,
            _vector(group, "MULTIPLY", given["Slope X"], along_x),
        ),
        seen,
    )
    group.links.new(
        _vector(group, "MULTIPLY_ADD", run, run, given["Variance"]), out["Variance"]
    )
    return group


def _lanes(triple: Triple, value: Callable[[Wave], float], pad: float = 0.0) -> tuple:
    return tuple(pad if wave is None else value(wave) for wave in triple)


def _set_triple(
    inputs: bpy.types.NodeInputs, triple: Triple, damped: frozenset[Wave]
) -> None:
    """The inputs of `_fade` and the slopes that place `triple`. An empty lane has no
    amplitude, so whatever it fades adds 0."""

    def wavelength(wave: Wave) -> float:
        return 2 * math.pi / wave.k_rad_m

    sets = {
        "Tx": _lanes(triple, lambda w: math.sin(w.toward_rad)),
        "Ty": _lanes(triple, lambda w: math.cos(w.toward_rad)),
        "Gone Sq": _lanes(
            triple, lambda w: float(fade_footprints_m(wavelength(w))[0]) ** 2, 1.0
        ),
        "Whole Sq": _lanes(
            triple, lambda w: float(fade_footprints_m(wavelength(w))[1]) ** 2
        ),
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


def _height_group(tall: tuple[Wave, ...]) -> bpy.types.NodeTree:
    """The drawn height of `tall` at `Position`, (east, north, time), faded as the
    normal is. One group, so each fixed-point step adds one node to the sea's tree."""
    group, given, out = _group(
        "sea_height",
        (("Position", "Vector"), *_FOOTPRINT),
        (("Height", "Float"),),
    )
    heights = None
    for triple in _triples(tall):
        node = group.nodes.new("ShaderNodeGroup")
        node.node_tree = _wave_group(calm=False)
        group.links.new(given["Position"], node.inputs["Position"])
        _feed(group, given, node, calmed=False)
        if heights is not None:
            group.links.new(heights, node.inputs["Sum Z"])
        _set_triple(node.inputs, triple, frozenset())
        node.inputs["Lift"].default_value = _lanes(triple, lambda w: w.amplitude_m)
        heights = node.outputs["Sum Z"]
    if heights is not None:
        group.links.new(_total(group, heights), out["Height"])
    return group


def _parallax(
    tree: bpy.types.NodeTree,
    time_s: bpy.types.NodeSocket,
    height_at: Callable[[bpy.types.NodeSocket], bpy.types.NodeSocket],
    run_variance: float | bpy.types.NodeSocket,
) -> bpy.types.NodeSocket:
    """Where the view ray meets the drawn height, (east, north, time), iterated from the
    flat sea towards its fixed point: x <- x0 + h(x) cot(elevation) along the ray. A
    face tilted from the camera then takes the little of the frame it shows.
    """
    geometry = tree.nodes.new("ShaderNodeNewGeometry")
    flat = tree.nodes.new("ShaderNodeSeparateXYZ")
    tree.links.new(geometry.outputs["Position"], flat.inputs["Vector"])
    toward = tree.nodes.new("ShaderNodeSeparateXYZ")
    tree.links.new(geometry.outputs["Incoming"], toward.inputs["Vector"])
    run = _math(
        tree,
        "SQRT",
        _math(
            tree,
            "MULTIPLY_ADD",
            toward.outputs["X"],
            toward.outputs["X"],
            _math(tree, "MULTIPLY", toward.outputs["Y"], toward.outputs["Y"]),
        ),
    )
    slope = _math(tree, "SQRT", run_variance)
    reach = _math(
        tree,
        "DIVIDE",
        1 / NormalDist().inv_cdf((1 + PARALLAX_CONVERGED) / 2),
        _math(tree, "MAXIMUM", _math(tree, "MULTIPLY", slope, run), 1e-9),
    )
    scale = _math(
        tree,
        "MINIMUM",
        _math(tree, "DIVIDE", 1.0, toward.outputs["Z"]),
        reach,
        name="parallax_scale",
    )
    run_east = _math(tree, "MULTIPLY", toward.outputs["X"], scale)
    run_north = _math(tree, "MULTIPLY", toward.outputs["Y"], scale)

    def at(
        east: bpy.types.NodeSocket, north: bpy.types.NodeSocket
    ) -> bpy.types.NodeSocket:
        # Height is left out, so the sea curving under the field cannot slide it.
        xyt = tree.nodes.new("ShaderNodeCombineXYZ")
        tree.links.new(east, xyt.inputs["X"])
        tree.links.new(north, xyt.inputs["Y"])
        tree.links.new(time_s, xyt.inputs["Z"])
        return xyt.outputs["Vector"]

    position = at(flat.outputs["X"], flat.outputs["Y"])
    for _ in range(PARALLAX_STEPS):
        height = height_at(position)
        position = at(
            _math(tree, "MULTIPLY_ADD", run_east, height, flat.outputs["X"]),
            _math(tree, "MULTIPLY_ADD", run_north, height, flat.outputs["Y"]),
        )
    position.node.name = "sea_parallax"
    return position


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


def _run_group(triples: list[Triple]) -> bpy.types.NodeTree:
    """The slope variance the pixel draws along the view, which is parallax's run. A
    group, as each node in a tree slows every edit to it."""
    group, given, out = _group("sea_run", _FOOTPRINT, (("Variance", "Float"),))
    variance = None
    for triple in triples:
        node = group.nodes.new("ShaderNodeGroup")
        node.node_tree = _run_variance_group()
        _feed(group, given, node, calmed=False)
        _set_triple(node.inputs, triple, frozenset())
        if variance is not None:
            group.links.new(variance, node.inputs["Variance"])
        variance = node.outputs["Variance"]
    if variance is not None:
        # A wave's variance over a cycle is half its squared amplitude.
        group.links.new(
            _vector(group, "DOT_PRODUCT", variance, (0.5, 0.5, 0.5)), out["Variance"]
        )
    return group


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


def _waves(
    tree: bpy.types.NodeTree,
    wind: tuple[Wave, ...],
    swell: tuple[Wave, ...],
    tall: tuple[Wave, ...],
    time_s: bpy.types.NodeSocket,
    pixel: _Pixel,
) -> bpy.types.NodeSocket:
    """Where the pixel reads the waves, (east, north, time), at the `tall` waves'
    height.

    ponytail: slicks are read at this position, so finding it ignores them. Under a
    slick, a damped wave keeps its height and the run its variance, which caps the
    reach short. If parallax reads wrong under slicks, calm both with a slick read at
    the flat sea.
    """
    heights = _height_group(tall)

    def height_at(position: bpy.types.NodeSocket) -> bpy.types.NodeSocket:
        node = _instance(tree, heights, pixel, None)
        tree.links.new(position, node.inputs["Position"])
        return node.outputs["Height"]

    triples = _triples(wind) + _triples(swell)
    run = _instance(tree, _run_group(triples), pixel, None)
    return _parallax(tree, time_s, height_at, run.outputs["Variance"])


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


def _instance(
    tree: bpy.types.NodeTree,
    group: bpy.types.NodeTree,
    pixel: _Pixel,
    calm: bpy.types.NodeSocket | None,
) -> bpy.types.Node:
    """`group` fed the pixel's footprint. Cycles merges the identical nodes each call
    adds."""
    link = tree.links.new
    across_sq = _math(tree, "MULTIPLY", pixel.across_m, pixel.across_m)
    along_sq = _math(tree, "MULTIPLY", pixel.along_m, pixel.along_m)
    node = tree.nodes.new("ShaderNodeGroup")
    node.node_tree = group
    link(pixel.along_dir, node.inputs["Along"])
    link(across_sq, node.inputs["Across Sq"])
    link(_math(tree, "SUBTRACT", along_sq, across_sq), node.inputs["Stretch"])
    if calm is not None:
        link(calm, node.inputs["Calm"])
    return node


def _history_group(calm: bool) -> bpy.types.NodeTree:
    """Three waves' downward acceleration at each foam age, added to `Sum i` per lane:
    cos(phase) times `Cos i` plus sin(phase) times `Sin i`, the waves faded and calmed
    as `_wave_group` draws them."""
    name = "waves_history_calm" if calm else "waves_history"
    if name in bpy.data.node_groups:
        return bpy.data.node_groups[name]
    ages = range(len(foam_ages_s()))
    group, given, out = _group(
        name,
        (
            ("Position", "Vector"),
            *_FOOTPRINT,
            ("Calm", "Float"),
            ("Kx", "Vector"),
            ("Ky", "Vector"),
            ("W", "Vector"),
            ("Phase", "Vector"),
            *_FADE,
            *(
                (f"{part} {i}", "Vector")
                for i in ages
                for part in ("Cos", "Sin", "Sum")
            ),
        ),
        tuple((f"Sum {i}", "Vector") for i in ages),
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
    cos, sin = (
        _vector(group, "MULTIPLY", _vector(group, trig, phase), seen)
        for trig in ("COSINE", "SINE")
    )
    for i in ages:
        turned = _vector(
            group, "MULTIPLY_ADD", sin, given[f"Sin {i}"], given[f"Sum {i}"]
        )
        group.links.new(
            _vector(group, "MULTIPLY_ADD", cos, given[f"Cos {i}"], turned),
            out[f"Sum {i}"],
        )
    return group


def _crest_history(
    tree: bpy.types.NodeTree,
    wind: tuple[Wave, ...],
    position: bpy.types.NodeSocket,
    pixel: _Pixel,
    calm: bpy.types.NodeSocket | None,
    damped: frozenset[Wave],
) -> list[bpy.types.NodeSocket]:
    """The breaking crests' downward acceleration in g at `position` at each foam age,
    youngest first. An age shifts each wave's phase by omega age:
    a k cos(phase + omega age) = a k cos(omega age) cos(phase)
    - a k sin(omega age) sin(phase), so the cosines are taken once."""
    ages = foam_ages_s()
    group, given, out = _group(
        "sea_crest_history",
        (("Position", "Vector"), *_FOOTPRINT, ("Calm", "Float")),
        tuple((f"Age {i}", "Float") for i in range(len(ages))),
    )
    previous = None
    for triple in _triples(breakers(wind)):
        calmed = not damped.isdisjoint(triple)
        node = group.nodes.new("ShaderNodeGroup")
        node.node_tree = _history_group(calmed)
        group.links.new(given["Position"], node.inputs["Position"])
        _feed(group, given, node, calmed)
        _set_triple(node.inputs, triple, damped)
        for i, age in enumerate(ages):
            node.inputs[f"Cos {i}"].default_value = _lanes(
                triple,
                lambda w, a=age: (
                    w.amplitude_m * w.k_rad_m * math.cos(w.omega_rad_s * a)
                ),
            )
            node.inputs[f"Sin {i}"].default_value = _lanes(
                triple,
                lambda w, a=age: (
                    -w.amplitude_m * w.k_rad_m * math.sin(w.omega_rad_s * a)
                ),
            )
            if previous is not None:
                group.links.new(previous.outputs[f"Sum {i}"], node.inputs[f"Sum {i}"])
        previous = node
    if previous is not None:
        for i in range(len(ages)):
            group.links.new(
                _total(group, previous.outputs[f"Sum {i}"]), out[f"Age {i}"]
            )
    node = _instance(tree, group, pixel, calm)
    tree.links.new(position, node.inputs["Position"])
    return [node.outputs[f"Age {i}"] for i in range(len(ages))]


def _footprint_table(name: str, value_at: Callable[[float], float]) -> bpy.types.Image:
    """`value_at` baked against log footprint: one lookup, not a sum over the waves per
    sample."""
    low, high = FOOTPRINT_RANGE_M
    texel = (np.arange(CURVE_SAMPLES) + 0.5) / CURVE_SAMPLES
    return curve_image(
        name, np.array([value_at(f) for f in low * (high / low) ** texel])
    )


def _at_footprint(
    tree: bpy.types.NodeTree,
    table: bpy.types.Image,
    footprint_m: bpy.types.NodeSocket,
    name: str,
) -> bpy.types.NodeSocket:
    low, high = FOOTPRINT_RANGE_M
    decades = math.log10(high / low)
    x = _math(
        tree,
        "MULTIPLY_ADD",
        _math(tree, "LOGARITHM", footprint_m, 10.0),
        1 / decades,
        -math.log10(low) / decades,
    )
    value = lookup(tree, table, x)
    value.node.name = name
    return value


def _whitecaps(
    tree: bpy.types.NodeTree,
    sea: Sea,
    wind: tuple[Wave, ...],
    history: list[bpy.types.NodeSocket],
    footprint_m: bpy.types.NodeSocket,
    gust: bpy.types.NodeSocket | None,
    gust_max: float,
    lace: tuple[bpy.types.NodeSocket, bpy.types.NodeSocket, bpy.types.NodeSocket],
) -> tuple[bpy.types.NodeSocket, bpy.types.NodeSocket, bpy.types.NodeSocket]:
    """How much of the pixel foam covers, how bright, as a share of fresh foam's, and
    the share of the sea under foam of any age, laced or not, where its bubbles are.
    `history` holds the breaking crests' downward acceleration at each foam age."""
    threshold_g = _breaking(tree, sea, wind, gust, gust_max)
    breaking = breakers(wind)
    table = _footprint_table(
        "sea_unresolved_acceleration",
        lambda f: unresolved_acceleration_variance(breaking, f),
    )
    unresolved = _at_footprint(tree, table, footprint_m, "sea_unresolved_acceleration")
    sigma = _math(
        tree,
        "MAXIMUM",
        _math(tree, "SQRT", _math(tree, "ADD", unresolved, foam_edge_g(wind) ** 2)),
        1e-6,
    )
    mean_g = whitecap_thresholds_g(wind, [whitecap_fraction(sea.wind_speed_mps)])
    fade_s = foam_fade_efold_s(wind, float(mean_g[0]), FRESH_FOAM_REFLECTANCE)
    shares, widths = _network_widths()
    # The fine lace as a uniform level in [g/2, 1 - g/2], g the popping ramp's width,
    # so all kept is solid and none kept is empty.
    sigmas = CDF_SIGMAS * (2 * (np.arange(CURVE_SAMPLES) + 0.5) / CURVE_SAMPLES - 1)
    cdf = curve_image("normal_cdf", np.vectorize(NormalDist().cdf)(sigmas))
    x = _math(tree, "MULTIPLY_ADD", lace[1], 1 / (2 * CDF_SIGMAS), 0.5)
    x.node.use_clamp = True
    level = _math(
        tree,
        "MULTIPLY_ADD",
        lookup(tree, cdf, x),
        1 - 1 / FOAM_POP_GAIN,
        0.5 / FOAM_POP_GAIN,
    )
    area: bpy.types.NodeSocket | float = 0.0
    foam: bpy.types.NodeSocket | float = 0.0
    under: bpy.types.NodeSocket | float = 0.0
    for age, acceleration in zip(foam_ages_s(), history, strict=True):
        node = tree.nodes.new("ShaderNodeGroup")
        node.node_tree = _foam_age_group()
        share = math.exp(-age / fade_s)
        constants = {
            "Thin": math.exp(-age / WHITECAP_AREA_EFOLD_S),
            "Share": share,
            "Wide": float(np.interp(share, shares, widths)),
        }
        for name, value in constants.items():
            node.inputs[name].default_value = value
        for name, socket in (
            ("Acceleration", acceleration),
            ("Threshold", threshold_g),
            ("Sigma", sigma),
            ("Footprint", footprint_m),
            ("Lace", lace[0]),
            ("Level", level),
            ("Network", lace[2]),
            ("Area", area),
            ("Foam", foam),
            ("Under", under),
        ):
            if isinstance(socket, bpy.types.NodeSocket):
                tree.links.new(socket, node.inputs[name])
            else:
                node.inputs[name].default_value = socket
        area, foam, under = (node.outputs[n] for n in ("Area", "Foam", "Under"))
    assert isinstance(area, bpy.types.NodeSocket)
    assert isinstance(foam, bpy.types.NodeSocket)
    area, foam, under = (
        _math(tree, "MAXIMUM", socket, 0.0, name=name)
        for socket, name in (
            (area, "whitecaps"),
            (foam, "whitecap_foam"),
            (under, "whitecap_bubbles"),
        )
    )
    return area, foam, under


def _glinting(
    tree: bpy.types.NodeTree,
    normal: bpy.types.NodeSocket,
    sun: tuple[float, float, float],
    variance: bpy.types.NodeSocket,
) -> bpy.types.NodeSocket:
    """1 where an unresolved facet can mirror a sun above the horizon, fading to 0 by
    `GLINT_REACH`.

    A tilt per cell counts the sun's glints but turns the sky behind each cell into a
    random patch of itself; where nothing can glint, the lobe takes the whole slope.
    """
    geometry = tree.nodes.new("ShaderNodeNewGeometry")
    toward = tree.nodes.new("ShaderNodeCombineXYZ")
    toward.name = "glint_sun"
    world = bpy.context.scene.world
    for axis, value, expression in zip(
        "XYZ", sun, ("cos(e) * sin(b)", "cos(e) * cos(b)", "sin(e)"), strict=True
    ):
        toward.inputs[axis].default_value = value
        if SUN[0] in world:
            drive(
                toward.inputs[axis],
                "default_value",
                expression,
                world,
                e=f'["{SUN[0]}"]',
                b=f'["{SUN[1]}"]',
            )
    rising = tree.nodes.new("ShaderNodeSeparateXYZ")
    tree.links.new(toward.outputs["Vector"], rising.inputs["Vector"])
    risen = _math(tree, "GREATER_THAN", rising.outputs["Z"], 0.0, name="glint_risen")
    # Incoming points at the camera: the mirroring facet faces the half-way vector.
    mirror = _vector(
        tree,
        "NORMALIZE",
        _vector(tree, "ADD", geometry.outputs["Incoming"], toward.outputs["Vector"]),
    )
    # 2 (1 - cos d) = d^2 - d^4 / 12.
    off_sq = _math(
        tree,
        "MULTIPLY",
        _math(tree, "SUBTRACT", 1.0, _vector(tree, "DOT_PRODUCT", normal, mirror)),
        2.0,
    )
    fade = tree.nodes.new("ShaderNodeMapRange")
    fade.name = "glint_reach"
    fade.interpolation_type = "SMOOTHSTEP"
    tree.links.new(
        _math(tree, "DIVIDE", off_sq, _math(tree, "MAXIMUM", variance, 1e-12)),
        fade.inputs["Value"],
    )
    fade.inputs["From Min"].default_value, fade.inputs["From Max"].default_value = (
        GLINT_REACH
    )
    fade.inputs["To Min"].default_value, fade.inputs["To Max"].default_value = 1.0, 0.0
    return _math(tree, "MULTIPLY", fade.outputs["Result"], risen)


def _glitter(
    tree: bpy.types.NodeTree,
    wind: tuple[Wave, ...],
    where: bpy.types.NodeSocket,
    pixel: _Pixel,
    unresolved: bpy.types.NodeSocket,
    glinting: bpy.types.NodeSocket,
) -> tuple[bpy.types.NodeSocket, bpy.types.NodeSocket]:
    """A Gaussian tilt per cell of one specular point at `where`, (east, north, time),
    re-drawn as it twinkles, and the slope variance the tilts carry out of the lobe.

    A pixel's s samples of its n cells count glints as n cells do if each cell's lobe
    catches the sun n / s times as often: variance r^2 (n / s - 1) about a sun of slope
    radius r, s being `cycles.samples`.

    ponytail: the cells keep the mean wind's size through a gust, as a size that
    varied over the sea would re-cut the grid as the gust moves; crossfade two fixed
    grids by the local wind if glints should thin in the lulls.
    """
    cell_m2 = specular_cell_m2(wind)
    link = tree.links.new
    cells = _math(
        tree,
        "DIVIDE",
        _math(tree, "MULTIPLY", pixel.along_m, pixel.across_m),
        cell_m2,
    )
    widen = _math(tree, "MULTIPLY_ADD", cells, 0.0, -1.0)
    drive(
        widen.node.inputs[1],
        "default_value",
        "1 / samples",
        bpy.context.scene,
        samples="cycles.samples",
    )
    lobe = _math(
        tree, "MULTIPLY", _math(tree, "MAXIMUM", widen, 0.0), SUN_SLOPE_RADIUS**2
    )
    carried = _math(
        tree,
        "MULTIPLY",
        _math(tree, "MAXIMUM", _math(tree, "SUBTRACT", unresolved, lobe), 0.0),
        glinting,
        name="glitter_variance",
    )
    split = tree.nodes.new("ShaderNodeSeparateXYZ")
    link(where, split.inputs["Vector"])
    time_s = split.outputs["Z"]
    flat = tree.nodes.new("ShaderNodeCombineXYZ")
    link(split.outputs["X"], flat.inputs["X"])
    link(split.outputs["Y"], flat.inputs["Y"])
    scaled = _vector(tree, "SCALE", flat.outputs["Vector"], name="glitter_cells")
    scaled.node.inputs["Scale"].default_value = 1 / math.sqrt(cell_m2)
    cell = _vector(tree, "FLOOR", scaled)
    # Each cell twinkles at its own phase, or the whole sea would flip at once.
    offset = tree.nodes.new("ShaderNodeTexWhiteNoise")
    offset.noise_dimensions = "3D"
    link(cell, offset.inputs["Vector"])
    draw = _math(
        tree,
        "FLOOR",
        _math(tree, "MULTIPLY_ADD", time_s, twinkle_hz(wind), offset.outputs["Value"]),
        name="glitter_draw",
    )
    noise = tree.nodes.new("ShaderNodeTexWhiteNoise")
    noise.noise_dimensions = "4D"
    link(cell, noise.inputs["Vector"])
    link(draw, noise.inputs["W"])
    uniform = tree.nodes.new("ShaderNodeSeparateXYZ")
    link(noise.outputs["Color"], uniform.inputs["Vector"])
    # Box-Muller: a radius from one uniform, an angle from the other.
    log_u = _math(
        tree, "LOGARITHM", _math(tree, "MAXIMUM", uniform.outputs["X"], 1e-7), math.e
    )
    sigma = _math(tree, "SQRT", _math(tree, "MULTIPLY", carried, 0.5))
    radius = _math(
        tree,
        "MULTIPLY",
        _math(tree, "SQRT", _math(tree, "MULTIPLY", log_u, -2.0)),
        sigma,
    )
    angle = _math(tree, "MULTIPLY", uniform.outputs["Y"], 2 * math.pi)
    east = tree.nodes.new("ShaderNodeCombineXYZ")
    link(radius, east.inputs["X"])
    tilt = tree.nodes.new("ShaderNodeVectorRotate")
    tilt.rotation_type = "Z_AXIS"
    tilt.name = "glitter_tilt"
    link(east.outputs["Vector"], tilt.inputs["Vector"])
    link(angle, tilt.inputs["Angle"])
    return tilt.outputs["Vector"], carried


def _lobe(
    tree: bpy.types.NodeTree,
    along: bpy.types.NodeSocket,
    across: bpy.types.NodeSocket,
) -> tuple[bpy.types.NodeSocket, bpy.types.NodeSocket]:
    """Beckmann roughness and aspect, alpha across / alpha along, for the unresolved
    slope variance at the two footprints. Per axis alpha = sqrt(2) sigma_axis =
    sqrt(variance); Blender's roughness is their geometric mean's square root."""
    along = _math(tree, "MAXIMUM", along, 1e-12)
    across = _math(tree, "MAXIMUM", across, 1e-12)
    product = _math(tree, "MULTIPLY", along, across)
    roughness = _math(
        tree, "MINIMUM", _math(tree, "POWER", product, 0.125), 1.0, name="sea_roughness"
    )
    return roughness, _math(tree, "SQRT", _math(tree, "DIVIDE", across, along))


def _incidence(
    tree: bpy.types.NodeTree, normal: bpy.types.NodeSocket
) -> bpy.types.NodeSocket:
    """|cos(theta)| between the wave normal and the viewing ray: against the wave
    normal, not the plane's, or a flat sea's emissivity lands on water visibly not
    flat."""
    geometry = tree.nodes.new("ShaderNodeNewGeometry")
    facing = _vector(tree, "DOT_PRODUCT", geometry.outputs["Incoming"], normal)
    return _math(tree, "ABSOLUTE", facing)


def _thermal(
    tree: bpy.types.NodeTree,
    sea: Sea,
    normal: bpy.types.NodeSocket,
    unresolved: tuple[bpy.types.NodeSocket, bpy.types.NodeSocket],
    slope_max: float,
    sky: SkyCurve,
) -> bpy.types.NodeSocket:
    """eps(theta) of the sea emitted, the remaining 1 - eps reflected from the sky.

    Complements, so the two very nearly cancel and the sea holds close to ambient at
    every angle. The reflection is a lookup in the wave normal's mirror direction, not
    a BSDF: Cycles sampling a lobe across a sky cold overhead leaves the sea grainy.
    Only the slope along the view enters: the slope across turns the reflection
    sideways, which lowers it only at second order. Derivation: a tilt d across the
    view scales the mirror's sin(elevation) by (1 - d^2) / (1 + d^2).

    ponytail: reflects the sky averaged round the horizon, no hulls or clouds at their
    bearing; trace a sharp mirror if those must show.

    ponytail: facet lean taken at the mirror elevation, not the view's off the wave
    normal, so the steep near sea reads warm; tabulate against both if it shows.
    """
    geometry = tree.nodes.new("ShaderNodeNewGeometry")
    # Incoming points at the camera; REFLECT takes the ray.
    ray = _vector(tree, "SCALE", geometry.outputs["Incoming"], -1.0)
    mirror = tree.nodes.new("ShaderNodeSeparateXYZ")
    tree.links.new(_vector(tree, "REFLECT", ray, normal), mirror.inputs["Vector"])
    along = _math(
        tree,
        "DIVIDE",
        _math(tree, "SQRT", unresolved[0]),
        slope_max,
        name="sea_slope_along",
    )
    sky_seen = lookup(
        tree,
        _reflection_image(sea.t_sea_k, sky, slope_max),
        mirror.outputs["Z"],
        along,
    )
    sky_seen.node.name = "sea_reflection"
    mean = _math(tree, "MULTIPLY", _math(tree, "ADD", *unresolved), 0.5)
    fraction = _math(tree, "DIVIDE", _math(tree, "SQRT", mean), slope_max)
    emissivity = lookup(
        tree,
        _emissivity_image(sea.t_sea_k, slope_max),
        _incidence(tree, normal),
        fraction,
    )
    emitted = _math(tree, "SUBTRACT", lwir.band_radiance(sea.t_sea_k), sky_seen)
    radiance = _math(
        tree, "MULTIPLY_ADD", emissivity, emitted, sky_seen, name="sea_radiance"
    )
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.inputs["Strength"].default_value = 1.0
    tree.links.new(radiance, emission.inputs["Color"])
    return emission.outputs["Emission"]


def _daylight(
    tree: bpy.types.NodeTree,
    sea: Sea,
    normal: bpy.types.NodeSocket,
    tangent: bpy.types.NodeSocket,
    unresolved: tuple[bpy.types.NodeSocket, bpy.types.NodeSocket],
    whitecaps: tuple[bpy.types.NodeSocket, bpy.types.NodeSocket, bpy.types.NodeSocket],
    wake: "_Wakes | None" = None,
) -> bpy.types.NodeSocket:
    """Water reflecting at seawater's IOR over its body's colour, white where its crests
    break and where a hull leaves foam.

    Beckmann, as Cox & Munk's slopes are near Gaussian; Principled offers only GGX. A
    conductor's Fresnel at zero extinction is the dielectric's.
    """
    roughness, aspect = _lobe(tree, *unresolved)
    link = tree.links.new
    mirror = tree.nodes.new("ShaderNodeBsdfMetallic")
    mirror.name = "sea_specular"
    mirror.distribution = "BECKMANN"
    mirror.fresnel_type = "PHYSICAL_CONDUCTOR"
    mirror.inputs["IOR"].default_value = (SEAWATER_IOR,) * 3
    mirror.inputs["Extinction"].default_value = (0.0, 0.0, 0.0)
    link(normal, mirror.inputs["Normal"])
    link(tangent, mirror.inputs["Tangent"])
    link(roughness, mirror.inputs["Roughness"])
    # Cycles' alpha_y / alpha_x = 1 - 0.9 a (kernel/svm/closure.h), so
    # a = (1 - aspect) / 0.9.
    link(
        _math(tree, "MULTIPLY", _math(tree, "SUBTRACT", 1.0, aspect), 1 / 0.9),
        mirror.inputs["Anisotropy"],
    )
    fresnel = tree.nodes.new("ShaderNodeFresnel")
    fresnel.inputs["IOR"].default_value = SEAWATER_IOR
    link(normal, fresnel.inputs["Normal"])
    through = _math(tree, "SUBTRACT", 1.0, fresnel.outputs["Fac"])
    # A whitecap's bubbles brighten the water under all its foam, as a wake's do.
    bubbles = _math(tree, "MULTIPLY", whitecaps[2], BUBBLE_GAIN)
    if wake is not None:
        bubbles = _math(tree, "MAXIMUM", bubbles, wake.bubbles)
    through = _math(tree, "MULTIPLY", through, _math(tree, "ADD", bubbles, 1.0))
    body = tree.nodes.new("ShaderNodeBsdfDiffuse")
    link(normal, body.inputs["Normal"])
    # Dimmed by Fresnel's 1 - F toward the viewer only.
    color = body_color(sea.chlorophyll_mg_m3, SEAWATER_IOR)
    link(
        _vector(tree, "SCALE", color, through, name="sea_body_color"),
        body.inputs["Color"],
    )
    water = tree.nodes.new("ShaderNodeAddShader")
    link(mirror.outputs["BSDF"], water.inputs[0])
    link(body.outputs["BSDF"], water.inputs[1])
    foam = tree.nodes.new("ShaderNodeBsdfDiffuse")
    link(normal, foam.inputs["Normal"])
    foam.inputs["Color"].default_value = (*(FRESH_FOAM_REFLECTANCE,) * 3, 1.0)
    mix = tree.nodes.new("ShaderNodeMixShader")
    cover = whitecaps[1]
    if wake is not None:
        # Wake and whitecap foam are one Diffuse, mixed at 1 - (1 - a)(1 - b): a second
        # Diffuse and Mix overflow Cycles' SVM stack. Mix clamps its factor to 1, the
        # product does not, so `cover` is clamped here.
        unfoamed = _math(
            tree,
            "MULTIPLY",
            _math(tree, "SUBTRACT", 1.0, _math(tree, "MINIMUM", cover, 1.0)),
            _math(tree, "SUBTRACT", 1.0, wake.foam),
        )
        cover = _math(tree, "SUBTRACT", 1.0, unfoamed)
    link(cover, mix.inputs["Factor"])
    # Mix Shader names both shader inputs "Shader", so they can only be indexed.
    link(water.outputs["Shader"], mix.inputs[1])
    link(foam.outputs["BSDF"], mix.inputs[2])
    return mix.outputs["Shader"]


@cache
def _network_widths() -> tuple[np.ndarray, np.ndarray]:
    """(share, width): the line width, in cells, along the foam network's edges that
    covers each share of the sea."""
    raw = np.loadtxt(_NETWORK_CSV, delimiter=",", comments="#")
    return raw[:, 0], raw[:, 1]


def _foam_age_group() -> bpy.types.NodeTree:
    """One foam age, folded into the older ages' `Area`, `Foam` and `Under` by
    maximum. Old foam tears into whole network segments rather than greying (Yingst,
    Alford & Parberry 2011)."""
    name = "whitecap_age"
    if name in bpy.data.node_groups:
        return bpy.data.node_groups[name]
    group, given, out = _group(
        name,
        tuple(
            (n, "Float")
            for n in (
                "Acceleration",
                "Threshold",
                "Sigma",
                "Thin",
                "Share",
                "Wide",
                "Footprint",
                "Lace",
                "Level",
                "Network",
                "Area",
                "Foam",
                "Under",
            )
        ),
        (("Area", "Float"), ("Foam", "Float"), ("Under", "Float")),
    )
    sigmas = CDF_SIGMAS * (2 * (np.arange(CURVE_SAMPLES) + 0.5) / CURVE_SAMPLES - 1)
    cdf = curve_image("normal_cdf", np.vectorize(NormalDist().cdf)(sigmas))
    levels = (np.arange(CURVE_SAMPLES) + 0.5) / CURVE_SAMPLES
    ppf = curve_image("normal_ppf", np.vectorize(NormalDist().inv_cdf)(1 - levels))

    def uniform(z: bpy.types.NodeSocket) -> bpy.types.NodeSocket:
        # Clamped, so an infinite threshold reads the table's end.
        x = _math(group, "MULTIPLY_ADD", z, 1 / (2 * CDF_SIGMAS), 0.5)
        x.node.use_clamp = True
        return lookup(group, cdf, x)

    excess = _math(group, "SUBTRACT", given["Acceleration"], given["Threshold"])
    fill = _math(
        group,
        "MULTIPLY",
        uniform(_math(group, "DIVIDE", excess, given["Sigma"])),
        given["Thin"],
    )
    laced = _math(
        group,
        "MULTIPLY_ADD",
        _math(group, "SUBTRACT", given["Lace"], lookup(group, ppf, fill)),
        2.0,
        0.5,
    )
    laced.node.use_clamp = True
    laced = _math(group, "MULTIPLY", laced, _math(group, "GREATER_THAN", fill, 1e-3))

    shares, widths = _network_widths()
    top = float(widths[-1])
    grid = (np.arange(CURVE_SAMPLES) + 0.5) / CURVE_SAMPLES * top
    cover = curve_image("foam_network_cover", np.interp(grid, widths, shares))
    width = _math(
        group,
        "MINIMUM",
        _math(
            group,
            "MAXIMUM",
            _math(group, "DIVIDE", given["Footprint"], FOAM_CELL_M),
            given["Wide"],
        ),
        top,
    )
    line = _math(
        group,
        "MULTIPLY_ADD",
        _math(group, "SUBTRACT", width, given["Network"]),
        1 / FOAM_LINE_SOFT,
        0.5,
    )
    line.node.use_clamp = True
    covered = lookup(group, cover, _math(group, "DIVIDE", width, top))
    kept = _math(
        group,
        "MINIMUM",
        _math(group, "DIVIDE", given["Share"], _math(group, "MAXIMUM", covered, 1e-4)),
        1.0,
    )
    stays = _math(
        group,
        "MULTIPLY_ADD",
        _math(group, "SUBTRACT", kept, given["Level"]),
        FOAM_POP_GAIN,
        0.5,
    )
    stays.node.use_clamp = True
    torn = _math(group, "MULTIPLY", laced, _math(group, "MULTIPLY", line, stays))
    for socket, value in (
        ("Area", laced),
        ("Foam", torn),
        ("Under", fill),
    ):
        group.links.new(_math(group, "MAXIMUM", given[socket], value), out[socket])
    return group


def _foam_lace(
    tree: bpy.types.NodeTree,
    sea: Sea,
    where: bpy.types.NodeSocket,
    outputs: Outputs,
    tile: bpy.types.Image,
) -> tuple[bpy.types.NodeSocket, bpy.types.NodeSocket, bpy.types.NodeSocket]:
    """(lace, fine lace, distance in cells to the foam network's nearest edge),
    drifting with the surface, read at `where`, (east, north, time). The lace is two
    independent unit Gaussians rescaled to one.

    ponytail: the network drifts on, unlooped, so a looped clip's seam moves it by the
    drift over the span; crossfade two networks if the pop shows."""
    east, north = _downwind(sea)
    drift = SLICK_DRIFT * sea.wind_speed_mps
    layers = []
    for scale, start in ((1.0, 0.0), (LACE_OCTAVE, 0.5), (LACE_OCTAVE, 0.25)):
        per_m = scale / FOAM_TILE_M
        along = per_m / FOAM_STREAK
        layers.append(
            _drifting(
                tree,
                where,
                outputs,
                tile,
                ((along * east, along * north), (per_m * north, -per_m * east)),
                (drift * east, drift * north),
                start,
                "whitecap_lace",
            )
        )
    weight = LACE_FINE_WEIGHT
    summed = _math(tree, "MULTIPLY_ADD", layers[1], weight, layers[0])
    # Carried with the surface, the network read where it started.
    split = tree.nodes.new("ShaderNodeSeparateXYZ")
    tree.links.new(where, split.inputs["Vector"])
    start_at = tree.nodes.new("ShaderNodeCombineXYZ")
    for axis, v in (("X", drift * east), ("Y", drift * north)):
        tree.links.new(
            _math(tree, "MULTIPLY_ADD", split.outputs["Z"], -v, split.outputs[axis]),
            start_at.inputs[axis],
        )
    # Blender's Noise colour is three noises in [0, 1]: centred, `FOAM_WARP` cells.
    bend = tree.nodes.new("ShaderNodeTexNoise")
    bend.noise_dimensions = "2D"
    bend.inputs["Scale"].default_value = 1 / FOAM_CELL_M
    bend.inputs["Detail"].default_value = 3.0
    tree.links.new(start_at.outputs["Vector"], bend.inputs["Vector"])
    reach = FOAM_WARP * FOAM_CELL_M
    bent = _vector(
        tree,
        "ADD",
        start_at.outputs["Vector"],
        _vector(
            tree, "MULTIPLY_ADD", bend.outputs["Color"], (2 * reach,) * 3, (-reach,) * 3
        ),
    )
    network = tree.nodes.new("ShaderNodeTexVoronoi")
    network.name = "foam_network"
    network.voronoi_dimensions = "2D"
    network.feature = "DISTANCE_TO_EDGE"
    network.inputs["Scale"].default_value = 1 / FOAM_CELL_M
    tree.links.new(bent, network.inputs["Vector"])
    return (
        _math(tree, "MULTIPLY", summed, 1 / math.sqrt(1 + weight**2)),
        layers[2],
        network.outputs["Distance"],
    )


def _drifting(
    tree: bpy.types.NodeTree,
    where: bpy.types.NodeSocket,
    outputs: Outputs,
    tile: bpy.types.Image,
    axes: tuple[tuple[float, float], tuple[float, float]],
    velocity_mps: tuple[float, float],
    start: float,
    name: str,
) -> bpy.types.NodeSocket:
    """`tile` laid on the sea along `axes`, each an (east, north) tile axis in tiles
    per metre, offset `start` tiles, frozen and carried at `velocity_mps`, (east,
    north); read at `where`, (east, north, time).

    A loop crossfades two layers, each carried for one span and back while its weight
    is 0 (Vlachos, "Water flow in Portal 2", SIGGRAPH 2010 course), over the root of
    their squared weights so the variance holds.
    """
    split = tree.nodes.new("ShaderNodeSeparateXYZ")
    tree.links.new(where, split.inputs["Vector"])
    time_s = split.outputs["Z"]
    flat = tree.nodes.new("ShaderNodeCombineXYZ")
    tree.links.new(split.outputs["X"], flat.inputs["X"])
    tree.links.new(split.outputs["Y"], flat.inputs["Y"])
    here = tree.nodes.new("ShaderNodeCombineXYZ")
    here.name = f"{name}_axes"
    for axis, out in zip(axes, ("X", "Y"), strict=True):
        along = _vector(tree, "DOT_PRODUCT", flat.outputs["Vector"])
        along.node.inputs["Vector_001"].default_value = (*axis, 0.0)
        tree.links.new(along, here.inputs[out])
    # The tile read at x - v t is the tile at x carried v t.
    carry = [-(velocity_mps[0] * e + velocity_mps[1] * n) for e, n in axes]

    def layer(carried_s: bpy.types.NodeSocket, shift: float) -> bpy.types.NodeSocket:
        at = _vector(tree, "MULTIPLY_ADD", name=f"{name}_carry")
        at.node.inputs["Vector"].default_value = (*carry, 0.0)
        tree.links.new(carried_s, at.node.inputs["Vector_001"])
        at.node.inputs["Vector_002"].default_value = (shift, shift, 0.0)
        texture = tree.nodes.new("ShaderNodeTexImage")
        texture.image = tile
        texture.extension = "REPEAT"
        tree.links.new(
            _vector(tree, "ADD", here.outputs["Vector"], at), texture.inputs["Vector"]
        )
        return texture.outputs["Color"]

    if not outputs.loop:
        return layer(time_s, start)
    span_s = outputs.span_s
    weighted = []
    # Half a tile apart, so the two layers are independent.
    for i, lag in enumerate((0.0, 0.5)):
        cycle = _math(
            tree,
            "FRACT",
            _math(tree, "MULTIPLY_ADD", time_s, 1 / span_s, lag),
        )
        weight = _math(
            tree,
            "SUBTRACT",
            1.0,
            _math(tree, "ABSOLUTE", _math(tree, "MULTIPLY_ADD", cycle, 2.0, -1.0)),
            name=f"{name}_weight_{i}",
        )
        carried_s = _math(tree, "MULTIPLY", cycle, span_s, name=f"{name}_carried_{i}")
        weighted.append((weight, layer(carried_s, start + lag)))
    (w0, n0), (w1, n1) = weighted
    norm = _math(
        tree,
        "SQRT",
        _math(tree, "MULTIPLY_ADD", w0, w0, _math(tree, "MULTIPLY", w1, w1)),
    )
    return _math(
        tree,
        "DIVIDE",
        _math(tree, "MULTIPLY_ADD", w0, n0, _math(tree, "MULTIPLY", w1, n1)),
        norm,
    )


def _downwind(sea: Sea) -> tuple[float, float]:
    # Wind is named for where it blows from.
    toward = math.radians(sea.wind_from_deg + 180.0)
    return math.sin(toward), math.cos(toward)


def _gust(
    tree: bpy.types.NodeTree,
    sea: Sea,
    where: bpy.types.NodeSocket,
    outputs: Outputs,
    tile: bpy.types.Image,
) -> bpy.types.NodeSocket:
    """The wind over its mean, less one, carried downwind at the mean wind; read at
    `where`, (east, north, time)."""
    per_m = 1 / GUST_TILE_M
    east, north = _downwind(sea)
    speed = sea.wind_speed_mps
    unit = _drifting(
        tree,
        where,
        outputs,
        tile,
        ((per_m, 0.0), (0.0, per_m)),
        (speed * east, speed * north),
        0.0,
        "gust",
    )
    return _math(tree, "MULTIPLY", unit, turbulence_intensity(speed), name="gust")


def _slick(
    tree: bpy.types.NodeTree,
    sea: Sea,
    where: bpy.types.NodeSocket,
    outputs: Outputs,
    rng: np.random.Generator,
) -> bpy.types.NodeSocket:
    """1 under a slick: the top `slick_cover` of a tile at the windrow spacing,
    stretched along the wind and drifting with it; read at `where`, (east, north,
    time).

    ponytail: the integral scale stands for the windrow spacing, a judgement, and
    there are rows at every wind, where below the cutoff real slicks lie in patches.
    Give the slicks their own spectrum if the rows read wrong. Whitecaps lose the
    damped waves but keep the clean sea's unresolved acceleration; mix in the
    survivors' if slicks break too often.
    """
    east, north = _downwind(sea)
    speed = sea.wind_speed_mps
    spacing_m = WINDROW_SPACING_S * max(speed, WINDROW_MIN_WIND_MPS)
    tile = von_karman_field(rng, GUST_CELLS, GUST_SPACING_M, spacing_m)
    across = 1 / GUST_TILE_M
    along = across / WINDROW_ASPECT
    unit = _drifting(
        tree,
        where,
        outputs,
        curve_image("sea_slick", tile),
        ((north * across, -east * across), (east * along, north * along)),
        (SLICK_DRIFT * speed * east, SLICK_DRIFT * speed * north),
        0.0,
        "slick",
    )
    # The tile is Gaussian, so its top `slick_cover` lies above this.
    threshold = NormalDist().inv_cdf(1.0 - sea.slick_cover)
    return _math(tree, "GREATER_THAN", unit, threshold, name="slick")


def _gusty(
    tree: bpy.types.NodeTree,
    sea: Sea,
    gust: bpy.types.NodeSocket,
    unresolved: tuple[bpy.types.NodeSocket, bpy.types.NodeSocket],
) -> tuple[bpy.types.NodeSocket, bpy.types.NodeSocket]:
    """`unresolved` with Cox & Munk's variance at the local wind for the mean's.

    ponytail: drawn waves and swell hold the mean wind, as waves long enough to draw
    grow too slowly to follow a gust (Plant 1982); a close camera that draws its
    capillaries sees no gust in them. Scale the drawn slopes below the growth cutoff if
    that shows. The tile is not filtered to the footprint either: past the pixel the
    samples average it, until too few fall in one; filter it to the footprint if
    distant gusts shimmer.
    """
    offset = _math(
        tree,
        "MULTIPLY",
        gust,
        gust_slope_variance(sea.wind_speed_mps),
        name="sea_gust_variance",
    )
    # A lull can take more than the pixel leaves.
    return tuple(
        _math(tree, "MAXIMUM", _math(tree, "ADD", variance, offset), 0.0)
        for variance in unresolved
    )


def _breaking(
    tree: bpy.types.NodeTree,
    sea: Sea,
    wind: tuple[Wave, ...],
    gust: bpy.types.NodeSocket | None,
    gust_max: float,
) -> float | bpy.types.NodeSocket:
    """The breaking threshold of Monahan's whitecap cover at the local wind.

    ponytail: the cover answers the gust at once; follow it with a lag if the patches
    track it too tightly.
    """
    speed = sea.wind_speed_mps
    if gust is None:
        return float(whitecap_thresholds_g(wind, [whitecap_fraction(speed)])[0])
    texel = (np.arange(CURVE_SAMPLES) + 0.5) / CURVE_SAMPLES
    thresholds = whitecap_thresholds_g(
        wind,
        [gusty_whitecap_fraction(speed, g) for g in gust_max * (2 * texel - 1)],
    )
    # Finite, so the linear lookup never mixes an infinity into NaN.
    big = float(np.finfo(np.float32).max)
    table = curve_image(
        "sea_breaking", np.nan_to_num(thresholds, posinf=big, neginf=-big)
    )
    at = _math(tree, "MULTIPLY_ADD", gust, 0.5 / gust_max, 0.5)
    threshold = lookup(tree, table, at)
    threshold.node.name = "sea_breaking"
    return threshold


class _Wakes(NamedTuple):
    band: bpy.types.NodeSocket  # 1 in the turbulent wake
    foam: bpy.types.NodeSocket  # fresh foam's cover
    bubbles: bpy.types.NodeSocket  # water-body reflectance added, a fraction of its own
    tilt: bpy.types.NodeSocket | None  # the Kelvin arms' -grad(height)


class _Path(NamedTuple):
    """Where a shading point lies against a hull's path."""

    along: bpy.types.NodeSocket  # behind the hull
    across: bpy.types.NodeSocket  # to its starboard
    d_along: bpy.types.NodeSocket  # gradients, (east, north)
    d_across: bpy.types.NodeSocket
    fixed: bpy.types.NodeSocket  # along and across, fixed in the water


def _pair(
    tree: bpy.types.NodeTree,
    east: bpy.types.NodeSocket | float,
    north: bpy.types.NodeSocket | float,
) -> bpy.types.NodeSocket:
    node = tree.nodes.new("ShaderNodeCombineXYZ")
    for socket, value in zip(("X", "Y"), (east, north), strict=True):
        if isinstance(value, bpy.types.NodeSocket):
            tree.links.new(value, node.inputs[socket])
        else:
            node.inputs[socket].default_value = value
    return node.outputs["Vector"]


def _orbit_path(
    tree: bpy.types.NodeTree,
    wake: Wake,
    time_s: bpy.types.NodeSocket,
    east: bpy.types.NodeSocket,
    north: bpy.types.NodeSocket,
) -> _Path:
    assert wake.orbit_m is not None
    bearing = _math(tree, "ARCTAN2", east, north)
    hull = _math(
        tree, "MULTIPLY_ADD", time_s, 2 * math.pi / wake.lap_s, wake.start_bearing_rad
    )
    # Clockwise: behind a hull is anticlockwise of it, as far as the hull after it.
    angle = _math(
        tree,
        "FLOORED_MODULO",
        _math(tree, "SUBTRACT", hull, bearing),
        2 * math.pi / wake.count,
    )
    radius = _vector(tree, "LENGTH", _pair(tree, east, north))
    cos_b, sin_b = _math(tree, "COSINE", bearing), _math(tree, "SINE", bearing)
    # Round the circle in whole foam tiles, so the bearing's wrap is no seam.
    round_m = FOAM_STREAK * FOAM_TILE_M
    round_m *= max(1, round(2 * math.pi * wake.orbit_m / round_m))
    return _Path(
        along=_math(tree, "MULTIPLY", angle, wake.orbit_m),
        # Clockwise, starboard is in towards the centre.
        across=_math(tree, "SUBTRACT", wake.orbit_m, radius),
        d_along=_pair(tree, _math(tree, "MULTIPLY", cos_b, -1.0), sin_b),
        d_across=_pair(
            tree,
            _math(tree, "MULTIPLY", sin_b, -1.0),
            _math(tree, "MULTIPLY", cos_b, -1.0),
        ),
        fixed=_pair(
            tree, _math(tree, "MULTIPLY", bearing, -round_m / (2 * math.pi)), radius
        ),
    )


def _line_path(
    tree: bpy.types.NodeTree,
    wake: Wake,
    time_s: bpy.types.NodeSocket,
    east: bpy.types.NodeSocket,
    north: bpy.types.NodeSocket,
) -> _Path:
    sin_h, cos_h = math.sin(wake.heading_rad), math.cos(wake.heading_rad)
    e0, n0 = wake.start_m
    hull_e = _math(tree, "MULTIPLY_ADD", time_s, wake.speed_mps * sin_h, e0)
    hull_n = _math(tree, "MULTIPLY_ADD", time_s, wake.speed_mps * cos_h, n0)
    de = _math(tree, "SUBTRACT", east, hull_e)
    dn = _math(tree, "SUBTRACT", north, hull_n)

    def ahead(e: bpy.types.NodeSocket, n: bpy.types.NodeSocket) -> bpy.types.NodeSocket:
        return _math(tree, "MULTIPLY_ADD", e, sin_h, _math(tree, "MULTIPLY", n, cos_h))

    def starboard(
        e: bpy.types.NodeSocket, n: bpy.types.NodeSocket
    ) -> bpy.types.NodeSocket:
        # Starboard of the heading is (cos, -sin).
        return _math(tree, "MULTIPLY_ADD", e, cos_h, _math(tree, "MULTIPLY", n, -sin_h))

    return _Path(
        along=_math(tree, "MULTIPLY", ahead(de, dn), -1.0),
        across=starboard(de, dn),
        d_along=_pair(tree, -sin_h, -cos_h),
        d_across=_pair(tree, cos_h, -sin_h),
        fixed=_pair(tree, ahead(east, north), starboard(east, north)),
    )


def _path(tree: bpy.types.NodeTree, wake: Wake, where: bpy.types.NodeSocket) -> _Path:
    position = tree.nodes.new("ShaderNodeSeparateXYZ")
    position.name = "wake_position"
    tree.links.new(where, position.inputs["Vector"])
    east, north, time_s = (position.outputs[c] for c in ("X", "Y", "Z"))
    if wake.orbit_m is not None:
        return _orbit_path(tree, wake, time_s, east, north)
    return _line_path(tree, wake, time_s, east, north)


def _gaussian(
    tree: bpy.types.NodeTree,
    x: bpy.types.NodeSocket,
    half: bpy.types.NodeSocket | float,
) -> bpy.types.NodeSocket:
    """exp(-(x / half)^2)."""
    ratio = _math(tree, "DIVIDE", x, half)
    return _math(
        tree,
        "EXPONENT",
        _math(tree, "MULTIPLY", _math(tree, "MULTIPLY", ratio, ratio), -1.0),
    )


def _wakes(
    tree: bpy.types.NodeTree,
    wakes: tuple[Wake, ...],
    where: bpy.types.NodeSocket,
    footprint_m: bpy.types.NodeSocket,
    noise: bpy.types.Image,
    sea_slope_variance: float,
) -> _Wakes:
    """Every hull's turbulent band, stern foam and Kelvin arms: the strongest band and
    foam, and the arms summed, read at `where`, (east, north, time).

    ponytail: the arms are Kelvin's straight-path pattern laid along the path, which a
    tight turn bends; sum the arms from past poses if a turn reads wrong.
    """
    levels = (np.arange(CURVE_SAMPLES) + 0.5) / CURVE_SAMPLES
    ppf = curve_image("normal_ppf", np.vectorize(NormalDist().inv_cdf)(1 - levels))
    band: bpy.types.NodeSocket | float = 0.0
    foam: bpy.types.NodeSocket | float = 0.0
    bubbles: bpy.types.NodeSocket | float = 0.0
    tilt = None
    for wake in wakes:
        path = _path(tree, wake, where)
        behind = _math(tree, "GREATER_THAN", path.along, 0.0)
        # Ahead of the hull nothing has aged: unclamped, the exponentials overflow
        # there, and the power below takes a negative base.
        astern = _math(tree, "MAXIMUM", path.along, 0.0)
        side = _math(tree, "ABSOLUTE", path.across)
        # A judgement: a beam wide at the stern, widening as distance^(1/5).
        width = _math(
            tree,
            "MULTIPLY",
            _math(
                tree,
                "POWER",
                _math(tree, "MULTIPLY_ADD", astern, 1 / wake.length_m, 1.0),
                0.2,
            ),
            wake.beam_m,
        )
        core = _math(
            tree,
            "MULTIPLY",
            behind,
            _gaussian(tree, side, _math(tree, "MULTIPLY", width, 0.5)),
        )
        band = _math(tree, "MAXIMUM", band, core)
        age = _math(tree, "DIVIDE", astern, wake.speed_mps)
        fresh = _math(tree, "EXPONENT", _math(tree, "MULTIPLY", age, -1 / FOAM_EFOLD_S))
        stern = _math(tree, "MULTIPLY", behind, _gaussian(tree, side, wake.beam_m / 2))
        cover = _math(tree, "MULTIPLY", stern, fresh)
        foam = _math(
            tree, "MAXIMUM", foam, _patchy(tree, cover, path.fixed, noise, ppf)
        )
        rising = _math(
            tree, "EXPONENT", _math(tree, "MULTIPLY", age, -1 / BUBBLE_EFOLD_S)
        )
        bubbles = _math(
            tree,
            "MAXIMUM",
            bubbles,
            _math(tree, "MULTIPLY", core, _math(tree, "MULTIPLY", rising, BUBBLE_GAIN)),
        )
        if wake.steepest_arm**2 / 2 < SEEN_SLOPE_VARIANCE * sea_slope_variance:
            continue
        lasting: bpy.types.NodeSocket = behind
        if wake.orbit_m is not None:
            # A judgement: faded over the radius, past which the straight-path pattern
            # no longer follows the turn.
            fade = _math(
                tree, "EXPONENT", _math(tree, "MULTIPLY", path.along, -1 / wake.orbit_m)
            )
            lasting = _math(tree, "MULTIPLY", behind, fade)
        # Above the narrowing only the divergent wave, round its peak (Darmon et al.).
        roots = (1.0, -1.0) if wake.froude <= NARROWING_FROUDE else (1.0,)
        for root in roots:
            arm = _vector(
                tree, "SCALE", _kelvin(tree, wake, root, path, footprint_m), lasting
            )
            tilt = arm if tilt is None else _vector(tree, "ADD", tilt, arm)
    return _Wakes(band, foam, bubbles, tilt)


def _kelvin(
    tree: bpy.types.NodeTree,
    wake: Wake,
    root: float,
    path: _Path,
    footprint_m: bpy.types.NodeSocket,
) -> bpy.types.NodeSocket:
    """One of Kelvin's two waves, divergent at `root` 1, transverse at -1, as
    -grad(height), (east, north).

    Stationary phase: at tan(psi) = side / along, the wave's direction theta solves
    tan(psi) (1 + 2 t^2) = t, t = tan(theta), and its phase is
    g / U^2 (along cos(theta) + side sin(theta)) / cos(theta)^2 (DLMF 36.13).
    """
    side = _math(tree, "ABSOLUTE", path.across)
    edge = math.tan(KELVIN_HALF_ANGLE_RAD)
    tan_psi = _math(tree, "DIVIDE", side, _math(tree, "MAXIMUM", path.along, 1e-3))
    inside = _math(tree, "LESS_THAN", tan_psi, edge)
    tan_psi = _math(tree, "MAXIMUM", _math(tree, "MINIMUM", tan_psi, edge), 1e-4)
    disc = _math(
        tree,
        "SQRT",
        _math(
            tree,
            "MAXIMUM",
            _math(
                tree,
                "MULTIPLY_ADD",
                _math(tree, "MULTIPLY", tan_psi, tan_psi),
                -8.0,
                1.0,
            ),
            0.0,
        ),
    )
    t = _math(
        tree,
        "DIVIDE",
        _math(tree, "MULTIPLY_ADD", disc, root, 1.0),
        _math(tree, "MULTIPLY", tan_psi, 4.0),
    )
    cos_sq = _math(tree, "DIVIDE", 1.0, _math(tree, "MULTIPLY_ADD", t, t, 1.0))
    cos_t = _math(tree, "SQRT", cos_sq)
    sin_t = _math(tree, "MULTIPLY", t, cos_t)
    k = _math(tree, "DIVIDE", GRAVITY_MS2 / wake.speed_mps**2, cos_sq)
    phase = _math(
        tree,
        "MULTIPLY",
        k,
        _math(
            tree,
            "MULTIPLY_ADD",
            path.along,
            cos_t,
            _math(tree, "MULTIPLY", side, sin_t),
        ),
    )
    envelope: bpy.types.NodeSocket | float = 1.0
    if wake.froude > NARROWING_FROUDE:
        peak = wake.peak_angle_rad
        # A judgement: the peak half as wide as its angle.
        off = _math(
            tree,
            "SUBTRACT",
            _math(tree, "ARCTANGENT", tan_psi),
            peak,
        )
        envelope = _gaussian(tree, off, peak / 2)
    # Kriebel & Seelig's height, half of it the amplitude.
    amplitude = _math(
        tree,
        "MULTIPLY",
        _math(
            tree,
            "POWER",
            _math(tree, "MAXIMUM", _math(tree, "DIVIDE", side, wake.length_m), 1e-2),
            -1 / 3,
        ),
        wake.height_scale_m / 2,
    )
    steepest = _math(
        tree,
        "MINIMUM",
        _math(tree, "MULTIPLY", amplitude, k),
        math.pi * MAX_STEEPNESS,
    )
    resolved = tree.nodes.new("ShaderNodeMapRange")
    resolved.interpolation_type = "SMOOTHSTEP"
    resolved.inputs["From Min"].default_value = FADE_FOOTPRINTS[0]
    resolved.inputs["From Max"].default_value = FADE_FOOTPRINTS[1]
    tree.links.new(
        _math(
            tree,
            "DIVIDE",
            _math(tree, "DIVIDE", 2 * math.pi, k),
            _math(tree, "MAXIMUM", footprint_m, 1e-3),
        ),
        resolved.inputs["Value"],
    )
    slope = steepest
    for factor in (
        _math(tree, "SINE", phase),
        envelope,
        inside,
        resolved.outputs["Result"],
    ):
        slope = _math(tree, "MULTIPLY", slope, factor)
    sign = _math(tree, "SIGN", path.across)
    direction = _vector(
        tree,
        "ADD",
        _vector(tree, "SCALE", path.d_along, cos_t),
        _vector(tree, "SCALE", path.d_across, _math(tree, "MULTIPLY", sin_t, sign)),
    )
    return _vector(tree, "SCALE", direction, slope)


def _patchy(
    tree: bpy.types.NodeTree,
    cover: bpy.types.NodeSocket,
    fixed: bpy.types.NodeSocket,
    noise: bpy.types.Image,
    ppf: bpy.types.Image,
) -> bpy.types.NodeSocket:
    """`cover` as white where the Gaussian `noise` tops its own `cover` fraction,
    `ppf` its quantiles: patches whose mean is the cover, laid along the path at
    `fixed`."""
    stretch = tree.nodes.new("ShaderNodeVectorMath")
    stretch.operation = "MULTIPLY"
    tree.links.new(fixed, stretch.inputs[0])
    stretch.inputs[1].default_value = (
        1 / (FOAM_STREAK * FOAM_TILE_M),
        1 / FOAM_TILE_M,
        0.0,
    )
    texture = tree.nodes.new("ShaderNodeTexImage")
    texture.image = noise
    texture.extension = "REPEAT"
    tree.links.new(stretch.outputs["Vector"], texture.inputs["Vector"])
    threshold = lookup(tree, ppf, cover)
    # A judgement: edges half the tile's sigma wide.
    edge = _math(
        tree,
        "MULTIPLY_ADD",
        _math(tree, "SUBTRACT", texture.outputs["Color"], threshold),
        2.0,
        0.5,
    )
    edge.node.use_clamp = True
    return _math(tree, "MULTIPLY", edge, _math(tree, "GREATER_THAN", cover, 1e-3))


def _smoothed(
    tree: bpy.types.NodeTree,
    slick: bpy.types.NodeSocket | None,
    wake: _Wakes | None,
) -> bpy.types.NodeSocket | None:
    """Where the unresolved roughness is a slick's: under a slick, and in a turbulent
    wake, whose short waves stay damped (Milgram et al. 1993)."""
    if wake is None:
        return slick
    if slick is None:
        return wake.band
    return _math(tree, "MAXIMUM", slick, wake.band)


def _unresolved(
    tree: bpy.types.NodeTree,
    sea: Sea,
    wind: tuple[Wave, ...],
    swell: tuple[Wave, ...],
    survivors: tuple[Wave, ...],
    pixel: _Pixel,
    gust: bpy.types.NodeSocket | None,
    smooth: bpy.types.NodeSocket | None,
) -> tuple[bpy.types.NodeSocket, bpy.types.NodeSocket]:
    """The slope variance each pixel does not draw, along and across its view: the
    gusty sea's, and a slick's where `smooth` is 1. Each is twice its axis's, the total
    an isotropic sea of that slope would hold.

    Cox & Munk's axes are the wind's. ponytail: the view's take their variances by cos^2
    of the angle between and leave out the covariance, so a lobe 45 degrees off the
    wind stays aligned with the view. Rotate the tangent if a glitter shows it.
    """
    speed = sea.wind_speed_mps
    # From or toward, the axis is the same.
    wind_axis = math.radians(sea.wind_from_deg)
    facing = _vector(
        tree,
        "DOT_PRODUCT",
        pixel.along_dir,
        (math.sin(wind_axis), math.cos(wind_axis), 0),
    )
    cos2 = _math(tree, "MULTIPLY", facing, facing, name="sea_wind_facing")
    footprints = (pixel.along_m, pixel.across_m)

    def tabled(
        name: str,
        field: tuple[Wave, ...],
        fit: Callable[[float], tuple[float, float]],
    ) -> tuple[bpy.types.NodeSocket, bpy.types.NodeSocket]:
        axes = tuple(
            _footprint_table(
                f"{name}_{axis}",
                lambda f, i=i: unresolved_slope_variances(
                    speed, wind_axis, field, swell, f, fit
                )[i],
            )
            for i, axis in enumerate(("upwind", "crosswind"))
        )
        out = []
        for f, on_up in zip(
            footprints, (cos2, _math(tree, "SUBTRACT", 1.0, cos2)), strict=True
        ):
            up, cross = (
                _at_footprint(tree, t, f, f"{name}_{axis}")
                for t, axis in zip(axes, ("upwind", "crosswind"), strict=True)
            )
            mixed = _math(
                tree, "MULTIPLY_ADD", _math(tree, "SUBTRACT", up, cross), on_up, cross
            )
            out.append(_math(tree, "MULTIPLY", mixed, 2.0))
        return out[0], out[1]

    unresolved = tabled("sea_unresolved_variance", wind, cox_munk_variances)
    if gust is not None:
        unresolved = _gusty(tree, sea, gust, unresolved)
    if smooth is None:
        return unresolved
    slick = tabled("sea_unresolved_slick_variance", survivors, cox_munk_slick_variances)
    along, across = (
        _math(tree, "MULTIPLY_ADD", smooth, _math(tree, "SUBTRACT", s, v), v)
        for s, v in zip(slick, unresolved, strict=True)
    )
    return along, across


def _widest(
    tree: bpy.types.NodeTree,
    pixel: _Pixel,
    unresolved: tuple[bpy.types.NodeSocket, bpy.types.NodeSocket],
) -> tuple[bpy.types.NodeSocket, bpy.types.NodeSocket, bpy.types.NodeSocket]:
    """The wider and narrower of the unresolved variances, and the horizontal along
    the wider: Cycles' Metallic BSDF stretches a lobe along its tangent only. Grazing,
    the footprints make it the view's; a steep view down a crosswind-heavy sea turns
    it."""
    across_dir = _vector(tree, "CROSS_PRODUCT", pixel.along_dir, (0.0, 0.0, 1.0))
    tangent = tree.nodes.new("ShaderNodeMix")
    tangent.data_type = "VECTOR"
    tree.links.new(
        _math(tree, "GREATER_THAN", unresolved[1], unresolved[0]),
        tangent.inputs["Factor"],
    )
    tree.links.new(pixel.along_dir, tangent.inputs["A"])
    tree.links.new(across_dir, tangent.inputs["B"])
    return (
        _math(tree, "MAXIMUM", *unresolved, name="sea_unresolved_wide"),
        _math(tree, "MINIMUM", *unresolved, name="sea_unresolved_narrow"),
        tangent.outputs["Result"],
    )


def material(
    sea: Sea,
    wind: tuple[Wave, ...],
    swell: tuple[Wave, ...],
    outputs: Outputs,
    rngs: tuple[np.random.Generator, np.random.Generator, np.random.Generator],
    wakes: tuple[Wake, ...],
    light: Light,
    finest_footprint_m: float = 0.0,
) -> bpy.types.Material:
    """Each pixel draws the waves it resolves and takes the rest as roughness;
    `rngs` draw its gusts, its slicks and its wakes' foam. With a sky the sea is ir and
    reflects it; without, eo, glittering towards the light's sun if it has one.

    No node draws a wave a pixel of `finest_footprint_m` cannot resolve; the tables
    and statistics still describe the whole sea."""
    gust_rng, slick_rng, foam_rng = rngs
    material = bpy.data.materials.new("sea")
    tree = material.node_tree
    tree.nodes.clear()
    pixel = _pixel(tree)
    time_s = _sea_time(tree, outputs)
    speed = sea.wind_speed_mps
    drawn = (
        resolvable(wind, finest_footprint_m),
        resolvable(swell, finest_footprint_m),
    )
    # Each sea on its own, or a swell's height would decide which wind waves count.
    tall = tallest(drawn[0], PARALLAX_HEIGHT_SHARE) + tallest(
        drawn[1], PARALLAX_HEIGHT_SHARE
    )
    position = _waves(tree, *drawn, tall, time_s, pixel)
    gust_max, gust, slick = 0.0, None, None
    if speed > 0.0:
        tile = von_karman_field(gust_rng, GUST_CELLS, GUST_SPACING_M, GUST_LENGTH_M)
        # A crossfade reaches sqrt(2) of the tile's peak at most.
        gust_max = math.sqrt(2) * np.abs(tile).max() * turbulence_intensity(speed)
        gust = _gust(tree, sea, position, outputs, curve_image("sea_gust", tile))
        if sea.slick_cover > 0.0:
            slick = _slick(tree, sea, position, outputs, slick_rng)
    # Made where foam shows: an image nothing reads is not saved.
    foam = cache(
        lambda: curve_image(
            "sea_foam",
            von_karman_field(foam_rng, FOAM_CELLS, FOAM_SPACING_M, FOAM_LENGTH_M),
        )
    )
    survivors = (
        wind
        if slick is None
        else slick_survivors(speed, wind, math.radians(sea.wind_from_deg))
    )
    damped = frozenset(wind) - frozenset(survivors)
    calm = None
    if slick is not None and damped:
        calm = _math(tree, "SUBTRACT", 1.0, slick)
    normal = _normal(tree, *drawn, position, pixel, calm, damped)
    wake = None
    if wakes:
        sea_variance = cox_munk_slope(speed) ** 2
        wake = _wakes(tree, wakes, position, pixel.along_m, foam(), sea_variance)
    smooth = _smoothed(tree, slick, wake)
    if wake is not None and wake.tilt is not None:
        normal = _vector(
            tree,
            "NORMALIZE",
            _vector(tree, "ADD", normal, wake.tilt),
            name="wake_normal",
        )
    unresolved = _unresolved(tree, sea, wind, swell, survivors, pixel, gust, smooth)

    if light.sky is None:
        wide, narrow, tangent = _widest(tree, pixel, unresolved)
        carried: float | bpy.types.NodeSocket = 0.0
        if wind and light.sun is not None:
            # The wider spread reaches the most facets. A judgement.
            glinting = _glinting(tree, normal, light.sun, wide)
            tilt, carried = _glitter(tree, wind, position, pixel, narrow, glinting)
            normal = _vector(
                tree,
                "NORMALIZE",
                _vector(tree, "ADD", normal, tilt),
                name="glitter_normal",
            )
        wide = _math(tree, "MAXIMUM", _math(tree, "SUBTRACT", wide, carried), 0.0)
        # Cycles stretches a lobe 10:1 at most, alpha 10 to 1, variance 100.
        unresolved = (
            wide,
            _math(
                tree,
                "MAXIMUM",
                _math(tree, "SUBTRACT", narrow, carried),
                _math(tree, "MULTIPLY", wide, 0.01),
            ),
        )
        # Along the view, the widest footprint, leaves the most unresolved. A judgement.
        whitecaps = _whitecaps(
            tree,
            sea,
            wind,
            _crest_history(tree, wind, position, pixel, calm, damped),
            pixel.along_m,
            gust,
            gust_max,
            _foam_lace(tree, sea, position, outputs, foam()),
        )
        surface = _daylight(tree, sea, normal, tangent, unresolved, whitecaps, wake)
    else:
        # The coarsest footprint leaves the most, Cox & Munk's and all the swell's;
        # along the view a pixel takes twice the wider axis's, and the strongest gust
        # adds the most.
        axes = unresolved_slope_variances(
            speed, math.radians(sea.wind_from_deg), wind, swell, FOOTPRINT_RANGE_M[1]
        )
        slope_max = math.sqrt(
            2 * max(axes) + gust_max * gust_slope_variance(sea.wind_speed_mps)
        )
        surface = _thermal(tree, sea, normal, unresolved, slope_max, light.sky)
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    tree.links.new(surface, output.inputs["Surface"])
    return material


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
    # Flat faces would show their edges in the specular.
    for face in water.data.polygons:
        face.use_smooth = True
    water.data.materials.append(material)
    return water
