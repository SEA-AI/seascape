"""The sea's mesh and material: waves as shading on a grid curved to the earth.

Sources
-------
Microfacet lobe: Walter, Marschner, Li & Torrance, "Microfacet models for refraction
through rough surfaces", EGSR 2007 (doi:10.2312/EGWR/EGSR07/195-206) for Beckmann's
distribution and its Smith shadowing; Burley, "Physically-based shading at Disney",
SIGGRAPH 2012 course notes, for the alpha = roughness^2 convention Cycles follows.

Displacement: Gerstner waves (Gerstner 1802) as in Tessendorf, "Simulating ocean
water", SIGGRAPH 2001 course notes, eq. 29, with lambda = 1.

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
import shutil
from collections.abc import Callable, Iterator
from functools import cache, lru_cache, partial
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
    BEAUFORT_7_MPS,
    BEAUFORT_8_MPS,
    FADE_FOOTPRINTS,
    FINE_WINDROW_SPACING_M,
    FOAM_VISIBLE_COVER,
    GRAVITY_MS2,
    GUST_LENGTH_M,
    THIN_FOAM_SHARE,
    WHITECAP_EFFECTIVE_REFLECTANCE,
    WINDROW_ASPECT,
    WINDROW_MIN_WIND_MPS,
    WINDROW_SPACING_S,
    FoamGrid,
    Wave,
    cox_munk_slick_variances,
    cox_munk_slope,
    cox_munk_variances,
    earth_radius_m,
    fade_footprints_m,
    foam_drift_mps,
    foam_lattice_field,
    foam_memory,
    foam_memory_thresholds_g,
    gaussian_ranks,
    gust_slope_variance,
    horizon_m,
    resolvable,
    sea_z_m,
    slick_survivors,
    specular_cell_m2,
    tallest,
    turbulence_intensity,
    twinkle_hz,
    unresolved_slope_variances,
    von_karman_field,
    whitecap_fraction,
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


# A judgement: the footprints a sea pixel spans.
FOOTPRINT_RANGE_M = (1e-3, 1e4)

# A judgement: emissivity and the reflected sky move slowly with slope.
SLOPE_ROWS = 16
# Measured against `lwir.reflected_sky` evaluated exactly: bilinear in these, the
# sea's brightness temperature stays within 0.25 K, grazing views included.
VIEW_ROWS = 32
MIRROR_SAMPLES = 256

# Elevation (rad, ascending) and the IR sky's radiance there.
type SkyCurve = tuple[np.ndarray, np.ndarray]


class Seen(NamedTuple):
    """What the cameras see of the sea: their finest footprint, and the grids their
    whitecaps are remembered on."""

    finest_footprint_m: float = 0.0
    foam_grids: tuple[FoamGrid, ...] = ()


UNSEEN = Seen()


class Light(NamedTuple):
    """The ir sky round the horizon, or the unit vector an eo sea glints towards. A sky
    makes the sea ir, its cool skin thinner with the sun up."""

    sky: SkyCurve | None = None
    sun: tuple[float, float, float] | None = None
    sun_up: bool = True


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
# A judgement: a whitecap outline's ramp, per sigma of its lace, fresh and aged.
FRESH_EDGE = 6.0
# A judgement: foam's albedo swings this share of fresh foam's per sigma of density.
FOAM_ALBEDO_SWING = 0.2
# A judgement: foam's lumps, bumped from its density.
FOAM_BUMP_STRENGTH = 1.0
FOAM_BUMP_M = 0.5
AGED_EDGE = 2.0
# A judgement: how much the lattice frays a whitecap's outline.
LACE_RAGGED_WEIGHT = 0.4
# A judgement: the finer rows' tile, a few texels to a band 3-5 m wide (Zedel & Farmer
# 1991).
FINE_ROWS_TEXEL_M = 0.5
# Numerics: the offsets within a texel, per axis, a lookup's spread is averaged over.
FOAM_LOOKUP_SHARES = 8
# Numerics: a finer foam grid fades in over this many of its cells from its edge.
FOAM_SEAM_CELLS = 8

# A judgement: foam patches a metre or two across, streaked 4:1 along the wake.
FOAM_CELLS = 512
FOAM_SPACING_M = 0.25
FOAM_LENGTH_M = 1.0
FOAM_TILE_M = FOAM_CELLS * FOAM_SPACING_M
FOAM_STREAK = 4.0
# A judgement: aged foam's lattice, the walls of cells 1.3 m and 40 cm across among
# openings of 2 m, over a 16 m tile.
LATTICE_CELLS = 512
LATTICE_TILE_M = 16.0
LATTICE_CELL_SIZES_M = (16 / 12, 16 / 40)
LATTICE_OPENING_M = 2.0
# A judgement: the width of a cell wall, in metres.
LATTICE_WALL_M = 0.15
# A judgement: a lattice edge's ramp, per sigma of the field.
LATTICE_SHARPNESS = 4.0
# A judgement: a second, finer octave of the lattice, weighted so.
LATTICE_OCTAVE = 4.0
LATTICE_FINE_WEIGHT = 0.8
# A judgement: how long a streak of aged foam is for its width at force 8.
LATTICE_STREAK = 4.0


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


def _foam_emissivity_image() -> bpy.types.Image:
    """`lwir.foam_emissivity_excess` against cos(theta) at texel centres."""
    mu = (np.arange(CURVE_SAMPLES) + 0.5) / CURVE_SAMPLES
    return curve_image(
        "sea_foam_emissivity", lwir.foam_emissivity_excess(np.arccos(mu))
    )


def _view_rows() -> np.ndarray:
    """|cos| of the view off the wave normal up a tile's rows, at texel centres: even
    in its square root, so the rows crowd toward grazing, where the reflection turns
    fastest."""
    return ((np.arange(VIEW_ROWS) + 0.5) / VIEW_ROWS) ** 2


def _mirror_columns() -> np.ndarray:
    """sin(mirror elevation) along a row, at texel centres: down to -1, as a wave
    tilted away steeply mirrors the sea."""
    return (np.arange(MIRROR_SAMPLES) + 0.5) / MIRROR_SAMPLES * 2.0 - 1.0


def _reflection_image(
    t_sea_k: float, sky: SkyCurve, slope_max: float
) -> bpy.types.Image:
    """`lwir.reflected_sky` baked against sin(mirror elevation) along a row
    (`_mirror_columns`), the view off the wave normal up a tile's rows
    (`_view_rows`), and the unresolved RMS slope from tile to tile, all at texel
    centres."""
    elevation, radiance = (np.ascontiguousarray(a, np.float64).tobytes() for a in sky)
    return curve_image(
        "sea_reflection", _reflection_table(t_sea_k, elevation, radiance, slope_max)
    )


# The bake is most of an IR build, and builds in one process often share a sea.
# Arrays do not hash, so the sky comes as float64 bytes.
@lru_cache(maxsize=8)
def _reflection_table(
    t_sea_k: float, elevation: bytes, radiance: bytes, slope_max: float
) -> np.ndarray:
    mirror = np.arcsin(_mirror_columns())
    sky = np.frombuffer(elevation), np.frombuffer(radiance)
    return np.array(
        [
            lwir.reflected_sky(
                mirror, *sky, t_sea_k=t_sea_k, slope_sigma=sigma, view_rad=view
            )
            for sigma in _row_sigmas(slope_max)
            for view in np.arcsin(_view_rows())
        ]
    )


def _tiled(
    tree: bpy.types.NodeTree,
    image: bpy.types.Image,
    x: bpy.types.NodeSocket,
    row: bpy.types.NodeSocket,
    tile: bpy.types.NodeSocket,
) -> bpy.types.NodeSocket:
    """`_reflection_image` at (x, row, tile), each in [0, 1], row the square root of
    `_view_rows`' |cos|. The image interpolates within a tile and two lookups mix
    across tiles, so no tile bleeds into the next."""
    # Held to texel centres, as EXTEND holds the image's own edges.
    texel = _math(
        tree,
        "MINIMUM",
        _math(tree, "MAXIMUM", _math(tree, "MULTIPLY", row, VIEW_ROWS), 0.5),
        VIEW_ROWS - 0.5,
    )
    at = _math(
        tree,
        "MINIMUM",
        _math(
            tree,
            "MAXIMUM",
            _math(tree, "MULTIPLY_ADD", tile, SLOPE_ROWS, -0.5),
            0.0,
        ),
        SLOPE_ROWS - 1.0,
    )
    lower = _math(tree, "FLOOR", at)
    upper = _math(tree, "MINIMUM", _math(tree, "ADD", lower, 1.0), SLOPE_ROWS - 1.0)
    height = SLOPE_ROWS * VIEW_ROWS
    lo, hi = (
        lookup(
            tree,
            image,
            x,
            _math(
                tree,
                "DIVIDE",
                _math(tree, "MULTIPLY_ADD", t, VIEW_ROWS, texel),
                height,
            ),
        )
        for t in (lower, upper)
    )
    weight = _math(tree, "SUBTRACT", at, lower)
    return _math(tree, "MULTIPLY_ADD", weight, _math(tree, "SUBTRACT", hi, lo), lo)


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


def _pixel(
    tree: bpy.types.NodeTree, incoming: bpy.types.NodeSocket | None = None
) -> _Pixel:
    """The footprint of a pixel of the scene's perspective camera at the render's
    resolution, Resolution % included. `incoming` stands in for the Geometry node's,
    which a displacement shader lacks."""
    camera = tree.nodes.new("ShaderNodeCameraData")
    geometry = tree.nodes.new("ShaderNodeNewGeometry")
    if incoming is None:
        incoming = geometry.outputs["Incoming"]
    facing = _vector(tree, "DOT_PRODUCT", incoming, geometry.outputs["Normal"])
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
    level = _vector(tree, "SUBTRACT", incoming, tilt)
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


def _clamped(tree: bpy.types.NodeTree, x: bpy.types.NodeSocket) -> bpy.types.NodeSocket:
    out = _math(tree, "MAXIMUM", x, 0.0)
    out.node.use_clamp = True
    return out


def _texels(tile: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """(rows, columns): `tile` (rows bottom first, as `curve_image` lays it) at each
    column's u and each row's v, in tiles, as Cycles' linear lookup with repeat reads
    it. Separable: two tile rows blended per row, then two columns per column."""
    height, width = tile.shape
    x, y = u * width - 0.5, v * height - 0.5
    i, j = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - i, (y - j)[:, None]
    rows = (1 - fy) * tile[j % height] + fy * tile[(j + 1) % height]
    return rows[:, i % width] * (1 - fx) + rows[:, (i + 1) % width] * fx


def _looked_up_sigma(tile: np.ndarray) -> float:
    """`tile`'s spread as Cycles' linear lookup reads it, between texels as on them:
    interpolation averages neighbours, so it is under the texels' own."""
    height, width = tile.shape
    shares = (np.arange(FOAM_LOOKUP_SHARES) + 0.5) / FOAM_LOOKUP_SHARES
    variance = np.mean(
        [
            np.var(
                _texels(
                    tile,
                    (np.arange(width) + 0.5 + a) / width,
                    (np.arange(height) + 0.5 + b) / height,
                )
            )
            for a in shares
            for b in shares
        ]
    )
    return float(np.sqrt(variance))


def _write_frames(
    folder: Path,
    grid: FoamGrid,
    frames: Iterator[tuple[np.ndarray, np.ndarray]],
) -> None:
    """Each frame's (cover, breaking) as an EXR in `folder`."""
    folder.mkdir(parents=True)
    image = bpy.data.images.new(
        "sea_foam_frame", grid.columns, grid.rows, float_buffer=True
    )
    image.colorspace_settings.name = "Non-Color"
    # `Image.save` writes 32-bit floats uncompressed; a scene's output settings write
    # half floats, which keep cover to a part in 2048, zipped.
    frames_scene = bpy.data.scenes.new("sea_foam_frames")
    settings = frames_scene.render.image_settings
    settings.file_format, settings.color_mode = "OPEN_EXR", "RGB"
    settings.color_depth, settings.exr_codec = "16", "ZIP"
    pixels = np.zeros((grid.rows, grid.columns, 4), np.float32)
    pixels[..., 3] = 1.0
    for i, (cover, breaking) in enumerate(frames):
        pixels[..., 0], pixels[..., 1] = cover, breaking
        image.pixels.foreach_set(pixels.ravel())
        image.save_render(str(folder / f"foam_{i + 1:04d}.exr"), scene=frames_scene)
    bpy.data.scenes.remove(frames_scene)
    bpy.data.images.remove(image)


# The foam memories the last build asked for and has not simulated: a build that never
# renders never pays for them.
_PENDING: list[Callable[[], None]] = []


def _frames_folder() -> Path:
    """Where a build writes its foam memory: Blender's session temporary folder, which
    it deletes on exit."""
    return Path(bpy.app.tempdir) / "sea_foam"


def forget_foam() -> None:
    """Drop the last build's foam memory, pending or written: every build starts here,
    with foam or without."""
    _PENDING.clear()
    shutil.rmtree(_frames_folder(), ignore_errors=True)


def _foam_sequences() -> list[bpy.types.Image]:
    return [
        image
        for image in bpy.data.images
        if image.name.startswith("sea_foam_") and image.source == "SEQUENCE"
    ]


def remember_foam(*_: object) -> None:
    """Simulate every foam memory the last build left pending, and point their image
    sequences at the frames: on `render_pre`, and by `keep_foam`."""
    while _PENDING:
        _PENDING.pop()()
    for image in _foam_sequences():
        image.reload()


def keep_foam(blend: Path) -> None:
    """Simulate the foam memory and move its frames beside `blend`, where a `.blend`
    about to be saved there finds them, as Blender keeps a simulation's bake."""
    remember_foam()
    # A rebuild without foam leaves no frames of the last one beside it.
    beside = blend.with_suffix(".foam")
    if beside.exists():
        shutil.rmtree(beside)
    frames = _frames_folder()
    if not frames.is_dir():
        return
    shutil.move(frames, beside)
    for image in _foam_sequences():
        moved = beside / Path(image.filepath).relative_to(frames)
        image.filepath = bpy.path.relpath(str(moved), start=str(blend.parent))


def _foam_frames(
    sea: Sea,
    wind: tuple[Wave, ...],
    outputs: Outputs,
    grids: tuple[FoamGrid, ...],
) -> list[tuple[FoamGrid, bpy.types.Image]]:
    """Each grid's foam memory over the clip, an image sequence of (cover, breaking),
    simulated before the first render, breaking at Monahan's cover for the mean wind.

    No gust moves the threshold: whitecaps break off the drawn waves, which grow too
    slowly to follow a gust (Plant 1982), as `_gusty` holds them at the mean wind."""
    drift = surface_drift(sea, outputs)
    fraction = whitecap_fraction(sea.wind_speed_mps)
    if remember_foam not in bpy.app.handlers.render_pre:
        bpy.app.handlers.render_pre.append(remember_foam)
    out = []
    for grid in grids:
        folder = _frames_folder() / f"{grid.cell_m:g}m"
        _PENDING.append(
            partial(_simulate, wind, outputs, grid, folder, drift, fraction)
        )
        sequence = bpy.data.images.new(
            f"sea_foam_{grid.cell_m:g}m_frames", 1, 1, float_buffer=True
        )
        sequence.source = "SEQUENCE"
        sequence.filepath = str(folder / "foam_0001.exr")
        sequence.colorspace_settings.name = "Non-Color"
        out.append((grid, sequence))
    return out


def _simulate(
    wind: tuple[Wave, ...],
    outputs: Outputs,
    grid: FoamGrid,
    folder: Path,
    drift: tuple[float, float],
    fraction: float,
) -> None:
    """One grid's foam memory over the clip, written to `folder`."""
    step = 1 / outputs.fps
    (threshold,) = foam_memory_thresholds_g(wind, drift, grid.cell_m, step, (fraction,))
    times = outputs.times_s
    _write_frames(
        folder,
        grid,
        foam_memory(wind, drift, grid, times[0], step, len(times), float(threshold)),
    )


def _remembered(
    tree: bpy.types.NodeTree,
    sea: Sea,
    outputs: Outputs,
    position: bpy.types.NodeSocket,
    footprint_m: bpy.types.NodeSocket,
    textures: "_FoamTexture",
    memory: list[tuple[FoamGrid, bpy.types.Image]],
    pulls: tuple[bpy.types.NodeSocket, ...],
) -> tuple[bpy.types.NodeSocket, bpy.types.NodeSocket, bpy.types.NodeSocket]:
    """How much of the pixel foam covers; that, as a share of fresh foam's brightness;
    and the share of the sea under foam, where its bubbles are: read off the finest
    foam memory that holds `position`, (east, north, time), aged foam `gathered` by the
    windrows, the lace above its quantile covering that share."""
    lace = textures.lace
    split = tree.nodes.new("ShaderNodeSeparateXYZ")
    tree.links.new(position, split.inputs["Vector"])
    drift = surface_drift(sea, outputs)
    # The memory's grid drifts: water at x is at grid x - v t.
    on_grid = [
        _math(tree, "MULTIPLY_ADD", split.outputs["Z"], -v, split.outputs[axis])
        for axis, v in zip("XY", drift, strict=True)
    ]
    # Past the coarsest grid the foam is finer than a pixel: Monahan's cover, as bright
    # on average as Koepke measured.
    cover: bpy.types.NodeSocket | float = whitecap_fraction(sea.wind_speed_mps)
    breaking: bpy.types.NodeSocket | float = (
        WHITECAP_EFFECTIVE_REFLECTANCE / FRESH_FOAM_REFLECTANCE - THIN_FOAM_SHARE
    ) / (1 - THIN_FOAM_SHARE)
    # Coarsest first, so each finer grid overwrites where it holds.
    for grid, sequence in sorted(memory, key=lambda m: -m[0].cell_m):
        u = _math(
            tree,
            "MULTIPLY_ADD",
            on_grid[0],
            1 / (grid.columns * grid.cell_m),
            -grid.west_m / (grid.columns * grid.cell_m),
        )
        w = _math(
            tree,
            "MULTIPLY_ADD",
            on_grid[1],
            1 / (grid.rows * grid.cell_m),
            -grid.south_m / (grid.rows * grid.cell_m),
        )
        at = tree.nodes.new("ShaderNodeCombineXYZ")
        tree.links.new(u, at.inputs["X"])
        tree.links.new(w, at.inputs["Y"])
        texture = tree.nodes.new("ShaderNodeTexImage")
        texture.name = f"sea_foam_{grid.cell_m:g}m"
        texture.image = sequence
        texture.extension = "CLIP"
        # Linear lookup lays the cells' squares into the lace's edge.
        texture.interpolation = "Cubic"
        texture.image_user.frame_duration = len(outputs.times_s)
        texture.image_user.frame_start = 0
        texture.image_user.use_auto_refresh = True
        tree.links.new(at.outputs["Vector"], texture.inputs["Vector"])
        channels = tree.nodes.new("ShaderNodeSeparateColor")
        tree.links.new(texture.outputs["Color"], channels.inputs["Color"])
        # Taken where the pixel spans under a cell, fading out by two, and faded in from
        # its edge, so neither a finer grid's reach nor its border leaves a seam.
        fine_enough = _clamped(
            tree, _math(tree, "MULTIPLY_ADD", footprint_m, -1 / grid.cell_m, 2.0)
        )
        inside = _math(
            tree,
            "MULTIPLY",
            *(
                _clamped(
                    tree,
                    _math(
                        tree,
                        "MULTIPLY",
                        _math(tree, "MINIMUM", x, _math(tree, "SUBTRACT", 1.0, x)),
                        cells / FOAM_SEAM_CELLS,
                    ),
                )
                for x, cells in ((u, grid.columns), (w, grid.rows))
            ),
        )
        inside = _math(tree, "MULTIPLY", inside, fine_enough)
        cover = _math(
            tree,
            "MULTIPLY_ADD",
            inside,
            _math(tree, "SUBTRACT", channels.outputs["Red"], cover),
            cover,
        )
        breaking = _math(
            tree,
            "MULTIPLY_ADD",
            inside,
            _math(tree, "SUBTRACT", channels.outputs["Green"], breaking),
            breaking,
        )
    assert isinstance(cover, bpy.types.NodeSocket)
    if pulls:
        # Each pull s g is clipped to [-a, a]: g is Gaussian, so 1 + clamp(s g) averages
        # 1 and Monahan's cover holds; (1 + a)^2 under 1 / cover keeps it under full.
        # The floor keeps INVERSE_SQRT, which reads 0 at 0, off an empty cell.
        headroom = _math(
            tree,
            "MINIMUM",
            _math(
                tree,
                "SUBTRACT",
                _math(tree, "INVERSE_SQRT", _math(tree, "MAXIMUM", cover, 1e-6)),
                1.0,
            ),
            1.0,
        )
        floor = _math(tree, "MULTIPLY", headroom, -1.0)
        densities = [
            _math(
                tree,
                "ADD",
                _math(tree, "MINIMUM", _math(tree, "MAXIMUM", pull, floor), headroom),
                1.0,
            )
            for pull in pulls
        ]
        gathered = densities[0]
        for density in densities[1:]:
            gathered = _math(tree, "MULTIPLY", gathered, density)
        gathered.node.name = "whitecap_gathered"
        # Foam still breaking has not been gathered yet.
        kept = _math(
            tree,
            "MULTIPLY_ADD",
            breaking,
            _math(tree, "SUBTRACT", 1.0, gathered),
            gathered,
        )
        cover = _math(tree, "MULTIPLY", cover, kept)
    levels = (np.arange(CURVE_SAMPLES) + 0.5) / CURVE_SAMPLES
    ppf = curve_image("normal_ppf", np.vectorize(NormalDist().inv_cdf)(1 - levels))
    # Above its 1 - cover quantile, the lace covers `cover` of the pixel.
    laced = _math(
        tree,
        "MULTIPLY_ADD",
        _math(tree, "SUBTRACT", lace, lookup(tree, ppf, cover)),
        # Foam still breaking is dense to its edge; aged foam thins out to it.
        _math(tree, "MULTIPLY_ADD", breaking, FRESH_EDGE - AGED_EDGE, AGED_EDGE),
        0.5,
    )
    laced.node.use_clamp = True
    laced = _math(
        tree,
        "MULTIPLY",
        laced,
        _math(tree, "GREATER_THAN", cover, FOAM_VISIBLE_COVER),
        name="whitecaps",
    )
    bright = _math(
        tree,
        "MULTIPLY_ADD",
        breaking,
        1 - THIN_FOAM_SHARE,
        THIN_FOAM_SHARE,
        name="whitecap_bright",
    )
    foam = _latticed(tree, laced, bright, textures, lookup(tree, ppf, bright))
    under = _math(tree, "MAXIMUM", cover, 0.0, name="whitecap_bubbles")
    return laced, foam, under


def _latticed(
    tree: bpy.types.NodeTree,
    laced: bpy.types.NodeSocket,
    bright: bpy.types.NodeSocket,
    texture: "_FoamTexture",
    threshold: bpy.types.NodeSocket,
) -> bpy.types.NodeSocket:
    """Foam over the `laced` area at `bright` of fresh foam's brightness, drawn as
    fresh foam's white over `bright` of that area, the lattice above `threshold`, its
    quantile, not a grey film: the same mean. Its albedo, varying with the texture's
    density about fresh foam's, is the node `whitecap_albedo`."""
    held = _math(
        tree,
        "MULTIPLY_ADD",
        _math(tree, "SUBTRACT", texture.lattice, threshold),
        LATTICE_SHARPNESS,
        0.5,
    )
    held.node.use_clamp = True
    # A pixel longer than the lattice's walls draws their mean: from one wall's width,
    # gone at two. Its longer side, as a mipmap without anisotropic filtering takes.
    fade = _clamped(
        tree, _math(tree, "MULTIPLY_ADD", texture.pixel_m, 1 / LATTICE_WALL_M, -1.0)
    )
    held = _math(
        tree, "MULTIPLY_ADD", fade, _math(tree, "SUBTRACT", bright, held), held
    )
    # The density's symmetric clip keeps the mean albedo fresh foam's; it fades with
    # the walls, and with it the bump it lifts.
    swing = _math(
        tree,
        "MULTIPLY",
        _math(tree, "MINIMUM", _math(tree, "MAXIMUM", texture.shade, -2.0), 2.0),
        _math(tree, "SUBTRACT", 1.0, fade),
    )
    _math(
        tree,
        "MULTIPLY_ADD",
        swing,
        FRESH_FOAM_REFLECTANCE * FOAM_ALBEDO_SWING,
        FRESH_FOAM_REFLECTANCE,
        name="whitecap_albedo",
    )
    return _math(tree, "MULTIPLY", laced, held, name="whitecap_foam")


def _bare(
    tree: bpy.types.NodeTree,
) -> tuple[bpy.types.NodeSocket, bpy.types.NodeSocket, bpy.types.NodeSocket]:
    """No whitecaps: a calm sea, or one no camera sees."""
    area, foam, under = (
        _math(tree, "MAXIMUM", 0.0, 0.0, name=name)
        for name in ("whitecaps", "whitecap_foam", "whitecap_bubbles")
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
    sun_up: bool,
    foam: bpy.types.NodeSocket,
) -> bpy.types.NodeSocket:
    """eps(theta) of the sea emitted, the remaining 1 - eps reflected from the sky.
    It emits at its skin's temperature; the tables keep the bulk's optical constants,
    which a skin a few kelvin off barely moves. `foam`, the share of the pixel under
    foam, raises eps by that share of foam's excess.

    Complements, so the two very nearly cancel and the sea holds close to ambient at
    every angle. The reflection is a lookup in the wave normal's mirror direction, not
    a BSDF: Cycles sampling a lobe across a sky cold overhead leaves the sea grainy.
    Only the slope along the view enters: the slope across turns the reflection
    sideways, which lowers it only at second order. Derivation: a tilt d across the
    view scales the mirror's sin(elevation) by (1 - d^2) / (1 + d^2).

    The unresolved facets lean about the wave normal, so the reflection is tabulated
    against the view off the wave normal as well as the mirror off the horizon.

    ponytail: reflects the sky averaged round the horizon, no hulls or clouds at their
    bearing; trace a sharp mirror if those must show.
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
    incidence = _incidence(tree, normal)
    sky_seen = _tiled(
        tree,
        _reflection_image(sea.t_sea_k, sky, slope_max),
        _math(tree, "MULTIPLY_ADD", mirror.outputs["Z"], 0.5, 0.5),
        _math(tree, "SQRT", incidence),
        along,
    )
    sky_seen.node.name = "sea_reflection"
    mean = _math(tree, "MULTIPLY", _math(tree, "ADD", *unresolved), 0.5)
    fraction = _math(tree, "DIVIDE", _math(tree, "SQRT", mean), slope_max)
    bare = lookup(tree, _emissivity_image(sea.t_sea_k, slope_max), incidence, fraction)
    excess = lookup(tree, _foam_emissivity_image(), incidence)
    emissivity = _math(
        tree,
        "MINIMUM",
        _math(tree, "MULTIPLY_ADD", foam, excess, bare),
        1.0,
        name="sea_emissivity",
    )
    skin_radiance = lwir.band_radiance(sea.t_skin_k(sun_up))
    emitted = _math(tree, "SUBTRACT", skin_radiance, sky_seen, name="sea_skin")
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
    if "whitecap_albedo" in tree.nodes:
        link(tree.nodes["whitecap_albedo"].outputs[0], foam.inputs["Color"])
        # Foam is lumpy: its own normal, bumped by its density over the water's.
        bump = tree.nodes.new("ShaderNodeBump")
        bump.name = "whitecap_bump"
        bump.inputs["Strength"].default_value = FOAM_BUMP_STRENGTH
        bump.inputs["Distance"].default_value = FOAM_BUMP_M
        link(tree.nodes["whitecap_albedo"].outputs[0], bump.inputs["Height"])
        link(normal, bump.inputs["Normal"])
        link(bump.outputs["Normal"], foam.inputs["Normal"])
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


class _FoamTexture(NamedTuple):
    """Unit Gaussians drifting with the surface: whitecaps' outline, aged foam's
    lattice, and its density; and the pixel's longer side, which fades the lattice."""

    lace: bpy.types.NodeSocket
    lattice: bpy.types.NodeSocket
    shade: bpy.types.NodeSocket
    pixel_m: bpy.types.NodeSocket


def _foam_texture(
    tree: bpy.types.NodeTree,
    sea: Sea,
    where: bpy.types.NodeSocket,
    outputs: Outputs,
    tile: bpy.types.Image,
    rng: np.random.Generator,
    pixel_m: bpy.types.NodeSocket,
) -> _FoamTexture:
    """Whitecap foam's texture read at `where`, (east, north, time): the lace off
    `tile`, frayed by a lattice drawn from `rng`, and that lattice twice more, for a
    pixel `pixel_m` long."""
    lattice = curve_image(
        "sea_foam_lattice",
        foam_lattice_field(
            rng,
            LATTICE_CELLS,
            LATTICE_TILE_M,
            LATTICE_CELL_SIZES_M,
            LATTICE_OPENING_M,
        ),
    )
    # A quarter tile apart, each read is independent of the others.
    ragged, holes, shade = (
        _foam_lattice(tree, sea, where, outputs, lattice, start)
        for start in (0.25, 0.0, 0.5)
    )
    return _FoamTexture(
        _foam_lace(tree, sea, where, outputs, tile, ragged), holes, shade, pixel_m
    )


def _foam_lace(
    tree: bpy.types.NodeTree,
    sea: Sea,
    where: bpy.types.NodeSocket,
    outputs: Outputs,
    tile: bpy.types.Image,
    ragged: bpy.types.NodeSocket,
) -> bpy.types.NodeSocket:
    """Whitecap foam's lace, a unit Gaussian drifting with the surface, read at `where`,
    (east, north, time): a read of `tile` and the `ragged` lattice, rescaled to one."""
    width, height = tile.size
    texels = np.empty(width * height * 4, np.float32)
    tile.pixels.foreach_get(texels)
    sigma = _looked_up_sigma(texels.reshape(height, width, 4)[..., 0])
    east, north = _downwind(sea)
    per_m = 1 / FOAM_TILE_M
    along = per_m / FOAM_STREAK
    coarse = _drifting(
        tree,
        where,
        outputs,
        tile,
        ((along * east, along * north), (per_m * north, -per_m * east)),
        surface_drift(sea, outputs),
        0.0,
        "whitecap_lace",
    )
    coarse = _math(tree, "MULTIPLY", coarse, 1 / sigma)
    weight = LACE_RAGGED_WEIGHT
    summed = _math(tree, "MULTIPLY_ADD", ragged, weight, coarse)
    return _math(tree, "MULTIPLY", summed, 1 / math.sqrt(1 + weight**2))


def _foam_lattice(
    tree: bpy.types.NodeTree,
    sea: Sea,
    where: bpy.types.NodeSocket,
    outputs: Outputs,
    tile: bpy.types.Image,
    start: float = 0.0,
) -> bpy.types.NodeSocket:
    """Aged foam's lattice, a unit Gaussian drifting with the surface, read at
    `where`, (east, north, time), offset `start` tiles."""
    width, height = tile.size
    texels = np.empty(width * height * 4, np.float32)
    tile.pixels.foreach_get(texels)
    sigma = _looked_up_sigma(texels.reshape(height, width, 4)[..., 0])
    east, north = _downwind(sea)
    layers = []
    # From force 7 the lattice is blown into streaks along the wind.
    stretch = 1 + (LATTICE_STREAK - 1) * _streaking(sea)
    for scale, shift in ((1.0, start), (LATTICE_OCTAVE, start + 0.5)):
        per_m = scale / LATTICE_TILE_M
        along = per_m / stretch
        layers.append(
            _drifting(
                tree,
                where,
                outputs,
                tile,
                ((along * east, along * north), (per_m * north, -per_m * east)),
                surface_drift(sea, outputs),
                shift,
                "whitecap_lattice",
            )
        )
    weight = LATTICE_FINE_WEIGHT
    summed = _math(tree, "MULTIPLY_ADD", layers[1], weight, layers[0])
    return _math(tree, "MULTIPLY", summed, 1 / (sigma * math.sqrt(1 + weight**2)))


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

    # A tile that holds still loops as it is.
    if not outputs.loop or velocity_mps == (0.0, 0.0):
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


def surface_drift(sea: Sea, outputs: Outputs) -> tuple[float, float]:
    """(east, north) m/s the surface drifts at, carrying foam and windrows alike; a
    loop's holds still, as the foam memory's drift would jump at its seam."""
    if outputs.loop:
        return 0.0, 0.0
    east, north = _downwind(sea)
    return foam_drift_mps(sea.wind_speed_mps, math.atan2(east, north))


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


class _RowRead(NamedTuple):
    """A row field's unit Gaussian convergence, made when first read, as an image
    nothing reads is not saved; and the rows' spacing."""

    read: Callable[[], bpy.types.NodeSocket]
    spacing_m: float


class _Rows(NamedTuple):
    windrows: _RowRead
    fine: _RowRead


def _langmuir_rows(
    tree: bpy.types.NodeTree,
    sea: Sea,
    where: bpy.types.NodeSocket,
    outputs: Outputs,
    rng: np.random.Generator,
) -> _Rows:
    """Langmuir circulation's convergence, which gathers slicks and aged foam alike:
    the windrows and the finer rows between them, each exactly Gaussian, so a
    symmetric clip keeps its mean and a quantile its share. `rng` draws the windrows
    first, as every reader reads them first.

    ponytail: the integral scale stands for the windrow spacing, a judgement, and
    there are rows at every wind, where below the cutoff real slicks lie in patches.
    Give the slicks their own spectrum if the rows read wrong.
    """
    spacing_m = WINDROW_SPACING_S * max(sea.wind_speed_mps, WINDROW_MIN_WIND_MPS)
    fine_tile_m = GUST_CELLS * FINE_ROWS_TEXEL_M

    def windrows() -> bpy.types.NodeSocket:
        tile = gaussian_ranks(
            von_karman_field(rng, GUST_CELLS, GUST_SPACING_M, spacing_m)
        )
        return _rows(tree, sea, where, outputs, tile, GUST_TILE_M, "windrows")

    def fine() -> bpy.types.NodeSocket:
        tile = gaussian_ranks(
            von_karman_field(rng, GUST_CELLS, FINE_ROWS_TEXEL_M, FINE_WINDROW_SPACING_M)
        )
        return _rows(tree, sea, where, outputs, tile, fine_tile_m, "fine_rows")

    return _Rows(
        _RowRead(cache(windrows), spacing_m),
        _RowRead(cache(fine), FINE_WINDROW_SPACING_M),
    )


def _rows(
    tree: bpy.types.NodeTree,
    sea: Sea,
    where: bpy.types.NodeSocket,
    outputs: Outputs,
    tile: np.ndarray,
    tile_m: float,
    name: str,
) -> bpy.types.NodeSocket:
    """`tile`, `tile_m` a side, stretched along the wind and drifting with the surface;
    read at `where`, (east, north, time)."""
    east, north = _downwind(sea)
    across = 1 / tile_m
    along = across / WINDROW_ASPECT
    return _drifting(
        tree,
        where,
        outputs,
        curve_image(f"sea_{name}", tile),
        ((north * across, -east * across), (east * along, north * along)),
        surface_drift(sea, outputs),
        0.0,
        name,
    )


def _slick(
    tree: bpy.types.NodeTree, sea: Sea, rows: _Rows
) -> bpy.types.NodeSocket | None:
    """1 under a slick: where the windrows' convergence is in its top `slick_cover`;
    None with no slicks.

    ponytail: whitecaps break as on the clean sea; simulate the foam memory on the
    survivors under a slick if slicks foam too often.
    """
    if sea.slick_cover <= 0.0:
        return None
    # The tile is Gaussian, so its top `slick_cover` lies above this.
    threshold = NormalDist().inv_cdf(1.0 - sea.slick_cover)
    return _math(tree, "GREATER_THAN", rows.windrows.read(), threshold, name="slick")


def _streaking(sea: Sea) -> float:
    """How far foam is blown into streaks: 0 below Beaufort 7, 1 from force 8."""
    # A judgement: streaks from none at force 7 to full strength at force 8.
    share = (sea.wind_speed_mps - BEAUFORT_7_MPS) / (BEAUFORT_8_MPS - BEAUFORT_7_MPS)
    return min(max(share, 0.0), 1.0)


def _gathered(
    tree: bpy.types.NodeTree,
    sea: Sea,
    footprint_m: bpy.types.NodeSocket,
    rows: _Rows,
) -> tuple[bpy.types.NodeSocket, ...]:
    """How hard each of `rows` pulls aged foam into its convergence g: s g, s rising
    from Beaufort 7's streaks to 8's; none below force 7."""
    share = _streaking(sea)
    if share == 0.0:
        return ()
    pulls = []
    for read, spacing_m in rows:
        # A pixel wider than half the rows' spacing cannot draw them: they fade, gone
        # at a whole spacing.
        fade = _clamped(
            tree, _math(tree, "MULTIPLY_ADD", footprint_m, -2 / spacing_m, 2.0)
        )
        pulls.append(
            _math(tree, "MULTIPLY", read(), _math(tree, "MULTIPLY", fade, share))
        )
    return tuple(pulls)


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
    seen: Seen = UNSEEN,
) -> bpy.types.Material:
    """Each pixel draws the waves it resolves and takes the rest as roughness;
    `rngs` draw its gusts, its windrows and its foam, whose whitecaps are remembered on
    the grids its cameras see. With a sky the sea is ir and reflects it; without, eo,
    glittering towards the light's sun if it has one.

    No node draws a wave a pixel of the cameras' finest footprint cannot resolve; the
    tables and statistics still describe the whole sea."""
    gust_rng, slick_rng, foam_rng = rngs
    material = bpy.data.materials.new("sea")
    tree = material.node_tree
    tree.nodes.clear()
    pixel = _pixel(tree)
    time_s = _sea_time(tree, outputs)
    speed = sea.wind_speed_mps
    drawn = (
        resolvable(wind, seen.finest_footprint_m),
        resolvable(swell, seen.finest_footprint_m),
    )
    tall, shaded = displaced(drawn)
    position = _flat(tree, time_s)
    rows = _langmuir_rows(tree, sea, position, outputs, slick_rng)
    gust_max, gust, slick, tile = 0.0, None, None, None
    if speed > 0.0:
        tile = von_karman_field(gust_rng, GUST_CELLS, GUST_SPACING_M, GUST_LENGTH_M)
        # A crossfade reaches sqrt(2) of the tile's peak at most.
        gust_max = math.sqrt(2) * np.abs(tile).max() * turbulence_intensity(speed)
        gust = _gust(tree, sea, position, outputs, curve_image("sea_gust", tile))
        slick = _slick(tree, sea, rows)
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
    calm = _math(tree, "SUBTRACT", 1.0, slick) if slick is not None and damped else None
    normal = _normal(tree, *shaded, position, pixel, calm, damped)
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
    whitecaps = (
        _remembered(
            tree,
            sea,
            outputs,
            position,
            pixel.across_m,
            _foam_texture(
                tree,
                sea,
                position,
                outputs,
                foam(),
                # Its own stream, so the lace's draws stay as they were.
                foam_rng.spawn(1)[0],
                pixel.along_m,
            ),
            _foam_frames(sea, wind, outputs, seen.foam_grids),
            _gathered(tree, sea, pixel.along_m, rows),
        )
        if wind and seen.foam_grids
        else _bare(tree)
    )

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
        surface = _thermal(
            tree,
            sea,
            normal,
            unresolved,
            slope_max,
            light.sky,
            light.sun_up,
            whitecaps[0]
            if wake is None
            else _math(tree, "MAXIMUM", whitecaps[0], wake.foam),
        )
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    tree.links.new(surface, output.inputs["Surface"])
    _displace(material, output, _displacement(tree, tall, position))
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
