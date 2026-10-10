"""The sea's mesh and material: waves as shading on a grid curved to the earth.

Sources
-------
Microfacet lobe: Walter, Marschner, Li & Torrance, "Microfacet models for refraction
through rough surfaces", EGSR 2007 (doi:10.2312/EGWR/EGSR07/195-206) for GGX; Burley,
"Physically-based shading at Disney", SIGGRAPH 2012 course notes, for the alpha =
roughness^2 convention Cycles follows.

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
from seascape.waves import (
    FADE_FOOTPRINTS,
    GRAVITY_MS2,
    GUST_LENGTH_M,
    SLICK_DRIFT,
    WINDROW_ASPECT,
    WINDROW_MIN_WIND_MPS,
    WINDROW_SPACING_S,
    Wave,
    breaking_threshold_g,
    cox_munk_slick_slope,
    cox_munk_slope,
    earth_radius_m,
    fade_footprints_m,
    gust_slope_variance,
    gusty_whitecap_fraction,
    horizon_m,
    sea_z_m,
    slick_survivors,
    specular_cell_m2,
    turbulence_intensity,
    twinkle_hz,
    unresolved_acceleration_variance,
    unresolved_slope_variance,
    von_karman_field,
    whitecap_fraction,
)

# Koepke 1984: a whitecap's effective reflectance in the visible, averaged over its
# fresh and decaying foam.
WHITECAP_REFLECTANCE = 0.22

# Quan & Fry 1995 at 550 nm, salinity 35, 15 C.
SEAWATER_IOR = 1.341

# Morel & Maritorena 2001 R(0-) at 0.2 mg/m^3 chlorophyll, CIE 1931 under D65 to
# linear sRGB, times t / n^2 (Lee et al. 2002): Principled dims its diffuse by the
# specular toward the viewer only.
WATER_BODY_COLOR = (0.0, 0.0065, 0.018)

# The Sky Texture's default sun_size, a diameter. A facet tilted by d turns the
# reflection by 2 d, so the sun spans half its radius in slope.
SUN_SLOPE_RADIUS = math.radians(0.545) / 4

# A judgement: the glitter's tilt fades out as d^2 / v runs 9 to 16, d the tilt off the
# drawn normal of the facet that mirrors the sun, v the unresolved variance; the
# Gaussian slope density there is exp(-d^2 / v) of its peak.
GLINT_REACH = (9.0, 16.0)

# Past 6 sigma the normal CDF is within 1e-9 of 0 or 1.
CDF_SIGMAS = 6.0

# A judgement: the footprints a sea pixel spans.
FOOTPRINT_RANGE_M = (1e-3, 1e4)

# A judgement: emissivity and the reflected sky move slowly with slope.
SLOPE_ROWS = 16

# Elevation (rad, ascending) and the IR sky's radiance there.
type SkyCurve = tuple[np.ndarray, np.ndarray]


class Light(NamedTuple):
    """What the sea takes from above besides the world: the ir sky round the horizon,
    or the unit vector an eo sea glints towards."""

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


def _wave_group() -> bpy.types.NodeTree:
    """One wave, as far as the pixel resolves it along the wave's own direction:
    footprint^2 = across^2 + (along^2 - across^2) cos^2. It adds its slope to `Gradient`
    and its downward acceleration, in g, to `Acceleration`. One group shared by every
    wave: each node in a tree slows every edit to it."""
    group = bpy.data.node_groups.new("wave", "ShaderNodeTree")
    inputs = (
        ("Position", "Vector"),
        ("Along", "Vector"),
        ("Across Sq", "Float"),
        ("Stretch", "Float"),
        ("Gradient", "Vector"),
        ("Acceleration", "Float"),
        ("Calm", "Float"),
        ("Wavenumber", "Vector"),
        ("Phase", "Float"),
        ("Toward", "Vector"),
        ("Gone Sq", "Float"),
        ("Whole Sq", "Float"),
        ("Slope", "Vector"),
        ("Lift", "Float"),
    )
    sockets = {
        name: group.interface.new_socket(
            name, in_out="INPUT", socket_type=f"NodeSocket{kind}"
        )
        for name, kind in inputs
    }
    sockets["Calm"].default_value = 1.0
    for name, kind in (("Gradient", "Vector"), ("Acceleration", "Float")):
        group.interface.new_socket(
            name, in_out="OUTPUT", socket_type=f"NodeSocket{kind}"
        )
    given = group.nodes.new("NodeGroupInput").outputs
    out = group.nodes.new("NodeGroupOutput").inputs
    link = group.links.new
    phase = _math(
        group,
        "ADD",
        _vector(group, "DOT_PRODUCT", given["Position"], given["Wavenumber"]),
        given["Phase"],
    )
    heading = _vector(group, "DOT_PRODUCT", given["Along"], given["Toward"])
    footprint_sq = _math(
        group,
        "MULTIPLY_ADD",
        _math(group, "MULTIPLY", heading, heading),
        given["Stretch"],
        given["Across Sq"],
    )

    def shown(trig: str) -> bpy.types.NodeSocket:
        # From Min above From Max: a wider footprint fades the wave out.
        fade = group.nodes.new("ShaderNodeMapRange")
        fade.interpolation_type = "SMOOTHSTEP"
        link(given["Gone Sq"], fade.inputs["From Min"])
        link(given["Whole Sq"], fade.inputs["From Max"])
        link(footprint_sq, fade.inputs["Value"])
        link(_math(group, trig, phase), fade.inputs["To Max"])
        return _math(group, "MULTIPLY", fade.outputs["Result"], given["Calm"])

    # -d height / dx of a cos(phase) is a k_x sin(phase).
    link(
        _vector(
            group, "MULTIPLY_ADD", given["Slope"], shown("SINE"), given["Gradient"]
        ),
        out["Gradient"],
    )
    link(
        _math(
            group, "MULTIPLY_ADD", shown("COSINE"), given["Lift"], given["Acceleration"]
        ),
        out["Acceleration"],
    )
    return group


def _waves(
    tree: bpy.types.NodeTree,
    field: tuple[Wave, ...],
    time_s: bpy.types.NodeSocket,
    pixel: _Pixel,
    calm: bpy.types.NodeSocket | None = None,
    damped: frozenset[Wave] = frozenset(),
) -> tuple[bpy.types.NodeSocket, list[bpy.types.Node]]:
    """The normal of what the pixel resolves, and each wave's node; `calm` scales the
    `damped` ones."""
    geometry = tree.nodes.new("ShaderNodeNewGeometry")
    position = tree.nodes.new("ShaderNodeSeparateXYZ")
    # Height is left out, so the sea curving under the field cannot slide it.
    xyt = tree.nodes.new("ShaderNodeCombineXYZ")
    link = tree.links.new
    link(geometry.outputs["Position"], position.inputs["Vector"])
    link(position.outputs["X"], xyt.inputs["X"])
    link(position.outputs["Y"], xyt.inputs["Y"])
    link(time_s, xyt.inputs["Z"])
    across_sq = _math(tree, "MULTIPLY", pixel.across_m, pixel.across_m)
    along_sq = _math(tree, "MULTIPLY", pixel.along_m, pixel.along_m)
    stretch = _math(tree, "SUBTRACT", along_sq, across_sq)

    group = _wave_group()
    gradient = None
    drawn = []
    for i, wave in enumerate(field):
        node = tree.nodes.new("ShaderNodeGroup")
        node.node_tree = group
        node.name = f"wave_{i}"
        inputs = node.inputs
        link(xyt.outputs["Vector"], inputs["Position"])
        link(pixel.along_dir, inputs["Along"])
        link(across_sq, inputs["Across Sq"])
        link(stretch, inputs["Stretch"])
        if gradient is not None:
            link(gradient, inputs["Gradient"])
        if calm is not None and wave in damped:
            link(calm, inputs["Calm"])
        gone, whole = fade_footprints_m(2 * math.pi / wave.k_rad_m)
        inputs["Wavenumber"].default_value = (
            wave.k_east_rad_m,
            wave.k_north_rad_m,
            -wave.omega_rad_s,
        )
        inputs["Phase"].default_value = wave.phase_rad
        inputs["Toward"].default_value = (
            math.sin(wave.toward_rad),
            math.cos(wave.toward_rad),
            0.0,
        )
        inputs["Gone Sq"].default_value = float(gone) ** 2
        inputs["Whole Sq"].default_value = float(whole) ** 2
        inputs["Slope"].default_value = (
            wave.amplitude_m * wave.k_east_rad_m,
            wave.amplitude_m * wave.k_north_rad_m,
            0.0,
        )
        inputs["Lift"].default_value = wave.amplitude_m * wave.k_rad_m
        gradient = node.outputs["Gradient"]
        drawn.append(node)
    tilted = geometry.outputs["Normal"]
    if gradient is not None:
        tilted = _vector(tree, "ADD", tilted, gradient)
    return _vector(tree, "NORMALIZE", tilted, name="wave_normal"), drawn


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
    wind: tuple[Wave, ...],
    drawn: list[bpy.types.Node],
    footprint_m: bpy.types.NodeSocket,
    threshold_g: float | bpy.types.NodeSocket,
) -> bpy.types.NodeSocket:
    """How much of the pixel whitecaps, as `waves.whitecap_cover`, past
    `threshold_g`, from the `drawn` wind waves' acceleration."""
    acceleration: float | bpy.types.NodeSocket = 0.0
    for _, node in zip(wind, drawn, strict=True):
        if isinstance(acceleration, bpy.types.NodeSocket):
            tree.links.new(acceleration, node.inputs["Acceleration"])
        acceleration = node.outputs["Acceleration"]
    table = _footprint_table(
        "sea_unresolved_acceleration",
        lambda f: unresolved_acceleration_variance(wind, f),
    )
    sigma = _math(
        tree,
        "MAXIMUM",
        _math(
            tree,
            "SQRT",
            _at_footprint(tree, table, footprint_m, "sea_unresolved_acceleration"),
        ),
        1e-6,
    )
    excess = _math(
        tree,
        "SUBTRACT",
        acceleration,
        threshold_g,
        name="whitecap_excess",
    )
    # Clamped, so an infinite threshold reads the table's end.
    x = _math(
        tree,
        "MULTIPLY_ADD",
        _math(tree, "DIVIDE", excess, sigma),
        1 / (2 * CDF_SIGMAS),
        0.5,
    )
    x.node.use_clamp = True
    sigmas = CDF_SIGMAS * (2 * (np.arange(CURVE_SAMPLES) + 0.5) / CURVE_SAMPLES - 1)
    cdf = curve_image("normal_cdf", np.vectorize(NormalDist().cdf)(sigmas))
    cover = lookup(tree, cdf, x)
    cover.node.name = "whitecaps"
    return cover


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
    `sun` follows the world's when the world carries one, as the Sky Texture does; a
    photo's is baked at build.
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
    # 2 (1 - cos d) is d^2 to fourth order.
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
    time_s: bpy.types.NodeSocket,
    pixel: _Pixel,
    unresolved: bpy.types.NodeSocket,
    glinting: bpy.types.NodeSocket,
) -> tuple[bpy.types.NodeSocket, bpy.types.NodeSocket]:
    """A Gaussian tilt per cell of one specular point, re-drawn as it twinkles, and the
    slope variance the tilts carry out of the lobe.

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
    geometry = tree.nodes.new("ShaderNodeNewGeometry")
    scaled = _vector(tree, "SCALE", geometry.outputs["Position"], name="glitter_cells")
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
    """GGX roughness and aspect, alpha across / alpha along, for the unresolved slope
    variance at the two footprints. Per axis alpha = sqrt(2) sigma_axis =
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
    whitecaps: bpy.types.NodeSocket,
    wake: "_Wakes | None" = None,
) -> bpy.types.NodeSocket:
    """Water refracting at seawater's IOR, white where its crests break and where a
    hull leaves foam."""
    roughness, aspect = _lobe(tree, *unresolved)
    principled = tree.nodes.new("ShaderNodeBsdfPrincipled")
    principled.inputs["Base Color"].default_value = (*WATER_BODY_COLOR, 1.0)
    if wake is not None:
        body = tree.nodes.new("ShaderNodeVectorMath")
        body.operation = "SCALE"
        body.inputs["Vector"].default_value = WATER_BODY_COLOR
        tree.links.new(_math(tree, "ADD", wake.bubbles, 1.0), body.inputs["Scale"])
        tree.links.new(body.outputs["Vector"], principled.inputs["Base Color"])
    principled.inputs["IOR"].default_value = SEAWATER_IOR
    link = tree.links.new
    link(normal, principled.inputs["Normal"])
    link(tangent, principled.inputs["Tangent"])
    link(roughness, principled.inputs["Roughness"])
    # Principled's alpha_y / alpha_x = 1 - 0.9 a, so a = (1 - aspect) / 0.9.
    link(
        _math(tree, "MULTIPLY", _math(tree, "SUBTRACT", 1.0, aspect), 1 / 0.9),
        principled.inputs["Anisotropic"],
    )
    foam = tree.nodes.new("ShaderNodeBsdfDiffuse")
    foam.inputs["Color"].default_value = (*(WHITECAP_REFLECTANCE,) * 3, 1.0)
    mix = tree.nodes.new("ShaderNodeMixShader")
    link(whitecaps, mix.inputs["Factor"])
    # Mix Shader names both shader inputs "Shader", so they can only be indexed.
    link(principled.outputs["BSDF"], mix.inputs[1])
    link(foam.outputs["BSDF"], mix.inputs[2])
    if wake is None:
        return mix.outputs["Shader"]
    fresh = tree.nodes.new("ShaderNodeBsdfDiffuse")
    fresh.inputs["Color"].default_value = (*(FRESH_FOAM_REFLECTANCE,) * 3, 1.0)
    churned = tree.nodes.new("ShaderNodeMixShader")
    link(wake.foam, churned.inputs["Factor"])
    link(mix.outputs["Shader"], churned.inputs[1])
    link(fresh.outputs["BSDF"], churned.inputs[2])
    return churned.outputs["Shader"]


def _drifting(
    tree: bpy.types.NodeTree,
    time_s: bpy.types.NodeSocket,
    outputs: Outputs,
    tile: bpy.types.Image,
    axes: tuple[tuple[float, float], tuple[float, float]],
    velocity_mps: tuple[float, float],
    start: float,
    name: str,
) -> bpy.types.NodeSocket:
    """`tile` laid on the sea along `axes`, each an (east, north) tile axis in tiles
    per metre, offset `start` tiles, frozen and carried at `velocity_mps`, (east,
    north).

    A loop crossfades two layers, each carried for one span and back while its weight
    is 0 (Vlachos, "Water flow in Portal 2", SIGGRAPH 2010 course), over the root of
    their squared weights so the variance holds.
    """
    geometry = tree.nodes.new("ShaderNodeNewGeometry")
    here = tree.nodes.new("ShaderNodeCombineXYZ")
    here.name = f"{name}_axes"
    for axis, out in zip(axes, ("X", "Y"), strict=True):
        along = _vector(tree, "DOT_PRODUCT", geometry.outputs["Position"])
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
    time_s: bpy.types.NodeSocket,
    outputs: Outputs,
    tile: bpy.types.Image,
) -> bpy.types.NodeSocket:
    """The wind over its mean, less one, carried downwind at the mean wind."""
    per_m = 1 / GUST_TILE_M
    east, north = _downwind(sea)
    speed = sea.wind_speed_mps
    unit = _drifting(
        tree,
        time_s,
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
    time_s: bpy.types.NodeSocket,
    outputs: Outputs,
    rng: np.random.Generator,
) -> bpy.types.NodeSocket:
    """1 under a slick: the top `slick_cover` of a tile at the windrow spacing,
    stretched along the wind and drifting with it.

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
        time_s,
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
        return breaking_threshold_g(wind, whitecap_fraction(speed))
    texel = (np.arange(CURVE_SAMPLES) + 0.5) / CURVE_SAMPLES
    thresholds = [
        breaking_threshold_g(wind, gusty_whitecap_fraction(speed, g))
        for g in gust_max * (2 * texel - 1)
    ]
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


def _path(tree: bpy.types.NodeTree, wake: Wake, time_s: bpy.types.NodeSocket) -> _Path:
    position = tree.nodes.new("ShaderNodeSeparateXYZ")
    geometry = tree.nodes.new("ShaderNodeNewGeometry")
    tree.links.new(geometry.outputs["Position"], position.inputs["Vector"])
    east, north = position.outputs["X"], position.outputs["Y"]
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
    time_s: bpy.types.NodeSocket,
    footprint_m: bpy.types.NodeSocket,
    foam_rng: np.random.Generator,
    sea_slope_variance: float,
) -> _Wakes:
    """Every hull's turbulent band, stern foam and Kelvin arms: the strongest band and
    foam, and the arms summed.

    ponytail: the arms are Kelvin's straight-path pattern laid along the path, which a
    tight turn bends; sum the arms from past poses if a turn reads wrong.
    """
    tile = von_karman_field(foam_rng, FOAM_CELLS, FOAM_SPACING_M, FOAM_LENGTH_M)
    noise = curve_image("sea_foam", tile)
    levels = (np.arange(CURVE_SAMPLES) + 0.5) / CURVE_SAMPLES
    ppf = curve_image("normal_ppf", np.vectorize(NormalDist().inv_cdf)(1 - levels))
    band: bpy.types.NodeSocket | float = 0.0
    foam: bpy.types.NodeSocket | float = 0.0
    bubbles: bpy.types.NodeSocket | float = 0.0
    tilt = None
    for wake in wakes:
        path = _path(tree, wake, time_s)
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
    gusty sea's, and a slick's where `smooth` is 1."""
    speed = sea.wind_speed_mps

    def unresolved_at(footprint_m: float) -> float:
        return unresolved_slope_variance(speed, wind, swell, footprint_m)

    table = _footprint_table("sea_unresolved_variance", unresolved_at)
    footprints = (pixel.along_m, pixel.across_m)
    unresolved = tuple(
        _at_footprint(tree, table, f, "sea_unresolved_variance") for f in footprints
    )
    if gust is not None:
        unresolved = _gusty(tree, sea, gust, unresolved)
    if smooth is None:
        return unresolved

    def slick_at(footprint_m: float) -> float:
        return unresolved_slope_variance(
            speed, survivors, swell, footprint_m, cox_munk_slick_slope
        )

    name = "sea_unresolved_slick_variance"
    slick_table = _footprint_table(name, slick_at)
    # The slick's own variance in place of the gusty sea's.
    return tuple(
        _math(
            tree,
            "MULTIPLY_ADD",
            smooth,
            _math(tree, "SUBTRACT", _at_footprint(tree, slick_table, f, name), v),
            v,
        )
        for v, f in zip(unresolved, footprints, strict=True)
    )


def material(
    sea: Sea,
    wind: tuple[Wave, ...],
    swell: tuple[Wave, ...],
    outputs: Outputs,
    rngs: tuple[np.random.Generator, np.random.Generator, np.random.Generator],
    wakes: tuple[Wake, ...],
    light: Light,
) -> bpy.types.Material:
    """Each pixel draws the waves it resolves and takes the rest as roughness;
    `rngs` draw its gusts, its slicks and its wakes' foam. With a sky the sea is ir and
    reflects it; without, eo, glittering towards the light's sun if it has one."""
    gust_rng, slick_rng, foam_rng = rngs
    material = bpy.data.materials.new("sea")
    tree = material.node_tree
    tree.nodes.clear()
    field = wind + swell
    pixel = _pixel(tree)
    time_s = _sea_time(tree, outputs)
    speed = sea.wind_speed_mps
    gust_max, gust, slick = 0.0, None, None
    if speed > 0.0:
        tile = von_karman_field(gust_rng, GUST_CELLS, GUST_SPACING_M, GUST_LENGTH_M)
        # A crossfade reaches sqrt(2) of the tile's peak at most.
        gust_max = math.sqrt(2) * np.abs(tile).max() * turbulence_intensity(speed)
        gust = _gust(tree, sea, time_s, outputs, curve_image("sea_gust", tile))
        if sea.slick_cover > 0.0:
            slick = _slick(tree, sea, time_s, outputs, slick_rng)
    wake = None
    if wakes:
        sea_variance = cox_munk_slope(speed) ** 2
        wake = _wakes(tree, wakes, time_s, pixel.along_m, foam_rng, sea_variance)
    smooth = _smoothed(tree, slick, wake)
    survivors = wind if slick is None else slick_survivors(speed, wind)
    damped = frozenset(wind) - frozenset(survivors)
    calm = None
    if slick is not None and damped:
        calm = _math(tree, "SUBTRACT", 1.0, slick)
    normal, drawn = _waves(tree, field, time_s, pixel, calm, damped)
    if wake is not None and wake.tilt is not None:
        normal = _vector(
            tree,
            "NORMALIZE",
            _vector(tree, "ADD", normal, wake.tilt),
            name="wake_normal",
        )
    unresolved = _unresolved(tree, sea, wind, swell, survivors, pixel, gust, smooth)

    def unresolved_at(footprint_m: float) -> float:
        return unresolved_slope_variance(speed, wind, swell, footprint_m)

    if light.sky is None:
        carried: float | bpy.types.NodeSocket = 0.0
        if wind and light.sun is not None:
            # The IR sky has no sun to glint, and its emissivity takes the whole slope.
            # Along the view, the wider spread, reaches the most facets.
            glinting = _glinting(tree, normal, light.sun, unresolved[0])
            tilt, carried = _glitter(tree, wind, time_s, pixel, unresolved[1], glinting)
            normal = _vector(
                tree,
                "NORMALIZE",
                _vector(tree, "ADD", normal, tilt),
                name="glitter_normal",
            )
        along = _math(
            tree, "MAXIMUM", _math(tree, "SUBTRACT", unresolved[0], carried), 0.0
        )
        # Principled stretches a lobe 10:1 at most, alpha 10 to 1, variance 100.
        across = _math(
            tree,
            "MAXIMUM",
            _math(tree, "SUBTRACT", unresolved[1], carried),
            _math(tree, "MULTIPLY", along, 0.01),
        )
        unresolved = (along, across)
        threshold = _breaking(tree, sea, wind, gust, gust_max)
        # Along the view, the widest footprint, leaves the most unresolved. A judgement.
        whitecaps = _whitecaps(tree, wind, drawn[: len(wind)], pixel.along_m, threshold)
        surface = _daylight(
            tree, sea, normal, pixel.along_dir, unresolved, whitecaps, wake
        )
    else:
        # The coarsest footprint leaves the most, Cox & Munk's and all the swell's,
        # and the strongest gust adds the most.
        slope_max = math.sqrt(
            unresolved_at(FOOTPRINT_RANGE_M[1])
            + gust_max * gust_slope_variance(sea.wind_speed_mps)
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
