"""The sea's mesh and materials: waves as shading on a grid curved to the earth."""

import math
from collections.abc import Callable
from statistics import NormalDist
from typing import NamedTuple

import bpy
import numpy as np

from seascape import lwir
from seascape.blend import CURVE_SAMPLES, animate, curve_image, place
from seascape.config import Band, Outputs, Rig, Sea
from seascape.waves import (
    FADE_FOOTPRINTS,
    Wave,
    breaking_threshold_g,
    earth_radius_m,
    horizon_m,
    pixel_acceleration_variance,
    pixel_slope_variance,
    sea_z_m,
    wave_slope,
    whitecap_fraction,
)

# Koepke 1984: a whitecap's effective reflectance in the visible, averaged over its
# fresh and decaying foam.
WHITECAP_REFLECTANCE = 0.22

# Past 6 sigma the normal CDF is within 1e-9 of 0 or 1.
CDF_SIGMAS = 6.0

# A judgement: the footprints a sea pixel spans.
FOOTPRINT_RANGE_M = (1e-3, 1e4)

# A judgement: emissivity moves slowly with slope.
EMISSIVITY_ROWS = 16

# Cells per side: enough for the tangent point to land on a face, not an accuracy
# knob. A cell's sagitta, width^2 / 8R, is far under a pixel at the horizon.
SEA_CELLS = 128

# Margin on the horizon, or the grid's own edge becomes the horizon.
SEA_MARGIN = 1.5


def sea_reach_m(rig: Rig, sea: Sea) -> float:
    """Half-width of the sea, a margin past the horizon."""
    return SEA_MARGIN * horizon_m(rig.height_m, sea.refraction_k)


def _emissivity_image(t_sea_k: float, sigma_max: float) -> bpy.types.Image:
    """`lwir.emissivity_curve` baked against cos(theta) along a row and the unresolved
    slope per axis up the rows, both at texel centres, which is what the shader
    samples.

    The curve is sampled uniformly in angle; the shader's dot product is uniform in its
    cosine, so it is resampled here rather than corrected in nodes.
    """
    mu = (np.arange(CURVE_SAMPLES) + 0.5) / CURVE_SAMPLES
    rows = []
    for j in range(EMISSIVITY_ROWS):
        sigma = (j + 0.5) / EMISSIVITY_ROWS * sigma_max
        theta, eps = lwir.emissivity_curve(t_sea_k=t_sea_k, slope_sigma=sigma)
        rows.append(np.interp(mu, np.cos(theta)[::-1], eps[::-1]))
    return curve_image("sea_emissivity", np.array(rows))


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
    tree: bpy.types.NodeTree, operation: str, *inputs: bpy.types.NodeSocket
) -> bpy.types.NodeSocket:
    node = tree.nodes.new("ShaderNodeVectorMath")
    node.operation = operation
    for socket, value in zip(node.inputs, inputs, strict=False):
        tree.links.new(value, socket)
    return node.outputs["Value" if operation == "DOT_PRODUCT" else "Vector"]


def _math(
    tree: bpy.types.NodeTree, operation: str, *inputs: float | bpy.types.NodeSocket
) -> bpy.types.NodeSocket:
    """A Math node. Unset inputs read 0.5."""
    node = tree.nodes.new("ShaderNodeMath")
    node.operation = operation
    for socket, value in zip(node.inputs, inputs, strict=False):
        if isinstance(value, bpy.types.NodeSocket):
            tree.links.new(value, socket)
        else:
            socket.default_value = value
    return node.outputs["Value"]


class _Pixel(NamedTuple):
    """A pixel's footprint on the sea; `along` is its unit direction."""

    across_m: bpy.types.NodeSocket
    along_m: bpy.types.NodeSocket
    along: bpy.types.NodeSocket


def _pixel(tree: bpy.types.NodeTree, pixel_rad: float) -> _Pixel:
    camera = tree.nodes.new("ShaderNodeCameraData")
    geometry = tree.nodes.new("ShaderNodeNewGeometry")
    facing = _vector(
        tree, "DOT_PRODUCT", geometry.outputs["Incoming"], geometry.outputs["Normal"]
    )
    across = _math(tree, "MULTIPLY", camera.outputs["View Distance"], pixel_rad)
    # Grazing stretches the pixel by 1 / cos along the view; the floor keeps it finite.
    cosine = _math(tree, "MAXIMUM", _math(tree, "ABSOLUTE", facing), 1e-3)
    tilt = _vector(tree, "SCALE", geometry.outputs["Normal"])
    tree.links.new(facing, tilt.node.inputs["Scale"])
    level = _vector(tree, "SUBTRACT", geometry.outputs["Incoming"], tilt)
    along = _vector(tree, "NORMALIZE", level)
    return _Pixel(across, _math(tree, "DIVIDE", across, cosine), along)


class _Drawn(NamedTuple):
    phase: bpy.types.NodeSocket
    footprint_sq: bpy.types.NodeSocket


def _wave(
    tree: bpy.types.NodeTree,
    i: int,
    wave: Wave,
    xyt: bpy.types.NodeSocket,
    pixel: _Pixel,
    stretch: bpy.types.NodeSocket,
    gradient: bpy.types.NodeSocket | None,
) -> tuple[bpy.types.NodeSocket, _Drawn]:
    """One wave's slope onto `gradient`, faded by how well the pixel resolves it along
    the wave's own direction: footprint^2 = across^2 + (along^2 - across^2) cos^2."""
    link = tree.links.new
    dot = tree.nodes.new("ShaderNodeVectorMath")
    dot.operation = "DOT_PRODUCT"
    dot.name = f"wave_{i}"
    dot.inputs["Vector_001"].default_value = (
        wave.k_east_rad_m,
        wave.k_north_rad_m,
        -wave.omega_rad_s,
    )
    link(xyt, dot.inputs["Vector"])
    phase = tree.nodes.new("ShaderNodeMath")
    phase.operation = "ADD"
    phase.name = f"wave_{i}_phase"
    phase.inputs["Value_001"].default_value = wave.phase_rad
    link(dot.outputs["Value"], phase.inputs["Value"])
    heading = _vector(tree, "DOT_PRODUCT", pixel.along)
    heading.node.inputs["Vector_001"].default_value = (
        math.sin(wave.toward_rad),
        math.cos(wave.toward_rad),
        0.0,
    )
    across_sq = _math(tree, "MULTIPLY", pixel.across_m, pixel.across_m)
    cos_sq = _math(
        tree, "MULTIPLY", heading.node.outputs["Value"], heading.node.outputs["Value"]
    )
    footprint_sq = _math(tree, "MULTIPLY_ADD", cos_sq, stretch, across_sq)
    fade = _fade(tree, wave, footprint_sq, _math(tree, "SINE", phase.outputs["Value"]))
    fade.node.name = f"wave_{i}_fade"
    # -d height / dx of a cos(phase) is a k_x sin(phase).
    term = tree.nodes.new("ShaderNodeVectorMath")
    term.operation = "MULTIPLY_ADD"
    term.name = f"wave_{i}_slope"
    term.inputs["Vector"].default_value = (
        wave.amplitude_m * wave.k_east_rad_m,
        wave.amplitude_m * wave.k_north_rad_m,
        0.0,
    )
    link(fade, term.inputs["Vector_001"])
    if gradient is not None:
        link(gradient, term.inputs["Vector_002"])
    return term.outputs["Vector"], _Drawn(phase.outputs["Value"], footprint_sq)


def _fade(
    tree: bpy.types.NodeTree,
    wave: Wave,
    footprint_sq: bpy.types.NodeSocket,
    whole: bpy.types.NodeSocket,
) -> bpy.types.NodeSocket:
    """`whole` where the pixel resolves `wave`, 0 where it cannot, as
    `waves.visibility`."""
    # From Min above From Max: a wider footprint fades the wave out.
    fade = tree.nodes.new("ShaderNodeMapRange")
    fade.interpolation_type = "SMOOTHSTEP"
    wavelength_m = 2 * math.pi / wave.k_rad_m
    low, high = FADE_FOOTPRINTS
    fade.inputs["From Min"].default_value = (wavelength_m / low) ** 2
    fade.inputs["From Max"].default_value = (wavelength_m / high) ** 2
    fade.inputs["To Min"].default_value = 0.0
    tree.links.new(footprint_sq, fade.inputs["Value"])
    tree.links.new(whole, fade.inputs["To Max"])
    return fade.outputs["Result"]


def _by_footprint(
    tree: bpy.types.NodeTree,
    name: str,
    value_at: Callable[[float], float],
    footprint: bpy.types.NodeSocket,
) -> bpy.types.NodeSocket:
    """`value_at(footprint)`, baked against log footprint: one lookup, not a sum over
    the waves per sample."""
    low, high = FOOTPRINT_RANGE_M
    decades = math.log10(high / low)
    footprints = low * 10 ** (
        decades * (np.arange(CURVE_SAMPLES) + 0.5) / CURVE_SAMPLES
    )
    lookup = tree.nodes.new("ShaderNodeCombineXYZ")
    texture = tree.nodes.new("ShaderNodeTexImage")
    texture.name = name
    texture.image = curve_image(name, np.array([value_at(f) for f in footprints]))
    texture.extension = "EXTEND"
    x = _math(
        tree,
        "MULTIPLY_ADD",
        _math(tree, "LOGARITHM", footprint, 10.0),
        1 / decades,
        -math.log10(low) / decades,
    )
    tree.links.new(x, lookup.inputs["X"])
    tree.links.new(lookup.outputs["Vector"], texture.inputs["Vector"])
    return texture.outputs["Color"]


class _Slopes(NamedTuple):
    """What a pixel resolves, as a normal, and the slope variance it leaves, total over
    both axes, at its footprint along the view and across it."""

    normal: bpy.types.NodeSocket
    along: bpy.types.NodeSocket
    across: bpy.types.NodeSocket
    tangent: bpy.types.NodeSocket
    footprint_m: bpy.types.NodeSocket  # along the view, the pixel's longest
    drawn: list[_Drawn]


def _wave_normals(
    tree: bpy.types.NodeTree,
    sea: Sea,
    field: tuple[Wave, ...],
    outputs: Outputs,
    pixel_rad: float,
) -> _Slopes:
    link = tree.links.new
    geometry = tree.nodes.new("ShaderNodeNewGeometry")
    position = tree.nodes.new("ShaderNodeSeparateXYZ")
    # Height is left out, so the sea curving under the field cannot slide it.
    xyt = tree.nodes.new("ShaderNodeCombineXYZ")
    link(geometry.outputs["Position"], position.inputs["Vector"])
    link(position.outputs["X"], xyt.inputs["X"])
    link(position.outputs["Y"], xyt.inputs["Y"])
    link(_sea_time(tree, outputs), xyt.inputs["Z"])
    pixel = _pixel(tree, pixel_rad)
    along_sq = _math(tree, "MULTIPLY", pixel.along_m, pixel.along_m)
    across_sq = _math(tree, "MULTIPLY", pixel.across_m, pixel.across_m)
    stretch = _math(tree, "SUBTRACT", along_sq, across_sq)

    relief = tree.nodes.new("ShaderNodeVectorMath")
    relief.operation = "SCALE"
    relief.name = "wave_relief"
    relief.inputs["Scale"].default_value = 1.0
    gradient = None
    drawn = []
    for i, wave in enumerate(field):
        gradient, one = _wave(
            tree, i, wave, xyt.outputs["Vector"], pixel, stretch, gradient
        )
        drawn.append(one)
    if gradient is not None:
        link(gradient, relief.inputs["Vector"])

    tilted = tree.nodes.new("ShaderNodeVectorMath")
    tilted.operation = "ADD"
    normal = tree.nodes.new("ShaderNodeVectorMath")
    normal.operation = "NORMALIZE"
    normal.name = "wave_normal"
    link(geometry.outputs["Normal"], tilted.inputs["Vector"])
    link(relief.outputs["Vector"], tilted.inputs["Vector_001"])
    link(tilted.outputs["Vector"], normal.inputs["Vector"])

    def unresolved(footprint_m: float) -> float:
        return pixel_slope_variance(sea.wind_speed_mps, field, footprint_m)

    return _Slopes(
        normal.outputs["Vector"],
        _by_footprint(tree, "sea_unresolved_variance", unresolved, pixel.along_m),
        _by_footprint(tree, "sea_unresolved_variance", unresolved, pixel.across_m),
        pixel.along,
        pixel.along_m,
        drawn,
    )


def _whitecaps(
    tree: bpy.types.NodeTree, wind: tuple[Wave, ...], slopes: _Slopes, fraction: float
) -> bpy.types.NodeSocket:
    """How much of the pixel whitecaps, as `waves.whitecap_cover`, the wind's waves
    being the first of `slopes.drawn`."""
    acceleration: float | bpy.types.NodeSocket = 0.0
    for wave, drawn in zip(wind, slopes.drawn, strict=False):
        shown = _fade(
            tree, wave, drawn.footprint_sq, _math(tree, "COSINE", drawn.phase)
        )
        acceleration = _math(
            tree, "MULTIPLY_ADD", shown, wave.amplitude_m * wave.k_rad_m, acceleration
        )
    # Along the view: the widest footprint, so the most left unresolved. A judgement.
    variance = _by_footprint(
        tree,
        "sea_unresolved_acceleration",
        lambda f: pixel_acceleration_variance(wind, f),
        slopes.footprint_m,
    )
    sigma = _math(tree, "MAXIMUM", _math(tree, "SQRT", variance), 1e-6)
    threshold = _math(
        tree, "SUBTRACT", acceleration, breaking_threshold_g(wind, fraction)
    )
    threshold.node.name = "whitecaps_threshold"
    z = _math(tree, "DIVIDE", threshold, sigma)
    # Clamped, so an infinite threshold reads the table's end.
    x = _math(tree, "MULTIPLY_ADD", z, 1 / (2 * CDF_SIGMAS), 0.5)
    x.node.use_clamp = True
    sigmas = CDF_SIGMAS * (2 * (np.arange(CURVE_SAMPLES) + 0.5) / CURVE_SAMPLES - 1)
    lookup = tree.nodes.new("ShaderNodeCombineXYZ")
    texture = tree.nodes.new("ShaderNodeTexImage")
    texture.image = curve_image("normal_cdf", np.vectorize(NormalDist().cdf)(sigmas))
    texture.extension = "EXTEND"
    tree.links.new(x, lookup.inputs["X"])
    tree.links.new(lookup.outputs["Vector"], texture.inputs["Vector"])
    cover = texture.outputs["Color"]
    texture.name = "whitecaps"
    return cover


class _Lobe(NamedTuple):
    roughness: bpy.types.NodeSocket
    aspect: bpy.types.NodeSocket  # alpha across / alpha along, at most 1


def _lobe(tree: bpy.types.NodeTree, slopes: _Slopes) -> _Lobe:
    """GGX widths for the unresolved slope, per axis alpha = sqrt(2) sigma_axis =
    sqrt(variance): Blender's roughness is their geometric mean's square root."""
    along = _math(tree, "MAXIMUM", slopes.along, 1e-12)
    across = _math(tree, "MAXIMUM", slopes.across, 1e-12)
    product = _math(tree, "MULTIPLY", along, across)
    roughness = _math(tree, "MINIMUM", _math(tree, "POWER", product, 0.125), 1.0)
    roughness.node.name = "sea_roughness"
    aspect = _math(tree, "SQRT", _math(tree, "DIVIDE", across, along))
    aspect.node.name = "sea_aspect"
    return _Lobe(roughness, aspect)


def _incidence_lookup(
    tree: bpy.types.NodeTree,
    table: bpy.types.Image,
    normal: bpy.types.NodeSocket,
    slope_fraction: bpy.types.NodeSocket,
) -> bpy.types.NodeSocket:
    """Sample `table` at |cos(theta)| between the wave normal and the viewing ray, and
    at the pixel's unresolved slope as a fraction of the table's.

    Against the wave normal, not the plane's, or a flat sea's emissivity gets applied
    to water that is visibly not flat.
    """
    geometry = tree.nodes.new("ShaderNodeNewGeometry")
    dot = tree.nodes.new("ShaderNodeVectorMath")
    dot.operation = "DOT_PRODUCT"
    lookup = tree.nodes.new("ShaderNodeCombineXYZ")
    texture = tree.nodes.new("ShaderNodeTexImage")
    texture.image = table
    texture.extension = "EXTEND"

    link = tree.links.new
    link(geometry.outputs["Incoming"], dot.inputs["Vector"])
    link(normal, dot.inputs["Vector_001"])
    link(_math(tree, "ABSOLUTE", dot.outputs["Value"]), lookup.inputs["X"])
    link(slope_fraction, lookup.inputs["Y"])
    link(lookup.outputs["Vector"], texture.inputs["Vector"])
    return texture.outputs["Color"]


def _thermal_sea(
    sea: Sea,
    wind: tuple[Wave, ...],
    swell: tuple[Wave, ...],
    outputs: Outputs,
    pixel_rad: float,
) -> bpy.types.Material:
    """eps(theta) of the sea emitted, the remaining 1 - eps reflected from the sky.

    Complements, so the two very nearly cancel and the sea holds close to ambient at
    every angle.
    """
    material = bpy.data.materials.new("sea")
    tree = material.node_tree
    tree.nodes.clear()
    field = wind + swell
    slopes = _wave_normals(tree, sea, field, outputs, pixel_rad)
    lobe = _lobe(tree, slopes)
    mirror = tree.nodes.new("ShaderNodeBsdfAnisotropic")
    # Glossy BSDF ships at 0.8 grey. The Mix Shader already applies the 1 - eps
    # weighting, so anything but white here absorbs reflected sky and cuts a dark
    # notch along the horizon.
    mirror.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.inputs["Strength"].default_value = lwir.band_radiance(sea.t_sea_k)
    mix = tree.nodes.new("ShaderNodeMixShader")
    output = tree.nodes.new("ShaderNodeOutputMaterial")

    link = tree.links.new
    link(slopes.normal, mirror.inputs["Normal"])
    link(slopes.tangent, mirror.inputs["Tangent"])
    link(lobe.roughness, mirror.inputs["Roughness"])
    # Glossy's alpha_y / alpha_x = (1 + A)^2 for A < 0, so A = sqrt(aspect) - 1.
    link(
        _math(tree, "SUBTRACT", _math(tree, "SQRT", lobe.aspect), 1.0),
        mirror.inputs["Anisotropy"],
    )
    # The facets lwir averages over are per axis: half the pixel's total, averaged over
    # its two footprints.
    per_axis = _math(
        tree, "MULTIPLY", _math(tree, "ADD", slopes.along, slopes.across), 0.25
    )
    sigma_max = wave_slope(sea.wind_speed_mps) / math.sqrt(2)
    fraction = _math(tree, "DIVIDE", _math(tree, "SQRT", per_axis), sigma_max)
    # Mix Shader names both shader inputs "Shader", so they can only be indexed. Factor
    # is emissivity: 0 at grazing incidence takes the mirror, 1 head-on takes emission.
    link(mirror.outputs["BSDF"], mix.inputs[1])
    link(emission.outputs["Emission"], mix.inputs[2])
    link(
        _incidence_lookup(
            tree, _emissivity_image(sea.t_sea_k, sigma_max), slopes.normal, fraction
        ),
        mix.inputs["Factor"],
    )
    link(mix.outputs["Shader"], output.inputs["Surface"])
    return material


def _water_material(
    sea: Sea,
    wind: tuple[Wave, ...],
    swell: tuple[Wave, ...],
    outputs: Outputs,
    pixel_rad: float,
) -> bpy.types.Material:
    """Daylight water, refracting at seawater's IOR, white where its crests break."""
    field = wind + swell
    material = bpy.data.materials.new("sea")
    tree = material.node_tree
    principled = tree.nodes["Principled BSDF"]
    principled.inputs["Base Color"].default_value = (0.004, 0.02, 0.035, 1.0)
    principled.inputs["IOR"].default_value = 1.33
    slopes = _wave_normals(tree, sea, field, outputs, pixel_rad)
    lobe = _lobe(tree, slopes)
    link = tree.links.new
    link(slopes.normal, principled.inputs["Normal"])
    link(slopes.tangent, principled.inputs["Tangent"])
    link(lobe.roughness, principled.inputs["Roughness"])
    # Principled's alpha_y / alpha_x = 1 - 0.9 a, so a = (1 - aspect) / 0.9.
    link(
        _math(tree, "MULTIPLY", _math(tree, "SUBTRACT", 1.0, lobe.aspect), 1 / 0.9),
        principled.inputs["Anisotropic"],
    )
    foam = tree.nodes.new("ShaderNodeBsdfDiffuse")
    foam.inputs["Color"].default_value = (*(WHITECAP_REFLECTANCE,) * 3, 1.0)
    mix = tree.nodes.new("ShaderNodeMixShader")
    cover = _whitecaps(tree, wind, slopes, whitecap_fraction(sea.wind_speed_mps))
    link(cover, mix.inputs["Factor"])
    # Mix Shader names both shader inputs "Shader", so they can only be indexed.
    link(principled.outputs["BSDF"], mix.inputs[1])
    link(foam.outputs["BSDF"], mix.inputs[2])
    link(mix.outputs["Shader"], tree.nodes["Material Output"].inputs["Surface"])
    return material


def water(
    sea: Sea,
    wind: tuple[Wave, ...],
    swell: tuple[Wave, ...],
    reach_m: float,
    band: Band,
    outputs: Outputs,
    pixel_rad: float,
) -> bpy.types.Object:
    """A grid curved to the earth. The waves are in its material.

    z = -(x^2 + y^2) / 2R osculates the sphere, off by d^4 / 8R^3 at distance d.
    Geometry here and not for waves: the bulge is kilometres across, never sub-pixel.
    """
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
    material = _water_material if band == "eo" else _thermal_sea
    water.data.materials.append(material(sea, wind, swell, outputs, pixel_rad))
    return water
