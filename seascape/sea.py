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
from seascape.blend import CURVE_SAMPLES, animate, curve_image, lookup, place
from seascape.config import Band, Outputs, Rig, Sea
from seascape.waves import (
    Wave,
    breaking_threshold_g,
    earth_radius_m,
    fade_footprints_m,
    horizon_m,
    sea_z_m,
    unresolved_acceleration_variance,
    unresolved_slope_variance,
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


def _emissivity_image(t_sea_k: float, slope_max: float) -> bpy.types.Image:
    """`lwir.emissivity_curve` baked against cos(theta) along a row and the unresolved
    RMS slope up the rows, to `slope_max`, both at texel centres, which is what the
    shader samples.

    The curve is sampled uniformly in angle; the shader's dot product is uniform in its
    cosine, so it is resampled here rather than corrected in nodes.
    """
    mu = (np.arange(CURVE_SAMPLES) + 0.5) / CURVE_SAMPLES
    rows = []
    for j in range(EMISSIVITY_ROWS):
        # lwir takes the slope per axis, the total over sqrt(2).
        sigma = (j + 0.5) / EMISSIVITY_ROWS * slope_max / math.sqrt(2)
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
    tree: bpy.types.NodeTree,
    operation: str,
    *inputs: bpy.types.NodeSocket,
    name: str | None = None,
) -> bpy.types.NodeSocket:
    node = tree.nodes.new("ShaderNodeVectorMath")
    node.operation = operation
    if name:
        node.name = name
    for socket, value in zip(node.inputs, inputs, strict=False):
        tree.links.new(value, socket)
    return node.outputs["Value" if operation == "DOT_PRODUCT" else "Vector"]


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


def _fade(
    tree: bpy.types.NodeTree,
    wave: Wave,
    footprint_sq: bpy.types.NodeSocket,
    whole: bpy.types.NodeSocket,
) -> bpy.types.NodeSocket:
    """`whole` scaled by `waves.visibility`."""
    # From Min above From Max: a wider footprint fades the wave out.
    fade = tree.nodes.new("ShaderNodeMapRange")
    fade.interpolation_type = "SMOOTHSTEP"
    gone, whole_m = fade_footprints_m(2 * math.pi / wave.k_rad_m)
    fade.inputs["From Min"].default_value = float(gone) ** 2
    fade.inputs["From Max"].default_value = float(whole_m) ** 2
    tree.links.new(footprint_sq, fade.inputs["Value"])
    tree.links.new(whole, fade.inputs["To Max"])
    return fade.outputs["Result"]


def _wave(
    tree: bpy.types.NodeTree,
    i: int,
    wave: Wave,
    xyt: bpy.types.NodeSocket,
    pixel: _Pixel,
    across_sq: bpy.types.NodeSocket,
    stretch: bpy.types.NodeSocket,
    gradient: bpy.types.NodeSocket | None,
) -> tuple[bpy.types.NodeSocket, _Drawn]:
    """`gradient` plus this wave's slope, as far as the pixel resolves it along the
    wave's own direction: footprint^2 = across^2 + (along^2 - across^2) cos^2."""
    dot = _vector(tree, "DOT_PRODUCT", xyt, name=f"wave_{i}")
    dot.node.inputs["Vector_001"].default_value = (
        wave.k_east_rad_m,
        wave.k_north_rad_m,
        -wave.omega_rad_s,
    )
    phase = _math(tree, "ADD", dot, wave.phase_rad, name=f"wave_{i}_phase")
    heading = _vector(tree, "DOT_PRODUCT", pixel.along_dir)
    heading.node.inputs["Vector_001"].default_value = (
        math.sin(wave.toward_rad),
        math.cos(wave.toward_rad),
        0.0,
    )
    cos_sq = _math(tree, "MULTIPLY", heading, heading)
    footprint_sq = _math(tree, "MULTIPLY_ADD", cos_sq, stretch, across_sq)
    # -d height / dx of a cos(phase) is a k_x sin(phase).
    shown = _fade(tree, wave, footprint_sq, _math(tree, "SINE", phase))
    shown.node.name = f"wave_{i}_fade"
    term = _vector(tree, "MULTIPLY_ADD", name=f"wave_{i}_slope")
    term.node.inputs["Vector"].default_value = (
        wave.amplitude_m * wave.k_east_rad_m,
        wave.amplitude_m * wave.k_north_rad_m,
        0.0,
    )
    tree.links.new(shown, term.node.inputs["Vector_001"])
    if gradient is not None:
        tree.links.new(gradient, term.node.inputs["Vector_002"])
    return term, _Drawn(phase, footprint_sq)


def _waves(
    tree: bpy.types.NodeTree,
    field: tuple[Wave, ...],
    outputs: Outputs,
    pixel: _Pixel,
) -> tuple[bpy.types.NodeSocket, list[_Drawn]]:
    """The normal of what the pixel resolves, and each wave as drawn."""
    geometry = tree.nodes.new("ShaderNodeNewGeometry")
    position = tree.nodes.new("ShaderNodeSeparateXYZ")
    # Height is left out, so the sea curving under the field cannot slide it.
    xyt = tree.nodes.new("ShaderNodeCombineXYZ")
    link = tree.links.new
    link(geometry.outputs["Position"], position.inputs["Vector"])
    link(position.outputs["X"], xyt.inputs["X"])
    link(position.outputs["Y"], xyt.inputs["Y"])
    link(_sea_time(tree, outputs), xyt.inputs["Z"])
    across_sq = _math(tree, "MULTIPLY", pixel.across_m, pixel.across_m)
    along_sq = _math(tree, "MULTIPLY", pixel.along_m, pixel.along_m)
    stretch = _math(tree, "SUBTRACT", along_sq, across_sq)

    gradient = None
    drawn = []
    for i, wave in enumerate(field):
        gradient, one = _wave(
            tree, i, wave, xyt.outputs["Vector"], pixel, across_sq, stretch, gradient
        )
        drawn.append(one)
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
    drawn: list[_Drawn],
    footprint_m: bpy.types.NodeSocket,
    fraction: float,
) -> bpy.types.NodeSocket:
    """How much of the pixel whitecaps, as `waves.whitecap_cover`."""
    acceleration: float | bpy.types.NodeSocket = 0.0
    for wave, one in zip(wind, drawn, strict=True):
        shown = _fade(tree, wave, one.footprint_sq, _math(tree, "COSINE", one.phase))
        acceleration = _math(
            tree, "MULTIPLY_ADD", shown, wave.amplitude_m * wave.k_rad_m, acceleration
        )
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
        breaking_threshold_g(wind, fraction),
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
    tangent: bpy.types.NodeSocket,
    unresolved: tuple[bpy.types.NodeSocket, bpy.types.NodeSocket],
    slope_max: float,
) -> bpy.types.NodeSocket:
    """eps(theta) of the sea emitted, the remaining 1 - eps reflected from the sky.

    Complements, so the two very nearly cancel and the sea holds close to ambient at
    every angle.
    """
    roughness, aspect = _lobe(tree, *unresolved)
    mirror = tree.nodes.new("ShaderNodeBsdfAnisotropic")
    # Glossy BSDF ships at 0.8 grey. The Mix Shader already applies the 1 - eps
    # weighting, so anything but white here absorbs reflected sky and cuts a dark
    # notch along the horizon.
    mirror.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.inputs["Strength"].default_value = lwir.band_radiance(sea.t_sea_k)
    mix = tree.nodes.new("ShaderNodeMixShader")

    link = tree.links.new
    link(normal, mirror.inputs["Normal"])
    link(tangent, mirror.inputs["Tangent"])
    link(roughness, mirror.inputs["Roughness"])
    # Glossy's alpha_y / alpha_x = (1 + A)^2 for A < 0, so A = sqrt(aspect) - 1.
    link(
        _math(tree, "SUBTRACT", _math(tree, "SQRT", aspect), 1.0),
        mirror.inputs["Anisotropy"],
    )
    mean = _math(tree, "MULTIPLY", _math(tree, "ADD", *unresolved), 0.5)
    fraction = _math(tree, "DIVIDE", _math(tree, "SQRT", mean), slope_max)
    emissivity = lookup(
        tree,
        _emissivity_image(sea.t_sea_k, slope_max),
        _incidence(tree, normal),
        fraction,
    )
    # Mix Shader names both shader inputs "Shader", so they can only be indexed. Factor
    # is emissivity: 0 at grazing incidence takes the mirror, 1 head-on takes emission.
    link(mirror.outputs["BSDF"], mix.inputs[1])
    link(emission.outputs["Emission"], mix.inputs[2])
    link(emissivity, mix.inputs["Factor"])
    return mix.outputs["Shader"]


def _daylight(
    tree: bpy.types.NodeTree,
    sea: Sea,
    normal: bpy.types.NodeSocket,
    tangent: bpy.types.NodeSocket,
    unresolved: tuple[bpy.types.NodeSocket, bpy.types.NodeSocket],
    whitecaps: bpy.types.NodeSocket,
) -> bpy.types.NodeSocket:
    """Water refracting at seawater's IOR, white where its crests break."""
    roughness, aspect = _lobe(tree, *unresolved)
    principled = tree.nodes.new("ShaderNodeBsdfPrincipled")
    principled.inputs["Base Color"].default_value = (*WATER_BODY_COLOR, 1.0)
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
    return mix.outputs["Shader"]


def _material(
    sea: Sea,
    wind: tuple[Wave, ...],
    swell: tuple[Wave, ...],
    band: Band,
    outputs: Outputs,
    pixel_rad: float,
) -> bpy.types.Material:
    """Each pixel draws the waves it resolves and takes the rest as roughness."""
    material = bpy.data.materials.new("sea")
    tree = material.node_tree
    tree.nodes.clear()
    field = wind + swell
    pixel = _pixel(tree, pixel_rad)
    normal, drawn = _waves(tree, field, outputs, pixel)

    def unresolved_at(footprint_m: float) -> float:
        return unresolved_slope_variance(sea.wind_speed_mps, wind, swell, footprint_m)

    table = _footprint_table("sea_unresolved_variance", unresolved_at)
    unresolved = (
        _at_footprint(tree, table, pixel.along_m, "sea_unresolved_variance"),
        _at_footprint(tree, table, pixel.across_m, "sea_unresolved_variance"),
    )
    if band == "eo":
        fraction = whitecap_fraction(sea.wind_speed_mps)
        # Along the view, the widest footprint, leaves the most unresolved. A judgement.
        whitecaps = _whitecaps(tree, wind, drawn[: len(wind)], pixel.along_m, fraction)
        surface = _daylight(tree, sea, normal, pixel.along_dir, unresolved, whitecaps)
    else:
        # The coarsest footprint leaves the most: Cox & Munk's and all the swell's.
        slope_max = math.sqrt(unresolved_at(FOOTPRINT_RANGE_M[1]))
        surface = _thermal(tree, sea, normal, pixel.along_dir, unresolved, slope_max)
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    tree.links.new(surface, output.inputs["Surface"])
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
    """A grid curved to the earth. The waves are in its material."""
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
    water.data.materials.append(_material(sea, wind, swell, band, outputs, pixel_rad))
    return water
