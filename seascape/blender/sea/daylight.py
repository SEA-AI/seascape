"""The EO sea: its microfacet lobe and sun glitter over its body's colour.

Sources
-------
Microfacet lobe: Walter, Marschner, Li & Torrance, "Microfacet models for refraction
through rough surfaces", EGSR 2007 (doi:10.2312/EGWR/EGSR07/195-206) for Beckmann's
distribution and its Smith shadowing; Burley, "Physically-based shading at Disney",
SIGGRAPH 2012 course notes, for the alpha = roughness^2 convention Cycles follows.

Seawater: Quan & Fry, "Empirical equation for the index of refraction of seawater",
Applied Optics 34(18) 3477, 1995 (doi:10.1364/AO.34.003477).
"""

import math

import bpy

from seascape.blend import (
    SUN,
    _math,
    _vector,
    drive,
)
from seascape.blender.sea.pixel import _Pixel
from seascape.blender.sea.wakes import _Wakes
from seascape.config import Sea
from seascape.physics.wakes import (
    BUBBLE_GAIN,
    FRESH_FOAM_REFLECTANCE,
)
from seascape.physics.water import body_color
from seascape.physics.waves import (
    Wave,
    specular_cell_m2,
    twinkle_hz,
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
# A judgement: foam's lumps, bumped from its density.
FOAM_BUMP_STRENGTH = 1.0
FOAM_BUMP_M = 0.5


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


def _daylight(
    tree: bpy.types.NodeTree,
    sea: Sea,
    normal: bpy.types.NodeSocket,
    tangent: bpy.types.NodeSocket,
    unresolved: tuple[bpy.types.NodeSocket, bpy.types.NodeSocket],
    whitecaps: tuple[bpy.types.NodeSocket, bpy.types.NodeSocket, bpy.types.NodeSocket],
    wake: _Wakes | None = None,
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
