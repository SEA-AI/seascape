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

from seascape.blender.blend import (
    SUN,
    Num,
    combine,
    drive,
    mix_shader,
    node,
    scalar,
    separate,
    vector,
)
from seascape.blender.sea.foam import _Whitecaps
from seascape.blender.sea.pixel import _Pixel, _redraw
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
    _, _, up = separate(tree, toward.outputs["Vector"])
    risen = scalar(tree, "GREATER_THAN", up, 0.0, name="glint_risen")
    # Incoming points at the camera: the mirroring facet faces the half-way vector.
    mirror = vector(
        tree,
        "NORMALIZE",
        vector(tree, "ADD", geometry.outputs["Incoming"], toward.outputs["Vector"]),
    )
    # 2 (1 - cos d) = d^2 - d^4 / 12.
    off_sq = (1.0 - Num(tree, vector(tree, "DOT_PRODUCT", normal, mirror))) * 2.0
    fade = node(
        tree,
        "ShaderNodeMapRange",
        {
            "Value": (off_sq / Num(tree, variance).op("MAXIMUM", 1e-12)).socket,
            "From Min": GLINT_REACH[0],
            "From Max": GLINT_REACH[1],
            "To Min": 1.0,
            "To Max": 0.0,
        },
        name="glint_reach",
        interpolation_type="SMOOTHSTEP",
    )
    return scalar(tree, "MULTIPLY", fade.outputs["Result"], risen)


def _glitter(
    tree: bpy.types.NodeTree,
    wind: tuple[Wave, ...],
    where: bpy.types.NodeSocket,
    pixel: _Pixel,
    unresolved: bpy.types.NodeSocket,
    glinting: bpy.types.NodeSocket,
) -> tuple[bpy.types.NodeSocket, bpy.types.NodeSocket]:
    """A Gaussian tilt per cell of one specular point, re-drawn as it twinkles, and the
    slope variance the tilts carry out of the lobe.

    A pixel's s samples of its n cells count glints as n cells do if each cell's lobe
    catches the sun n / s times as often: variance r^2 (n / s - 1) about a sun of slope
    radius r, s being `cycles.samples`.

    Limitation: the cells keep the mean wind's size through a gust, as a size that
    varied over the sea would re-cut the grid as the gust moves; crossfade two fixed
    grids by the local wind if glints should thin in the lulls.

    Args:
        tree: Where the nodes go.
        wind: The wind waves, which size the cells and set their twinkle.
        where: (east, north, time) the shader reads at.
        pixel: The footprint, which sets n.
        unresolved: The slope variance the lobe would take without the tilts.
        glinting: 1 where a facet can mirror the sun, as `_glinting` gives it; the
            tilts carry variance only there."""
    cell_m2 = specular_cell_m2(wind)
    cells = Num(tree, pixel.along_m) * pixel.across_m / cell_m2
    widen = cells.madd(0.0, -1.0)
    drive(
        widen.socket.node.inputs[1],
        "default_value",
        "1 / samples",
        bpy.context.scene,
        samples="cycles.samples",
    )
    lobe = widen.op("MAXIMUM", 0.0) * SUN_SLOPE_RADIUS**2
    carried = (Num(tree, unresolved) - lobe).op("MAXIMUM", 0.0) * glinting
    carried.socket.node.name = "glitter_variance"
    east, north, time_s = separate(tree, where)
    scaled = vector(
        tree,
        "SCALE",
        combine(tree, east, north),
        1 / math.sqrt(cell_m2),
        name="glitter_cells",
    )
    # Each cell twinkles at its own phase, or the whole sea would flip at once.
    log_u, angle_u = _redraw(
        tree, vector(tree, "FLOOR", scaled), time_s, twinkle_hz(wind), "glitter_draw"
    )
    # Box-Muller: a radius from one uniform, an angle from the other.
    sigma = (carried * 0.5).op("SQRT")
    radius = (log_u * -2.0).op("SQRT") * sigma
    angle = angle_u * (2 * math.pi)
    tilt = node(
        tree,
        "ShaderNodeVectorRotate",
        {"Vector": combine(tree, radius.socket), "Angle": angle.socket},
        rotation_type="Z_AXIS",
        name="glitter_tilt",
    )
    return tilt.outputs["Vector"], carried.socket


def _lobe(
    tree: bpy.types.NodeTree,
    along: bpy.types.NodeSocket,
    across: bpy.types.NodeSocket,
) -> tuple[bpy.types.NodeSocket, bpy.types.NodeSocket]:
    """Beckmann roughness and Cycles anisotropy for the unresolved slope variance at
    the two footprints. Per axis alpha = sqrt(2) sigma_axis = sqrt(variance); Blender's
    roughness is their geometric mean's square root.

    Args:
        tree: Where the nodes go.
        along: The variance along the lobe's tangent.
        across: The variance across it."""
    along, across = (Num(tree, s).op("MAXIMUM", 1e-12) for s in (along, across))
    roughness = (along * across).op("POWER", 0.125).op("MINIMUM", 1.0)
    roughness.socket.node.name = "sea_roughness"
    aspect = (across / along).op("SQRT")
    # Cycles' alpha_y / alpha_x = 1 - 0.9 a (kernel/svm/closure.h), so
    # a = (1 - aspect) / 0.9.
    return roughness.socket, ((1.0 - aspect) * (1 / 0.9)).socket


def _stretchable(
    tree: bpy.types.NodeTree, wide: bpy.types.NodeSocket, narrow: bpy.types.NodeSocket
) -> bpy.types.NodeSocket:
    """A lobe's narrower slope variance, no narrower than Cycles stretches a lobe: 10 to
    1 in alpha, 100 in variance.

    Args:
        tree: Where the nodes go.
        wide: The lobe's wider variance.
        narrow: The variance to hold to a hundredth of the wider."""
    return scalar(tree, "MAXIMUM", narrow, scalar(tree, "MULTIPLY", wide, 0.01))


def _daylight(
    tree: bpy.types.NodeTree,
    sea: Sea,
    normal: bpy.types.NodeSocket,
    tangent: bpy.types.NodeSocket,
    unresolved: tuple[bpy.types.NodeSocket, bpy.types.NodeSocket],
    whitecaps: _Whitecaps,
    wake: _Wakes | None,
) -> bpy.types.NodeSocket:
    """Water reflecting at seawater's IOR over its body's colour, white where its crests
    break and where a hull leaves foam.

    Beckmann, as Cox & Munk's slopes are near Gaussian; Principled offers only GGX. A
    conductor's Fresnel at zero extinction is the dielectric's.
    """
    roughness, anisotropy = _lobe(tree, *unresolved)
    link = tree.links.new
    mirror = node(
        tree,
        "ShaderNodeBsdfMetallic",
        {
            "IOR": (SEAWATER_IOR,) * 3,
            "Extinction": (0.0, 0.0, 0.0),
            "Normal": normal,
            "Tangent": tangent,
            "Roughness": roughness,
            "Anisotropy": anisotropy,
        },
        name="sea_specular",
        distribution="BECKMANN",
        fresnel_type="PHYSICAL_CONDUCTOR",
    )
    fresnel = node(tree, "ShaderNodeFresnel", {"IOR": SEAWATER_IOR, "Normal": normal})
    through = 1.0 - Num(tree, fresnel.outputs["Fac"])
    # A whitecap's bubbles brighten the water under all its foam, as a wake's do.
    bubbles = Num(tree, whitecaps.bubbles) * BUBBLE_GAIN
    if wake is not None:
        bubbles = bubbles.op("MAXIMUM", wake.bubbles)
    through = through * (bubbles + 1.0)
    # Dimmed by Fresnel's 1 - F toward the viewer only.
    color = body_color(sea.chlorophyll_mg_m3, SEAWATER_IOR)
    body = node(
        tree,
        "ShaderNodeBsdfDiffuse",
        {
            "Normal": normal,
            "Color": vector(
                tree, "SCALE", color, through.socket, name="sea_body_color"
            ),
        },
    )
    water = tree.nodes.new("ShaderNodeAddShader")
    link(mirror.outputs["BSDF"], water.inputs[0])
    link(body.outputs["BSDF"], water.inputs[1])
    foam = node(
        tree,
        "ShaderNodeBsdfDiffuse",
        {"Normal": normal, "Color": (*(FRESH_FOAM_REFLECTANCE,) * 3, 1.0)},
    )
    if whitecaps.albedo is not None:
        link(whitecaps.albedo, foam.inputs["Color"])
        # Foam is lumpy: its own normal, bumped by its density over the water's.
        bump = node(
            tree,
            "ShaderNodeBump",
            {
                "Strength": FOAM_BUMP_STRENGTH,
                "Distance": FOAM_BUMP_M,
                "Height": whitecaps.albedo,
                "Normal": normal,
            },
        )
        link(bump.outputs["Normal"], foam.inputs["Normal"])
    cover = whitecaps.foam
    if wake is not None:
        # Wake and whitecap foam are one Diffuse, mixed at 1 - (1 - a)(1 - b): a second
        # Diffuse and Mix overflow Cycles' SVM stack. Mix clamps its factor to 1, the
        # product does not, so `cover` is clamped here.
        unfoamed = (1.0 - Num(tree, cover).op("MINIMUM", 1.0)) * (
            1.0 - Num(tree, wake.foam)
        )
        cover = (1.0 - unfoamed).socket
    return mix_shader(tree, cover, water.outputs["Shader"], foam.outputs["BSDF"])


def _widest(
    tree: bpy.types.NodeTree,
    pixel: _Pixel,
    unresolved: tuple[bpy.types.NodeSocket, bpy.types.NodeSocket],
) -> tuple[bpy.types.NodeSocket, bpy.types.NodeSocket, bpy.types.NodeSocket]:
    """The wider and narrower of the unresolved variances, and the horizontal along
    the wider: Cycles' Metallic BSDF stretches a lobe along its tangent only. Grazing,
    the footprints make it the view's; a steep view down a crosswind-heavy sea turns
    it."""
    across_dir = vector(tree, "CROSS_PRODUCT", pixel.along_dir, (0.0, 0.0, 1.0))
    tangent = node(
        tree,
        "ShaderNodeMix",
        {
            "Factor": scalar(tree, "GREATER_THAN", unresolved[1], unresolved[0]),
            "A": pixel.along_dir,
            "B": across_dir,
        },
        data_type="VECTOR",
    )
    return (
        scalar(tree, "MAXIMUM", *unresolved, name="sea_unresolved_wide"),
        scalar(tree, "MINIMUM", *unresolved, name="sea_unresolved_narrow"),
        tangent.outputs["Result"],
    )
