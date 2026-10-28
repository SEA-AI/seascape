"""The world: the sky each band sees, its sun, and the haze between."""

import math

import bpy
import numpy as np

from seascape import skies
from seascape.assets import download
from seascape.blender import sea
from seascape.blender.blend import (
    CURVE_SAMPLES,
    SUN,
    _math,
    curve_image,
    drive,
    lookup,
)
from seascape.config import (
    Band,
    Sky,
)
from seascape.physics import lwir

# Judgement: nearer than a lens is to anything it sees.
HAZE_NEAR_M = 0.1

# Judgement.
THERMAL_SKY_PX = 2048


def _sky(sky: Sky, band: Band) -> bpy.types.World:
    world = bpy.data.worlds.new("sky")
    # Nonzero, EEVEE turns world light above it into a sun a mirror cannot see.
    world.sun_threshold = 0.0
    tree = world.node_tree
    if band == "ir":
        return _thermal_sky(world, sky)
    if sky.hdri is not None:
        direction = tree.nodes.new("ShaderNodeTexCoord").outputs["Generated"]
        tree.links.new(
            _photo(tree, sky.hdri, sky.sun_bearing_deg, direction),
            tree.nodes["Background"].inputs["Color"],
        )
        return world
    node = _sky_texture(tree, sky)
    for prop in SUN:
        world[prop] = getattr(node, prop)
        drive(node, prop, "sky", world, sky=f'["{prop}"]')
    for prop in SUN[:2]:
        world.id_properties_ui(prop).update(subtype="ANGLE")
    tree.links.new(node.outputs["Color"], tree.nodes["Background"].inputs["Color"])
    return world


def _photo(
    tree: bpy.types.NodeTree,
    name: str,
    sun_bearing_deg: float,
    direction: bpy.types.NodeSocket,
) -> bpy.types.NodeSocket:
    """The photographed sky `name` in `direction`, turned so its sun sits at
    `sun_bearing_deg`."""
    return _turned(tree, _photo_image(name), name, sun_bearing_deg, direction)


def _photo_image(name: str) -> bpy.types.Image:
    photo = skies.library()[name]
    return bpy.data.images.load(
        str(download(name, photo.url, photo.sha256)), check_existing=True
    )


def _photo_turn_rad(name: str, sun_bearing_deg: float) -> float:
    """How far to turn the photographed sky `name` to put its sun at
    `sun_bearing_deg`. Turning a lookup by d carries the image's sun clockwise by d: no
    negation."""
    return math.radians(sun_bearing_deg - skies.library()[name].sun_bearing_deg)


def _turned(
    tree: bpy.types.NodeTree,
    image: bpy.types.Image,
    name: str,
    sun_bearing_deg: float,
    direction: bpy.types.NodeSocket,
) -> bpy.types.NodeSocket:
    """`image`, laid out as the photographed sky `name`, in `direction`, turned so the
    photo's sun sits at `sun_bearing_deg`."""
    turn = tree.nodes.new("ShaderNodeMapping")
    turn.inputs["Rotation"].default_value = (
        0.0,
        0.0,
        _photo_turn_rad(name, sun_bearing_deg),
    )
    tree.links.new(direction, turn.inputs["Vector"])
    node = tree.nodes.new("ShaderNodeTexEnvironment")
    node.image = image
    tree.links.new(turn.outputs["Vector"], node.inputs["Vector"])
    return node.outputs["Color"]


def _sky_texture(tree: bpy.types.NodeTree, sky: Sky) -> bpy.types.Node:
    assert sky.sun_elevation_deg is not None
    node = tree.nodes.new("ShaderNodeTexSky")
    # `turbidity` belongs to Preetham and Hosek-Wilkie and is silently ignored here.
    node.sky_type = "MULTIPLE_SCATTERING"
    node.sun_elevation = math.radians(sky.sun_elevation_deg)
    # An azimuth, clockwise from +Y, though Blender calls it a rotation.
    node.sun_rotation = math.radians(sky.sun_bearing_deg)
    # Its aerosol stays at Blender's default: more darkens its horizon, where maritime
    # aerosol, nearly non-absorbing and grey across the visible (Hess et al. 1998),
    # whitens it. `visibility_km` hazes what is in front of it.
    return node


def _haze(sky: Sky, band: Band, far_m: float) -> None:
    """Koschmieder's airlight, in every surface: a ray keeps exp(-optical depth) of the
    light it brings back and takes the rest from the sky ahead of it, no lower than the
    horizon. That sky is constant along the ray, so the mix is exact over its length,
    and hazing a ray to the sky would change nothing: the world has none.

    EO's extinction is uniform. LWIR's optical depth grows slower than the path.
    """
    if band == "eo" and not sky.extinction_per_m:
        return
    group = bpy.data.node_groups.new("haze", "ShaderNodeTree")
    for in_out in ("INPUT", "OUTPUT"):
        group.interface.new_socket(
            "Shader", in_out=in_out, socket_type="NodeSocketShader"
        )
    link = group.links.new
    geometry = group.nodes.new("ShaderNodeNewGeometry")
    ahead = group.nodes.new("ShaderNodeVectorMath")
    ahead.name = "haze_ahead"
    ahead.operation = "SCALE"
    link(geometry.outputs["Incoming"], ahead.inputs["Vector"])
    # Incoming points back at the camera.
    ahead.inputs["Scale"].default_value = -1.0
    horizon = group.nodes.new("ShaderNodeVectorMath")
    horizon.name = "haze_horizon"
    horizon.operation = "MAXIMUM"
    link(ahead.outputs["Vector"], horizon.inputs[0])
    # Below the horizon the sky is the ground's, not the air's.
    horizon.inputs[1].default_value = (-1.0, -1.0, 0.0)
    airlight = group.nodes.new("ShaderNodeEmission")
    airlight.name = "haze_airlight"
    link(
        _sky_ahead(group, sky, band, horizon.outputs["Vector"]),
        airlight.inputs["Color"],
    )
    depth = _optical_depth(group, sky, band, far_m)
    kept = _math(group, "EXPONENT", _math(group, "MULTIPLY", depth, -1.0))
    mix = group.nodes.new("ShaderNodeMixShader")
    mix.name = "haze_mix"
    link(_math(group, "SUBTRACT", 1.0, kept), mix.inputs["Fac"])
    link(group.nodes.new("NodeGroupInput").outputs["Shader"], mix.inputs[1])
    link(airlight.outputs["Emission"], mix.inputs[2])
    link(mix.outputs["Shader"], group.nodes.new("NodeGroupOutput").inputs["Shader"])
    for material in bpy.data.materials:
        haze(material)


def haze(material: bpy.types.Material) -> None:
    """Put the build's haze between `material`'s surface and its output. `build` does
    it for every material it makes; one made after is seen through clear air."""
    group = bpy.data.node_groups.get("haze")
    tree = material.node_tree
    if group is None or tree is None:
        return
    output = tree.get_output_node("CYCLES")
    if output is None or not output.inputs["Surface"].links:
        return
    surface = output.inputs["Surface"].links[0].from_node
    # Twice would haze it twice.
    if getattr(surface, "node_tree", None) == group:
        return
    node = tree.nodes.new("ShaderNodeGroup")
    node.name = "haze"
    node.node_tree = group
    tree.links.new(output.inputs["Surface"].links[0].from_socket, node.inputs["Shader"])
    tree.links.new(node.outputs["Shader"], output.inputs["Surface"])
    # The airlight emits, and in LWIR so does every surface. As a light, each sample
    # runs the whole shader for light the rays that hit it already bring back.
    material.cycles.emission_sampling = "NONE"


def _sky_ahead(
    tree: bpy.types.NodeTree, sky: Sky, band: Band, direction: bpy.types.NodeSocket
) -> bpy.types.NodeSocket:
    """The band's sky in `direction`."""
    if band == "eo" and sky.hdri is not None:
        return _photo(tree, sky.hdri, sky.sun_bearing_deg, direction)
    if band == "eo":
        node = _sky_texture(tree, sky)
        node.name = "haze_sky"
        world = bpy.context.scene.world
        for prop in SUN:
            drive(node, prop, "sky", world, sky=f'["{prop}"]')
        # The sun disc disables the Vector input, and a link to it is then ignored.
        node.sun_disc = False
        tree.links.new(direction, node.inputs["Vector"])
        return node.outputs["Color"]
    height = tree.nodes.new("ShaderNodeSeparateXYZ")
    tree.links.new(direction, height.inputs["Vector"])
    return lookup(tree, _sky_image(sky), height.outputs["Z"])


def _haze_ranges_m(far_m: float) -> np.ndarray:
    """Texel centres of the LWIR depth table, log-spaced out to the far plane."""
    texel = (np.arange(CURVE_SAMPLES) + 0.5) / CURVE_SAMPLES
    return HAZE_NEAR_M * (far_m / HAZE_NEAR_M) ** texel


def _optical_depth(
    tree: bpy.types.NodeTree, sky: Sky, band: Band, far_m: float
) -> bpy.types.NodeSocket:
    """The band's optical depth over the ray that reached the shading point.

    ponytail: exact along camera rays. LWIR takes a secondary ray, the sea reflected in
    a hull's paint, as if it ran straight out from the camera, ending at the shading
    point's distance; its true start needs the path's length so far, which a shader
    cannot read.
    """
    length = tree.nodes.new("ShaderNodeLightPath").outputs["Ray Length"]
    if band == "eo":
        return _math(tree, "MULTIPLY", length, sky.extinction_per_m, name="haze_beta")
    ranges = _haze_ranges_m(far_m)
    image = curve_image(
        "haze_depth", lwir.path_optical_depth(ranges, sky.visibility_km, sky.atmosphere)
    )
    span = far_m / HAZE_NEAR_M

    def at(range_m: bpy.types.NodeSocket) -> bpy.types.NodeSocket:
        # Blender's log of a negative number is 0, which is a range inside the table.
        near = _math(tree, "MAXIMUM", range_m, HAZE_NEAR_M)
        texel = _math(tree, "LOGARITHM", near, span)
        texel = _math(tree, "SUBTRACT", texel, math.log(HAZE_NEAR_M, span))
        return lookup(tree, image, texel)

    distance = tree.nodes.new("ShaderNodeVectorMath")
    distance.operation = "LENGTH"
    camera = tree.nodes.new("ShaderNodeTexCoord").outputs["Camera"]
    tree.links.new(camera, distance.inputs["Vector"])
    start = _math(tree, "SUBTRACT", distance.outputs["Value"], length)
    return _math(tree, "SUBTRACT", at(distance.outputs["Value"]), at(start))


def _sky_image(sky: Sky) -> bpy.types.Image:
    """`lwir.sky_radiance` baked against sin(elevation) at texel centres, which is what
    the shader samples.

    A world shader's ray direction is a unit vector, so its Z is already sin(elevation)
    and no arcsine node is needed. Below the horizon Z is clamped to 0, where the curve
    holds at its first texel.
    """
    # The world baked it first; `build` starts from factory settings, so the name is
    # this build's.
    if (baked := bpy.data.images.get("sky_radiance")) is not None:
        return baked
    sin_elevation = (np.arange(CURVE_SAMPLES) + 0.5) / CURVE_SAMPLES
    radiance = lwir.sky_radiance(np.arcsin(sin_elevation), sky.t_air_k, sky.atmosphere)
    return curve_image("sky_radiance", radiance)


def _cloudy_sky_image(sky: Sky) -> bpy.types.Image:
    """`lwir.cloudy_sky_radiance` under the photo's clouds, laid out as the photo."""
    assert sky.hdri is not None
    photo = _photo_image(sky.hdri)
    w, h = photo.size
    pixels = np.empty(w * h * 4, np.float32)
    photo.pixels.foreach_get(pixels)
    bpy.data.images.remove(photo)
    # Bottom row first, as Blender stores an image: the sky is the top half.
    cloud = np.zeros((h, w), np.float32)
    cloud[h // 2 :] = skies.cloud(pixels.reshape(h, w, 4)[h // 2 :, :, :3])
    f = max(1, w // THERMAL_SKY_PX)
    cloud = cloud[: h // f * f, : w // f * f]
    cloud = cloud.reshape(h // f, f, w // f, f).mean(axis=(1, 3))
    rows = cloud.shape[0]
    elevation = np.radians((np.arange(rows) + 0.5) / rows * 180.0 - 90.0)
    radiance = lwir.cloudy_sky_radiance(
        elevation[:, None], cloud, sky.cloud_base_m, sky.t_air_k, sky.atmosphere
    )
    image = curve_image("cloudy_sky_radiance", radiance)
    # As a horizontal surface sees the sky: each row by its projected solid angle.
    up = elevation > 0
    weight = np.sin(elevation[up]) * np.cos(elevation[up])
    fraction = weight @ cloud[up].mean(axis=1) / weight.sum()
    # A weighted mean of values in [0, 1] can round past 1.
    image["cloud_fraction"] = float(np.clip(fraction, 0.0, 1.0))
    return image


def _clouds(sky: Sky) -> sea.Clouds | None:
    """The IR world's clouds by bearing, read off the image the world baked first."""
    if sky.hdri is None:
        return None
    image = bpy.data.images["cloudy_sky_radiance"]
    w, h = image.size
    pixels = np.empty(w * h * 4, np.float32)
    image.pixels.foreach_get(pixels)
    # Bottom row first, as `_cloudy_sky_image` lays it out.
    elevation = np.radians((np.arange(h) + 0.5) / h * 180.0 - 90.0)
    radiance = pixels.reshape(h, w, 4)[..., 0].astype(np.float64)
    return sea.Clouds(
        elevation, radiance, _photo_turn_rad(sky.hdri, sky.sun_bearing_deg)
    )


def _sky_round(sky: Sky, clouds: sea.Clouds | None) -> sea.SkyCurve:
    """The IR world's radiance against elevation, averaged round the horizon: its clear
    sky's table, or its clouds' rows."""
    if clouds is None:
        elevation = np.arcsin((np.arange(CURVE_SAMPLES) + 0.5) / CURVE_SAMPLES)
        return elevation, lwir.sky_radiance(elevation, sky.t_air_k, sky.atmosphere)
    return clouds.elevation, clouds.radiance.mean(axis=1)


def _thermal_sky(world: bpy.types.World, sky: Sky) -> bpy.types.World:
    """Downwelling radiance, as raw W m^-2 sr^-1: against elevation, or under the
    photo's clouds."""
    tree = world.node_tree
    tree.nodes.clear()
    link = tree.links.new
    coord = tree.nodes.new("ShaderNodeTexCoord")
    background = tree.nodes.new("ShaderNodeBackground")
    output = tree.nodes.new("ShaderNodeOutputWorld")
    link(background.outputs["Background"], output.inputs["Surface"])
    if sky.hdri is not None:
        radiance = _turned(
            tree,
            _cloudy_sky_image(sky),
            sky.hdri,
            sky.sun_bearing_deg,
            coord.outputs["Generated"],
        )
        # Linear shows the texel grid where a texel spans several pixels.
        radiance.node.interpolation = "Cubic"
        link(radiance, background.inputs["Color"])
        return world
    height = tree.nodes.new("ShaderNodeSeparateXYZ")
    above = tree.nodes.new("ShaderNodeMath")
    above.operation = "MAXIMUM"
    above.use_clamp = True
    above.inputs["Value_001"].default_value = 0.0

    link(coord.outputs["Generated"], height.inputs["Vector"])
    link(height.outputs["Z"], above.inputs["Value"])
    radiance = lookup(tree, _sky_image(sky), above.outputs["Value"])
    link(radiance, background.inputs["Color"])
    return world


def _sun_vector(sky: Sky) -> tuple[float, float, float]:
    """Unit vector towards the sun. A direction, so the bearing is not negated: that
    belongs to rotations, and `vessels._pose` places by the same sin/cos."""
    assert sky.sun_elevation_deg is not None
    elevation = math.radians(sky.sun_elevation_deg)
    bearing = math.radians(sky.sun_bearing_deg)
    return (
        math.cos(elevation) * math.sin(bearing),
        math.cos(elevation) * math.cos(bearing),
        math.sin(elevation),
    )


def _light(sky: Sky, band: Band) -> sea.Light:
    if band == "ir":
        clouds = _clouds(sky)
        return sea.Light(sky=_sky_round(sky, clouds), sun_up=sky.sun_up, clouds=clouds)
    if sky.sun_elevation_deg is None:
        return sea.Light()
    return sea.Light(sun=_sun_vector(sky))
