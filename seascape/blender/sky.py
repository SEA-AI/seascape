"""The world: the sky each band sees, its sun, and the haze between."""

import math

import bpy
import cv2
import numpy as np
from mathutils import Vector

from seascape.assets import download, skies
from seascape.blender import sea
from seascape.blender.blend import (
    CURVE_SAMPLES,
    SUN,
    Num,
    clamped,
    curve_image,
    drive,
    lookup,
    mix_shader,
    node,
    pixels,
    scalar,
    separate,
    value,
    vector,
)
from seascape.config import (
    Band,
    Sky,
)
from seascape.physics import aerosol, lwir, solar

# Judgement: nearer than a lens is to anything it sees.
HAZE_NEAR_M = 0.1

# Judgement.
THERMAL_SKY_PX = 2048

# The Astronomical Almanac: the sun's mean semi-diameter is 16 arcminutes.
SUN_DIAMETER_DEG = 32.0 / 60.0


def add_world(sky: Sky, band: Band) -> bpy.types.World:
    world = bpy.data.worlds.new("sky")
    # Nonzero, EEVEE turns world light above it into a sun a mirror cannot see.
    world.sun_threshold = 0.0
    tree = world.node_tree
    if band == "ir":
        return _thermal_sky(world, sky)
    background = tree.nodes["Background"].inputs["Color"]
    ray = tree.nodes.new("ShaderNodeTexCoord").outputs["Generated"]
    if sky.hdri is not None:
        if sky.fog_weight:
            colour = _fogged_photo(tree, sky, ray, sky.sun_transmittance)
        else:
            colour = _photo(tree, sky.hdri, sky.sun_bearing_deg, ray)
        tree.links.new(colour, background)
        if sky.sun_diffuse_fraction is not None:
            _add_hidden_sun(sky)
        return world
    node = _sky_texture(tree, sky)
    for prop in SUN:
        world[prop] = getattr(node, prop)
        drive(node, prop, "sky", world, sky=f'["{prop}"]')
    for prop in SUN[:2]:
        world.id_properties_ui(prop).update(subtype="ANGLE")
    if not sky.fog_weight:
        tree.links.new(node.outputs["Color"], background)
        return world
    clear = _clear_eo(tree, sky, world, ray)
    # The sun's disc alone: the sky with it less the sky without.
    disc = vector(tree, "SUBTRACT", node.outputs["Color"], clear)
    kept = value(tree, "fog_sun", sky.sun_transmittance)
    # aerosol.sun_transmittance, as the sun moves.
    drive(
        kept,
        "default_value",
        f"exp(-{aerosol.fog_beam_depth(sky.visibility_km)!r} / max(sin(e), 1e-6))",
        world,
        e=f'["{SUN[0]}"]',
    )
    dome = _misted(tree, sky, clear, _clear_eo(tree, sky, world, _horizon(tree, ray)))
    sun = vector(tree, "SCALE", disc, kept)
    tree.links.new(vector(tree, "ADD", dome, sun), background)
    return world


def _add_hidden_sun(sky: Sky) -> None:
    assert sky.hdri is not None
    assert sky.sun_diffuse_fraction is not None
    assert sky.sun_elevation_deg is not None
    diffuse = skies.library()[sky.hdri].diffuse_irradiance
    assert diffuse is not None
    beam = solar.beam_normal(
        diffuse, sky.sun_diffuse_fraction, math.radians(sky.sun_elevation_deg)
    )
    light = bpy.data.lights.new("sun", "SUN")
    light.angle = math.radians(SUN_DIAMETER_DEG)
    # Measured: a Sun's strength is its irradiance across the beam, in the units the
    # world's radiance integrates to.
    # Limitation: the diffuse light is the unmisted photo's, and the beam fog takes is
    # not given back as diffuse light, as LWIR gives it.
    light.energy = beam * sky.sun_transmittance
    sun = bpy.data.objects.new("sun", light)
    # Blender hides a new Sun from camera rays. Shown, it is the beam through the cloud.
    sun.visible_camera = True
    # A Sun shines down its -Z.
    sun.rotation_euler = Vector(sky.sun_vector).to_track_quat("Z", "Y").to_euler()
    bpy.context.scene.collection.objects.link(sun)


def _fogged_photo(
    tree: bpy.types.NodeTree,
    sky: Sky,
    direction: bpy.types.NodeSocket,
    sun_kept: float,
) -> bpy.types.NodeSocket:
    """The photo in `direction` misted, its sun's disc keeping `sun_kept` of itself. A
    sun behind cloud greys with it."""
    assert sky.hdri is not None
    colour = _photo(tree, sky.hdri, sky.sun_bearing_deg, direction)
    horizon = _photo(tree, sky.hdri, sky.sun_bearing_deg, _horizon(tree, direction))
    misted = _misted(tree, sky, colour, horizon)
    # The mist keeps 1 - fog_weight of the disc.
    extra = sun_kept - (1.0 - sky.fog_weight)
    floor = skies.library()[sky.hdri].sun_floor
    if floor is None or not extra:
        return misted
    unit = vector(tree, "NORMALIZE", direction)
    toward = vector(tree, "DOT_PRODUCT", unit, sky.sun_vector)
    near = scalar(tree, "GREATER_THAN", toward, math.cos(math.radians(skies.DISC_DEG)))
    luminance = vector(
        tree, "DOT_PRODUCT", colour, tuple(float(c) for c in skies.LUMINANCE)
    )
    bright = scalar(tree, "GREATER_THAN", luminance, floor)
    disc = scalar(tree, "MULTIPLY", near, bright, name="photo_disc")
    sun = vector(tree, "SCALE", colour, scalar(tree, "MULTIPLY", disc, extra))
    return vector(tree, "ADD", misted, sun)


def _clear_eo(
    tree: bpy.types.NodeTree,
    sky: Sky,
    world: bpy.types.World,
    direction: bpy.types.NodeSocket,
) -> bpy.types.NodeSocket:
    """The Sky Texture without its sun disc in `direction`, its sun the world's."""
    node = _sky_texture(tree, sky)
    for prop in SUN:
        drive(node, prop, "sky", world, sky=f'["{prop}"]')
    # The sun disc disables the Vector input, and a link to it is then ignored.
    node.sun_disc = False
    tree.links.new(direction, node.inputs["Vector"])
    return node.outputs["Color"]


def _horizon(
    tree: bpy.types.NodeTree, direction: bpy.types.NodeSocket
) -> bpy.types.NodeSocket:
    """The horizon under or over `direction`."""
    flat = vector(tree, "MULTIPLY", direction, (1.0, 1.0, 0.0))
    return vector(tree, "NORMALIZE", flat)


def _misted(
    tree: bpy.types.NodeTree,
    sky: Sky,
    colour: bpy.types.NodeSocket,
    horizon: bpy.types.NodeSocket,
) -> bpy.types.NodeSocket:
    """`colour` moved `Sky.fog_weight` of the way to the fog's grey of `horizon`."""
    grey = _grey(tree, horizon)
    if sky.fog_weight == 1.0:
        return grey
    toward = vector(tree, "SUBTRACT", grey, colour)
    return vector(tree, "ADD", colour, vector(tree, "SCALE", toward, sky.fog_weight))


def _grey(
    tree: bpy.types.NodeTree, colour: bpy.types.NodeSocket
) -> bpy.types.NodeSocket:
    """Fog's sky, from the clear sky's `colour` at the horizon ahead.

    Judgement: fog drops scatter the visible without colour and many times over. Its
    brightness is the clear horizon's, which the haze fades every target into, so fog
    and sky meet without a seam.
    """
    return node(tree, "ShaderNodeRGBToBW", {"Color": colour}).outputs["Val"]


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
    turn_rad = _photo_turn_rad(name, sun_bearing_deg)
    turn = node(
        tree,
        "ShaderNodeMapping",
        {"Rotation": (0.0, 0.0, turn_rad), "Vector": direction},
    )
    environment = node(
        tree,
        "ShaderNodeTexEnvironment",
        {"Vector": turn.outputs["Vector"]},
        image=image,
    )
    return environment.outputs["Color"]


def _sky_texture(tree: bpy.types.NodeTree, sky: Sky) -> bpy.types.Node:
    assert sky.sun_elevation_deg is not None
    # Its aerosol stays at Blender's default: more darkens its horizon, where maritime
    # aerosol, nearly non-absorbing and grey across the visible (Hess et al. 1998),
    # whitens it. `visibility_km` hazes what is in front of it.
    texture = tree.nodes.new("ShaderNodeTexSky")
    # `turbidity` belongs to Preetham and Hosek-Wilkie and is silently ignored here.
    texture.sky_type = "MULTIPLE_SCATTERING"
    texture.sun_elevation = math.radians(sky.sun_elevation_deg)
    # An azimuth, clockwise from +Y, though Blender calls it a rotation.
    texture.sun_rotation = math.radians(sky.sun_bearing_deg)
    return texture


def add_haze(sky: Sky, band: Band, far_m: float) -> None:
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
    geometry = group.nodes.new("ShaderNodeNewGeometry")
    # Incoming points back at the camera.
    ahead = vector(
        group, "SCALE", geometry.outputs["Incoming"], -1.0, name="haze_ahead"
    )
    # Below the horizon the sky is the ground's, not the air's.
    horizon = vector(group, "MAXIMUM", ahead, (-1.0, -1.0, 0.0), name="haze_horizon")
    airlight = node(group, "ShaderNodeEmission", name="haze_airlight")
    group.links.new(_sky_ahead(group, sky, band, horizon), airlight.inputs["Color"])
    depth = _optical_depth(group, sky, band, far_m)
    kept = (Num(group, depth) * -1.0).op("EXPONENT")
    unlit = node(group, "NodeGroupInput").outputs["Shader"]
    shader = mix_shader(group, (1.0 - kept).socket, unlit, airlight.outputs["Emission"])
    shader.node.name = "haze_mix"
    if band == "ir":
        shader = _less_sky(group, sky, ahead, shader)
    node(group, "NodeGroupOutput", {"Shader": shader})
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
    wrapper = node(tree, "ShaderNodeGroup", name="haze", node_tree=group)
    tree.links.new(
        output.inputs["Surface"].links[0].from_socket, wrapper.inputs["Shader"]
    )
    tree.links.new(wrapper.outputs["Shader"], output.inputs["Surface"])
    # The airlight emits, and in LWIR so does every surface. As a light, each sample
    # runs the whole shader for light the rays that hit it already bring back.
    material.cycles.emission_sampling = "NONE"


def _sky_ahead(
    tree: bpy.types.NodeTree, sky: Sky, band: Band, direction: bpy.types.NodeSocket
) -> bpy.types.NodeSocket:
    """The band's sky in `direction`."""
    if band == "ir":
        return _ir_sky(tree, sky, direction)
    if sky.hdri is not None and sky.fog_weight:
        # The Sky Texture's airlight has no disc either.
        return _fogged_photo(tree, sky, direction, 0.0)

    def clear(toward: bpy.types.NodeSocket) -> bpy.types.NodeSocket:
        if sky.hdri is not None:
            return _photo(tree, sky.hdri, sky.sun_bearing_deg, toward)
        colour = _clear_eo(tree, sky, bpy.context.scene.world, toward)
        colour.node.name = "haze_sky"
        return colour

    if not sky.fog_weight:
        return clear(direction)
    return _misted(tree, sky, clear(direction), clear(_horizon(tree, direction)))


def _haze_ranges_m(far_m: float) -> np.ndarray:
    """Texel centres of the LWIR depth table, log-spaced out to the far plane."""
    texel = (np.arange(CURVE_SAMPLES) + 0.5) / CURVE_SAMPLES
    return HAZE_NEAR_M * (far_m / HAZE_NEAR_M) ** texel


def _optical_depth(
    tree: bpy.types.NodeTree, sky: Sky, band: Band, far_m: float
) -> bpy.types.NodeSocket:
    """The band's optical depth over the ray that reached the shading point.

    Limitation: exact along camera rays. LWIR takes a secondary ray, the sea reflected
    in a hull's paint, as if it ran straight out from the camera, ending at the shading
    point's distance; its true start needs the path's length so far, which a shader
    cannot read.
    """
    light_path = tree.nodes.new("ShaderNodeLightPath")
    secondary = scalar(tree, "SUBTRACT", 1.0, light_path.outputs["Is Camera Ray"])
    distance = tree.nodes.new("ShaderNodeCameraData").outputs["View Distance"]
    # Cycles starts a camera ray at the near plane; the air starts at the lens.
    start = scalar(
        tree,
        "MULTIPLY",
        scalar(tree, "SUBTRACT", distance, light_path.outputs["Ray Length"]),
        secondary,
        name="haze_start",
    )
    if band == "eo":
        path = scalar(tree, "SUBTRACT", distance, start)
        return scalar(tree, "MULTIPLY", path, sky.extinction_per_m, name="haze_beta")
    ranges = _haze_ranges_m(far_m)
    image = curve_image(
        "haze_depth", lwir.path_optical_depth(ranges, sky.visibility_km, sky.atmosphere)
    )
    span = far_m / HAZE_NEAR_M

    def at(range_m: bpy.types.NodeSocket) -> bpy.types.NodeSocket:
        # Blender's log of a negative number is 0, which is a range inside the table.
        texel = Num(tree, range_m).op("MAXIMUM", HAZE_NEAR_M).op("LOGARITHM", span)
        return lookup(tree, image, (texel - math.log(HAZE_NEAR_M, span)).socket)

    # at(0) reads the depth to HAZE_NEAR_M, not 0.
    behind = scalar(tree, "MULTIPLY", at(start), secondary)
    return scalar(tree, "SUBTRACT", at(distance), behind)


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
    return curve_image("sky_radiance", _table_sky(sky, _curve_elevation()))


def _curve_elevation() -> np.ndarray:
    """The elevation at each texel centre of a curve baked against sin(elevation)."""
    return np.arcsin((np.arange(CURVE_SAMPLES) + 0.5) / CURVE_SAMPLES)


def _table_sky(sky: Sky, elevation: np.ndarray) -> np.ndarray:
    """`lwir.sky_radiance` at `elevation`, through the aerosol."""
    radiance = lwir.sky_radiance(elevation, sky.t_air_k, sky.atmosphere)
    return _hazed(sky, radiance, elevation)


def _hazed(sky: Sky, radiance: np.ndarray, elevation: np.ndarray) -> np.ndarray:
    return aerosol.sky_radiance(
        radiance, elevation, sky.visibility_km, sky.t_air_k, sky.atmosphere
    )


def cloud_fraction(sky: Sky) -> float:
    """The share of a horizontal surface's sky the photo's cloud covers, 0 under a
    Sky Texture. Bakes the cloud images if the world has not."""
    if sky.hdri is None:
        return 0.0
    _, image = _cloud_images(sky)
    return image["cloud_fraction"]


def _cloud_images(sky: Sky) -> tuple[bpy.types.Image, bpy.types.Image]:
    """The photo's cloud share at its own resolution, and `lwir.cloudy_sky_radiance`
    under it at THERMAL_SKY_PX, both laid out as the photo."""
    assert sky.hdri is not None
    # An earlier call baked them.
    if (baked := bpy.data.images.get("cloudy_sky_radiance")) is not None:
        return bpy.data.images["cloud_share"], baked
    photo = _photo_image(sky.hdri)
    rgba = pixels(photo)
    bpy.data.images.remove(photo)
    # Bottom row first, as Blender stores an image; `skies` reads top row first.
    photo_sky = skies.library()[sky.hdri]
    share = skies.cloud(rgba[::-1, :, :3], photo_sky)[::-1]
    # Judgement: steps of 1/255 of a whole cloud, for a quarter of a float image.
    share_image = curve_image("cloud_share", share, float_buffer=False)
    h, w = share.shape
    f = max(1, w // THERMAL_SKY_PX)
    cloud = cv2.resize(share, (w // f, h // f), interpolation=cv2.INTER_AREA)
    rows = cloud.shape[0]
    elevation = np.radians((np.arange(rows) + 0.5) / rows * 180.0 - 90.0)
    # One height a row keeps cloud a function of elevation, as the world draws it:
    # radiance is close to linear over a cloud's few kelvin.
    genus = photo_sky.cloud_genus
    height = np.zeros(rows)
    if genus != "cirrus":
        _, top = lwir.CLOUD_LAYER_M[genus]
        depth = max(0.0, top - sky.cloud_base_m)
        height = skies.cloud_height_m(cloud[::-1], sky.cloud_base_m, depth)[::-1]
    radiance = _cloud_radiance(
        sky, genus, elevation[:, None], cloud, height_m=height[:, None]
    )
    image = curve_image(
        "cloudy_sky_radiance", _hazed(sky, radiance, elevation[:, None])
    )
    _bake_cloud_excess(sky, genus, elevation, height)
    # As a horizontal surface sees the sky: each row by its projected solid angle.
    up = elevation > 0
    weight = np.sin(elevation[up]) * np.cos(elevation[up])
    fraction = weight @ cloud[up].mean(axis=1) / weight.sum()
    # A weighted mean of values in [0, 1] can round past 1.
    image["cloud_fraction"] = float(np.clip(fraction, 0.0, 1.0))
    return share_image, image


def _cloudy_sky(
    tree: bpy.types.NodeTree, sky: Sky, direction: bpy.types.NodeSocket
) -> bpy.types.NodeSocket:
    """The photo's sky in `direction`: radiance is affine in the cloud share, so the
    clear sky plus the share of what a whole cloud adds."""
    assert sky.hdri is not None
    share_image, _ = _cloud_images(sky)
    share = _turned(tree, share_image, sky.hdri, sky.sun_bearing_deg, direction)
    # Linear shows the texel grid where a texel spans several pixels.
    share.node.interpolation = "Cubic"
    up = _sin_elevation(tree, direction)
    excess = lookup(tree, _cloud_excess_image(sky), up)
    return Num(tree, share).madd(excess, lookup(tree, _sky_image(sky), up)).socket


def _cloud_radiance(
    sky: Sky,
    genus: lwir.CloudGenus,
    elevation: np.ndarray,
    cloud: np.ndarray | float,
    height_m: np.ndarray,
) -> np.ndarray:
    """The sky under a `cloud` fraction of the photo's `genus`."""
    if genus == "cirrus":
        return lwir.cirrus_sky_radiance(elevation, cloud, sky.t_air_k, sky.atmosphere)
    return lwir.cloudy_sky_radiance(
        elevation,
        cloud,
        sky.cloud_base_m,
        sky.t_air_k,
        sky.atmosphere,
        height_m=height_m,
    )


def _cloud_excess_image(sky: Sky) -> bpy.types.Image:
    """What a whole cloud adds to the hazed clear sky, baked with `_cloud_images`."""
    _cloud_images(sky)
    return bpy.data.images["cloud_excess"]


def _bake_cloud_excess(
    sky: Sky, genus: lwir.CloudGenus, rows_rad: np.ndarray, height_m: np.ndarray
) -> None:
    """`_cloud_excess_image`, a whole cloud `height_m` over its base at `rows_rad`,
    baked as `_sky_image` is."""
    elevation = _curve_elevation()
    height = np.interp(elevation, rows_rad, height_m)
    whole = _cloud_radiance(sky, genus, elevation, 1.0, height_m=height)
    excess = _hazed(sky, whole, elevation) - _table_sky(sky, elevation)
    curve_image("cloud_excess", excess)


def _clouds(sky: Sky) -> sea.Clouds | None:
    """The IR world's clouds by bearing, read off the image the world baked first."""
    if sky.hdri is None:
        return None
    _, image = _cloud_images(sky)
    rgba = pixels(image)
    h = rgba.shape[0]
    # Bottom row first, as `_cloud_images` lays it out.
    elevation = np.radians((np.arange(h) + 0.5) / h * 180.0 - 90.0)
    radiance = rgba[..., 0].astype(np.float64)
    return sea.Clouds(
        elevation, radiance, _photo_turn_rad(sky.hdri, sky.sun_bearing_deg)
    )


def _sky_round(sky: Sky, clouds: sea.Clouds | None) -> sea.SkyCurve:
    """The IR world's radiance against elevation, averaged round the horizon: its clear
    sky's table, or its clouds' rows."""
    if clouds is None:
        elevation = _curve_elevation()
        return elevation, _table_sky(sky, elevation)
    return clouds.elevation, clouds.radiance.mean(axis=1)


def _thermal_sky(world: bpy.types.World, sky: Sky) -> bpy.types.World:
    """Downwelling radiance, as raw W m^-2 sr^-1: against elevation, or under the
    photo's clouds. A rough glossy ray sees none of it (`_less_sky`)."""
    tree = world.node_tree
    tree.nodes.clear()
    coord = tree.nodes.new("ShaderNodeTexCoord")
    radiance = _ir_sky(tree, sky, coord.outputs["Generated"])
    shown = scalar(tree, "SUBTRACT", 1.0, _rough_glossy(tree))
    scaled = vector(tree, "SCALE", radiance, shown)
    background = node(tree, "ShaderNodeBackground", {"Color": scaled})
    node(tree, "ShaderNodeOutputWorld", {"Surface": background.outputs["Background"]})
    # A light sample reads the world as no glossy ray, so it would add the sky the
    # sea's tables already hold.
    world.cycles.sampling_method = "NONE"
    return world


def _ir_sky(
    tree: bpy.types.NodeTree, sky: Sky, direction: bpy.types.NodeSocket
) -> bpy.types.NodeSocket:
    """The ir world's radiance in `direction`."""
    if sky.hdri is not None:
        return _cloudy_sky(tree, sky, direction)
    return lookup(tree, _sky_image(sky), _sin_elevation(tree, direction))


def _sin_elevation(
    tree: bpy.types.NodeTree, direction: bpy.types.NodeSocket
) -> bpy.types.NodeSocket:
    """A unit `direction`'s Z, held at 0 below the horizon."""
    _, _, z = separate(tree, direction)
    return clamped(tree, z)


def _rough_glossy(tree: bpy.types.NodeTree) -> bpy.types.NodeSocket:
    """1 on a glossy ray off a lobe, 0 on any other: a hull's glass and bare metal are
    sharp mirrors."""
    path = tree.nodes.new("ShaderNodeLightPath")
    sharp = scalar(tree, "SUBTRACT", 1.0, path.outputs["Is Singular Ray"])
    return scalar(tree, "MULTIPLY", path.outputs["Is Glossy Ray"], sharp)


def _less_sky(
    tree: bpy.types.NodeTree,
    sky: Sky,
    ahead: bpy.types.NodeSocket,
    shader: bpy.types.NodeSocket,
) -> bpy.types.NodeSocket:
    """`shader`, less the sky a rough glossy ray would have met `ahead` of it.

    The sea's tables integrate the sky exactly, and sampling it through the lobe is
    what leaves the sea grainy. So the world shows a rough glossy ray nothing, and what
    the ray meets instead returns only its difference from that sky: a control variate.
    Outside the haze, as the world's sky is.
    """
    behind = vector(
        tree,
        "SCALE",
        _ir_sky(tree, sky, ahead),
        scalar(tree, "MULTIPLY", _rough_glossy(tree), -1.0),
    )
    emission = node(tree, "ShaderNodeEmission", {"Color": behind}, name="haze_less_sky")
    add = node(
        tree,
        "ShaderNodeAddShader",
        {"Shader": shader, "Shader_001": emission.outputs["Emission"]},
    )
    return add.outputs["Shader"]


def sea_light(sky: Sky, band: Band) -> sea.Light:
    if band == "ir":
        clouds = _clouds(sky)
        return sea.Light(sky=_sky_round(sky, clouds), sun_up=sky.sun_up, clouds=clouds)
    if sky.sun_elevation_deg is None:
        return sea.Light()
    return sea.Light(sun=sky.sun_vector)
