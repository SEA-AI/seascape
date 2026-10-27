"""Build a Blender scene from a scenario. Nothing here renders."""

import contextlib
import math
import uuid
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from itertools import chain, combinations
from pathlib import Path
from typing import NamedTuple, get_args

import bpy
import numpy as np
from mathutils import Matrix, Vector

from seascape import faces, lwir, sea, sensor, skies, thermal, wakes, waves
from seascape.assets import Asset, Turbine, Zones, download, fetch, manifest
from seascape.blend import (
    CURVE_SAMPLES,
    SUN,
    animate,
    curve_image,
    drive,
    lookup,
    place,
    sine,
    yaw,
)
from seascape.calibration import CameraCalibration, Matrix4
from seascape.config import (
    Band,
    Drift,
    Farm,
    ImageFormat,
    Mount,
    Object,
    Outputs,
    Ownship,
    Scenario,
    Sky,
    Targets,
    substream,
)

# Judgement: glass and bare metal mirror the sky they do not emit; paint, rubber,
# fabric and wood scatter it. ponytail: dull stainless, as thermal takes it, is rougher
# than a mirror.
SPECULAR_ZONES = frozenset({"glass", "bare_metal"})

# Judgement: nearer than a lens is to anything it sees.
HAZE_NEAR_M = 0.1

# Judgement.
THERMAL_SKY_PX = 2048
# A foam memory's finest cells resolve waves a metre long, whose crests break in
# whitecaps as wide and as brief as field video measures (Korinenko & Malinovsky 2023;
# Korinenko et al. 2022).
FOAM_NEAR_CELL_M = 0.25
# Korinenko & Malinovsky 2023 filmed breakers 3.7-18.7 m long: a grid resolving no wave
# that short would lay whitecaps larger than any they measured, so the sea past it
# takes Monahan's mean.
FOAM_LONGEST_BREAKER_M = 18.7
# Numerics: how much wider each next grid's cells are, taking over where a pixel spans
# a quarter of one; rays per side of a frame its view is sampled at; a grid's widest
# side, in cells.
FOAM_LEVEL_RATIO = 4
FOAM_RAYS = 9
FOAM_MAX_CELLS = 1024


def wind_waves(scenario: Scenario) -> tuple[waves.Wave, ...]:
    sea = scenario.sea
    return waves.wind_sea(
        sea.wind_speed_mps,
        sea.wind_from_deg,
        sea.fetch_m,
        scenario.outputs.period_s,
        substream(scenario.seed, "sea/surface"),
    )


def swell_waves(scenario: Scenario) -> tuple[waves.Wave, ...]:
    swell = scenario.sea.swell
    if swell is None:
        return ()
    return waves.swell(
        swell.height_m,
        swell.period_s,
        swell.from_deg,
        scenario.outputs.period_s,
        substream(scenario.seed, "sea/swell"),
    )


def wave_field(scenario: Scenario) -> tuple[waves.Wave, ...]:
    return wind_waves(scenario) + swell_waves(scenario)


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
    kept = sea._math(group, "EXPONENT", sea._math(group, "MULTIPLY", depth, -1.0))
    mix = group.nodes.new("ShaderNodeMixShader")
    mix.name = "haze_mix"
    link(sea._math(group, "SUBTRACT", 1.0, kept), mix.inputs["Fac"])
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
        return sea._math(
            tree, "MULTIPLY", length, sky.extinction_per_m, name="haze_beta"
        )
    ranges = _haze_ranges_m(far_m)
    image = curve_image(
        "haze_depth", lwir.path_optical_depth(ranges, sky.visibility_km, sky.atmosphere)
    )
    span = far_m / HAZE_NEAR_M

    def at(range_m: bpy.types.NodeSocket) -> bpy.types.NodeSocket:
        # Blender's log of a negative number is 0, which is a range inside the table.
        near = sea._math(tree, "MAXIMUM", range_m, HAZE_NEAR_M)
        texel = sea._math(tree, "LOGARITHM", near, span)
        texel = sea._math(tree, "SUBTRACT", texel, math.log(HAZE_NEAR_M, span))
        return lookup(tree, image, texel)

    distance = tree.nodes.new("ShaderNodeVectorMath")
    distance.operation = "LENGTH"
    camera = tree.nodes.new("ShaderNodeTexCoord").outputs["Camera"]
    tree.links.new(camera, distance.inputs["Vector"])
    start = sea._math(tree, "SUBTRACT", distance.outputs["Value"], length)
    return sea._math(tree, "SUBTRACT", at(distance.outputs["Value"]), at(start))


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
    belongs to rotations, and `_pose` places by the same sin/cos."""
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


def _thermal_skin(name: str, t_k: float, sky: Sky) -> bpy.types.Material:
    """eps of a painted hull emitted, the remaining 1 - eps reflected from the sky.

    Diffuse: flat marine paint is near-Lambertian in this band.
    """
    material = bpy.data.materials.new(name)
    tree = material.node_tree
    tree.nodes.clear()
    emissivity = thermal.ZONES["paint_hull"].emissivity
    skin = _reflecting(tree, _sunlit_emission(tree, t_k, sky), emissivity, None)
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    tree.links.new(skin, output.inputs["Surface"])
    return material


def _sunlit_emission(
    tree: bpy.types.NodeTree, t_k: float, sky: Sky
) -> bpy.types.NodeSocket:
    """Emission graded from shaded to sunlit by Lambert's cosine on the real sun.

    Two emissions mixed by `max(0, n . sun)`, so both ends are the exact band radiance
    and only the middle interpolates: no shader node can evaluate the Planck integral
    at the blended temperature.
    """
    shaded = tree.nodes.new("ShaderNodeEmission")
    shaded.inputs["Strength"].default_value = lwir.band_radiance(t_k)
    if sky.sun_elevation_deg is None:
        return shaded.outputs["Emission"]
    sunlit = tree.nodes.new("ShaderNodeEmission")
    sunlit.inputs["Strength"].default_value = lwir.band_radiance(t_k + sky.solar_gain_k)

    geometry = tree.nodes.new("ShaderNodeNewGeometry")
    facing = tree.nodes.new("ShaderNodeVectorMath")
    facing.operation = "DOT_PRODUCT"
    facing.inputs[1].default_value = _sun_vector(sky)
    # Clamped, or a surface turned away from the sun would cool below shaded.
    lit = tree.nodes.new("ShaderNodeMath")
    lit.operation = "MAXIMUM"
    lit.inputs[1].default_value = 0.0

    grade = tree.nodes.new("ShaderNodeMixShader")
    link = tree.links.new
    link(geometry.outputs["Normal"], facing.inputs[0])
    link(facing.outputs["Value"], lit.inputs[0])
    link(lit.outputs["Value"], grade.inputs["Factor"])
    link(shaded.outputs["Emission"], grade.inputs[1])
    link(sunlit.outputs["Emission"], grade.inputs[2])
    return grade.outputs["Shader"]


class _Heat(NamedTuple):
    """What a hull's zones settle against, and the water its wetted faces sit in."""

    env: thermal.Environment
    t_water_k: float
    sun: tuple[float, float, float] | None

    def stopped(self, minutes: float) -> "_Heat":
        return self._replace(env=self.env.stopped(minutes * 60.0))


def _heat(scenario: Scenario) -> _Heat:
    sky = scenario.sky
    elevation = sky.sun_elevation_deg
    clouds = 0.0
    if sky.hdri is not None:
        clouds = bpy.data.images["cloudy_sky_radiance"]["cloud_fraction"]
    env = thermal.Environment(
        t_air_k=sky.t_air_k,
        t_sea_k=scenario.sea.t_skin_k(sky.sun_up),
        # ponytail: the true wind; a hull underway convects in its own way through the
        # air, and every hull shares one skin per zone.
        wind_m_s=scenario.sea.wind_speed_mps,
        sun_elevation_rad=None if elevation is None else math.radians(elevation),
        cloud_fraction=clouds,
    )
    sun = _sun_vector(sky) if elevation is not None and elevation > 0 else None
    return _Heat(env, scenario.sea.t_sea_k, sun)


def _zone_skin(
    zone_name: thermal.ZoneName, heat: _Heat, paint_absorptance: float | None = None
) -> bpy.types.Material:
    """A zone's eps emitted at its heat balance and 1 - eps reflected, shared by every
    hull: each face looks up its temperature by max(0, n . s) and the normal's z.
    Below the waterline, the bulk water's. `paint_absorptance` repaints the zones of
    `thermal.PAINTED`.

    No face shades another: a shadow on a hull is narrower than a pixel at the ranges
    a camera finds it, and the balance is steady, so a shadow would be as sharp and
    as deep as one that had sat for hours."""
    zone = thermal.ZONES[zone_name]
    name = f"thermal_{zone_name}"
    if paint_absorptance is not None and zone_name in thermal.PAINTED:
        # Hulls a hundredth apart share a skin, built at the absorptance it is named.
        absorptance = round(paint_absorptance, 2)
        zone = replace(zone, absorptance=absorptance)
        name += f"_a{absorptance:.2f}"
    if zone.interior in ("engine_room", "exhaust"):
        # Hulls whose engines stopped at different times differ here.
        name += f"_{heat.env.interior_k(zone.interior):.1f}K"
    if (found := bpy.data.materials.get(name)) is not None:
        return found
    material = bpy.data.materials.new(name)
    tree = material.node_tree
    tree.nodes.clear()
    link = tree.links.new

    sun_cos, up, table = thermal.temperature_table_k(zone, heat.env)
    image = curve_image(name, np.vectorize(lwir.band_radiance)(table))
    geometry = tree.nodes.new("ShaderNodeNewGeometry")
    normal = tree.nodes.new("ShaderNodeSeparateXYZ")
    link(geometry.outputs["Normal"], normal.inputs["Vector"])
    # At texel centres: the lookup interpolates between them.
    y = sea._math(
        tree,
        "MULTIPLY_ADD",
        sea._math(tree, "MULTIPLY_ADD", normal.outputs["Z"], 0.5, 0.5),
        (len(up) - 1) / len(up),
        0.5 / len(up),
    )
    if heat.sun is None:
        x = tree.nodes.new("ShaderNodeValue")
        x.outputs["Value"].default_value = 0.5 / len(sun_cos)
        at = x.outputs["Value"]
    else:
        facing = tree.nodes.new("ShaderNodeVectorMath")
        facing.operation = "DOT_PRODUCT"
        facing.inputs["Vector_001"].default_value = heat.sun
        link(geometry.outputs["Normal"], facing.inputs["Vector"])
        lit = sea._math(tree, "MAXIMUM", facing.outputs["Value"], 0.0)
        at = sea._math(
            tree,
            "MULTIPLY_ADD",
            lit,
            (len(sun_cos) - 1) / len(sun_cos),
            0.5 / len(sun_cos),
        )
    emission = tree.nodes.new("ShaderNodeEmission")
    link(lookup(tree, image, at, y), emission.inputs["Color"])
    skin = _reflecting(tree, emission.outputs["Emission"], zone.emissivity, zone_name)

    water = tree.nodes.new("ShaderNodeEmission")
    water.inputs["Strength"].default_value = lwir.band_radiance(heat.t_water_k)
    wetted = _reflecting(tree, water.outputs["Emission"], thermal.SEA_EMISSIVITY, None)
    # The fit puts the waterline at z = 0; the join keeps the hull's frame in metres.
    coords = tree.nodes.new("ShaderNodeTexCoord")
    height = tree.nodes.new("ShaderNodeSeparateXYZ")
    link(coords.outputs["Object"], height.inputs["Vector"])
    under = sea._math(tree, "LESS_THAN", height.outputs["Z"], 0.0)
    mix = tree.nodes.new("ShaderNodeMixShader")
    link(under, mix.inputs["Factor"])
    link(skin, mix.inputs["Shader"])
    link(wetted, mix.inputs["Shader_001"])
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    link(mix.outputs["Shader"], output.inputs["Surface"])
    return material


def _reflecting(
    tree: bpy.types.NodeTree,
    emission: bpy.types.NodeSocket,
    emissivity: float,
    zone_name: str | None,
) -> bpy.types.NodeSocket:
    """`emission` weighted by `emissivity`, the rest a white reflection: the Mix Shader
    already weights it by 1 - eps, so a grey would absorb part of it twice."""
    if zone_name in SPECULAR_ZONES:
        scatter = tree.nodes.new("ShaderNodeBsdfGlossy")
        scatter.distribution = "GGX"
        scatter.inputs["Roughness"].default_value = 0.0
    else:
        scatter = tree.nodes.new("ShaderNodeBsdfDiffuse")
    scatter.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    mix = tree.nodes.new("ShaderNodeMixShader")
    mix.inputs["Factor"].default_value = emissivity
    tree.links.new(scatter.outputs["BSDF"], mix.inputs["Shader"])
    tree.links.new(emission, mix.inputs["Shader_001"])
    return mix.outputs["Shader"]


def _joined(
    parts: list[bpy.types.Object], frame: Matrix | None = None
) -> bpy.types.Object:
    """Every mesh under `parts` as one object at `frame`, its vertices in metres about
    it; the default, the fitted origin, puts the waterline at z = 0."""
    everything = [o for part in parts for o in [part, *part.children_recursive]]
    meshes = [o for o in everything if o.type == "MESH"]
    others = [o for o in everything if o.type != "MESH"]
    bpy.context.view_layer.update()
    # A skinned mesh takes its pose from an armature modifier, which the join keeps
    # pointing at an armature removed below: bake the pose in first.
    depsgraph = bpy.context.evaluated_depsgraph_get()
    posed = {
        obj: bpy.data.meshes.new_from_object(
            obj.evaluated_get(depsgraph),
            preserve_all_data_layers=True,
            depsgraph=depsgraph,
        )
        for obj in meshes
        if obj.modifiers
    }
    for obj, mesh in posed.items():
        rest = obj.data
        obj.modifiers.clear()
        obj.data = mesh
        if not rest.users:
            bpy.data.meshes.remove(rest)
    with bpy.context.temp_override(
        active_object=meshes[0], selected_editable_objects=meshes
    ):
        bpy.ops.object.join()
    hull = meshes[0]
    frame = Matrix.Identity(4) if frame is None else frame
    baked = frame.inverted() @ hull.matrix_world
    hull.parent = None
    hull.matrix_parent_inverse.identity()
    hull.data.transform(baked)
    if baked.determinant() < 0:  # a mirrored part turns inside out
        hull.data.flip_normals()
    hull.matrix_basis = frame
    for obj in others:
        bpy.data.objects.remove(obj)
    return hull


def _zone_slots(name: str, meshes: list[bpy.types.Object], zones: Zones | None) -> None:
    """Give each face its zone as a material slot, one placeholder per zone, which the
    join merges. Unlabelled, every face is `paint_hull`.

    The asset's own materials go: they are albedo, which says nothing about 8-14 um,
    and a later import sharing their names would be renamed past its zone labels.
    """
    vocabulary = list(get_args(thermal.ZoneName.__value__))
    found = faces.found(meshes)
    if zones is not None:
        # A name another object in the file already took renumbers the islands.
        faces.check(name, found, zones)
    per_face = (
        faces.face_zones(found, zones)
        if zones is not None
        else [
            np.full(len(obj.data.polygons), vocabulary.index("paint_hull"))
            for obj in found.objects
        ]
    )
    originals = {s.material for obj in found.objects for s in obj.material_slots}
    for obj, zone in zip(found.objects, per_face, strict=True):
        present, index = np.unique(zone, return_inverse=True)
        obj.data.materials.clear()
        for z in present:
            name = f"zone_{vocabulary[z]}"
            placeholder = bpy.data.materials.get(name) or bpy.data.materials.new(name)
            obj.data.materials.append(placeholder)
        obj.data.polygons.foreach_set("material_index", index.astype(np.int32))
    for material in originals - {None}:
        bpy.data.materials.remove(material)


def _unslotted(joined: list[bpy.types.Object]) -> None:
    """Record each joined object's zone per slot in its `["zones"]` and empty the
    slots; each copy fills them with its own skin."""
    placeholders = set()
    for obj in joined:
        mesh = obj.data
        obj["zones"] = [m.name.removeprefix("zone_") for m in mesh.materials]
        placeholders.update(mesh.materials)
        for i in range(len(mesh.materials)):
            mesh.materials[i] = None
    for material in placeholders:
        bpy.data.materials.remove(material)


class _RigObjects(NamedTuple):
    root: bpy.types.Object
    rigs: dict[str, bpy.types.Object]
    cameras: dict[str, bpy.types.Object]


def _rigs(scenario: Scenario, far_m: float) -> _RigObjects:
    root = bpy.data.objects.new("rigs", None)
    bpy.context.collection.objects.link(root)
    place(root, 0.0, 0.0, 0.0)

    rigs: dict[str, bpy.types.Object] = {}
    for name, rig in scenario.rigs.items():
        # Blender takes an inverted frustum without complaint and renders nothing.
        if rig.near_clip_m >= far_m:
            raise ValueError(
                f"{name}: near clip {rig.near_clip_m} m is past the far plane at "
                f"{far_m:.0f} m"
            )
        empty = bpy.data.objects.new(f"rig_{name}", None)
        bpy.context.collection.objects.link(empty)
        empty.parent = root
        place(empty, rig.offset_x_m, rig.offset_y_m, rig.height_m)
        # XYZ euler is Rz @ Ry @ Rx: yaw, then pitch about the rig's own transverse
        # axis, so a pitched rig rolls the horizon of its off-axis cameras.
        empty.rotation_euler = (math.radians(rig.pitch_deg), 0.0, yaw(rig.yaw_deg))
        rigs[name] = empty

    cameras: dict[str, bpy.types.Object] = {}
    for mount in scenario.mounts:
        data = bpy.data.cameras.new(mount.name)
        # AUTO fits the field of view to whichever image dimension is larger, so a
        # portrait sensor would silently reinterpret hfov as a vertical angle.
        data.sensor_fit = "HORIZONTAL"
        data.angle_x = math.radians(mount.camera.hfov_deg)
        data.clip_start = mount.rig.near_clip_m
        # The default 1000 m renders a target past it as sky, and the clip boundary
        # reads as the horizon. Nothing warns.
        data.clip_end = far_m
        camera = bpy.data.objects.new(data.name, data)
        bpy.context.collection.objects.link(camera)
        camera.parent = rigs[mount.rig_name]
        camera.rotation_mode = "XYZ"
        # A camera looks down its local -Z; +90 deg about X aims it at the horizon.
        camera.rotation_euler = (
            math.radians(90.0 + mount.camera.pitch_deg),
            0.0,
            yaw(mount.camera.yaw_deg),
        )
        cameras[mount.name] = camera
    return _RigObjects(root, rigs, cameras)


def boresight_deg(camera: bpy.types.Object) -> tuple[float, float]:
    """Bearing and elevation a built camera points at, in degrees, in the world.
    `matrix_world` is stale until the depsgraph runs, so build first.
    """
    forward = camera.matrix_world.to_3x3() @ Vector((0.0, 0.0, -1.0))
    forward.normalize()
    return (
        math.degrees(math.atan2(forward.x, forward.y)),
        math.degrees(math.asin(min(1.0, max(-1.0, forward.z)))),
    )


# Blender's camera looks down -Z with +Y up; OpenCV's looks down +Z with +Y down.
_BLENDER_TO_CV = Matrix.Diagonal((1.0, -1.0, -1.0, 1.0))


@dataclass(frozen=True)
class Built:
    """What `build` made, by role, valid until the next build. A name in `bpy.data`
    belongs to whoever took it first, and an imported asset can carry any name."""

    vessel: bpy.types.Object
    rigs: dict[str, bpy.types.Object]
    cameras: dict[str, bpy.types.Object]
    targets: dict[str, list[bpy.types.Object]]  # by asset


def calibrate(built: Built, mount: Mount, image: str) -> CameraCalibration:
    """A built camera's geometry, read off the scene."""
    camera = built.cameras[mount.name]
    world = camera.matrix_world @ _BLENDER_TO_CV
    vessel = built.vessel.matrix_world
    rig = built.rigs[mount.rig_name].matrix_world
    width, height = mount.camera.width_px, mount.camera.height_px
    f = (width / 2) / math.tan(camera.data.angle_x / 2)
    return CameraCalibration(
        name=mount.name,
        band=mount.camera.band,
        image=image,
        rig=mount.rig_name,
        model=mount.rig.model,
        width_px=width,
        height_px=height,
        # Blender's frame spans pixel edges, so its centre is half a pixel past
        # OpenCV's.
        K=((f, 0.0, (width - 1) / 2), (0.0, f, (height - 1) / 2), (0.0, 0.0, 1.0)),
        extrinsics={
            "world": _rows(world),
            "vessel": _rows(vessel.inverted() @ world),
            "rig": _rows(rig.inverted() @ world),
        },
    )


def heading_rad(anchor: bpy.types.Object) -> float:
    """Where a hull's bow points, clockwise from north: bow to +Y, as fitted."""
    bow = anchor.matrix_world.to_3x3() @ Vector((0.0, 1.0, 0.0))
    return math.atan2(bow.x, bow.y)


def waterline_m(anchor: bpy.types.Object) -> np.ndarray:
    """Where a hull's edges cross its waterline, as world east and north, (N, 2).

    The waterline is z = 0 in the attitude's frame, which heaves, pitches and rolls
    with the waves: `_fit` puts the keel a draught below it.
    """
    # ponytail: the crossings, not the segments between them, so the nearest one to a
    # camera at range r is up to side^2 / 2r further than a side facing it.
    (attitude,) = anchor.children
    to_world = np.array(attitude.matrix_world)
    to_hull = np.linalg.inv(to_world)
    crossings = []
    for part in _meshes([anchor]):
        mesh = part.data
        local = np.empty(3 * len(mesh.vertices), dtype=np.float32)
        mesh.vertices.foreach_get("co", local)
        ends = np.empty(2 * len(mesh.edges), dtype=np.int32)
        mesh.edges.foreach_get("vertices", ends)
        m = to_hull @ np.array(part.matrix_world)
        hull = local.reshape(-1, 3) @ m[:3, :3].T + m[:3, 3]
        a, b = hull[ends.reshape(-1, 2)].transpose(1, 0, 2)
        cross = a[:, 2] * b[:, 2] < 0
        a, b = a[cross], b[cross]
        t = -a[:, 2] / (b[:, 2] - a[:, 2])
        at = a + t[:, None] * (b - a)
        crossings.append((at @ to_world[:3, :3].T + to_world[:3, 3])[:, :2])
    return np.concatenate(crossings)


def _rows(m: Matrix) -> Matrix4:
    x, y, z, w = map(tuple, m)
    return x, y, z, w


def _corners(
    objects: Iterable[bpy.types.Object], into: Matrix | None = None
) -> list[Vector]:
    """Corners of the axis-aligned box around the vertices of the meshes in
    `objects`, in the frame `into` maps world space to (default: world).

    Vertices, not `bound_box`: a part turned off its parent's axes carries its own box
    turned with it, which can be far wider than the part.
    """
    depsgraph = bpy.context.evaluated_depsgraph_get()
    points = []
    for obj in objects:
        if obj.type != "MESH":
            continue
        # Evaluated, so an armature's pose and the modifiers move the vertices.
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        co = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", co)
        evaluated.to_mesh_clear()
        if not co.size:
            continue
        to = np.array((into or Matrix()) @ obj.matrix_world)
        moved = co.reshape(-1, 3) @ to[:3, :3].T + to[:3, 3]
        points += [moved.min(axis=0), moved.max(axis=0)]
    if not points:
        return []
    low, high = np.min(points, axis=0), np.max(points, axis=0)
    return [
        Vector((x, y, z))
        for x in (low[0], high[0])
        for y in (low[1], high[1])
        for z in (low[2], high[2])
    ]


def _fit(corners: Iterable[Vector], asset: Asset) -> Matrix:
    """Bow to +Y, scaled to the manifest size, centred, keel at the draught.

    Turned first so a hull's length is measured bow to stern whatever the authored axis.
    """
    turn = Matrix.Rotation(yaw(-asset.bow_deg), 4, "Z")
    axes = list(zip(*(turn @ c for c in corners), strict=True))
    low, high = Vector([min(a) for a in axes]), Vector([max(a) for a in axes])
    scale = asset.scale(tuple(high - low))
    low, high = low * scale, high * scale
    east, north = asset.centre(tuple(low), tuple(high))
    return (
        Matrix.Translation((-east, -north, -low.z - asset.draught_m))
        @ Matrix.Scale(scale, 4)
        @ turn
    )


def _meshes(parts: Iterable[bpy.types.Object]) -> list[bpy.types.Object]:
    return [
        obj
        for part in parts
        for obj in [part, *part.children_recursive]
        if obj.type == "MESH"
    ]


def _load(path: Path) -> set[bpy.types.Object]:
    """Import a mesh file; returns what it added. Caches the parse as a .blend beside
    the file and appends from it on later calls."""
    stat = path.stat()
    version = ".".join(map(str, bpy.app.version))
    imported = path.with_name(
        f"{path.name}.{stat.st_size}-{stat.st_mtime_ns}.{version}.blend"
    )
    if imported.exists():
        try:
            with bpy.data.libraries.load(str(imported)) as (found, into):
                into.objects = found.objects
            if None in into.objects:  # Blender leaves None for what it cannot read
                raise RuntimeError(f"{imported} loaded partly")
        except (OSError, RuntimeError):
            with contextlib.suppress(OSError):
                imported.unlink(missing_ok=True)
        else:
            for obj in into.objects:
                bpy.context.scene.collection.objects.link(obj)
            # The fit reads matrix_world, which an append leaves stale.
            bpy.context.view_layer.update()
            return set(into.objects)
    objects = _parse(path)
    # A killed write must not leave a truncated file under the cache name.
    part = imported.with_name(f"{imported.name}.{uuid.uuid4().hex}.part")
    try:
        bpy.data.libraries.write(
            str(part), objects, path_remap="ABSOLUTE", compress=False
        )
        part.replace(imported)
    except (OSError, RuntimeError):  # a read-only or full cache is only slower
        part.unlink(missing_ok=True)
    return objects


def _parse(path: Path) -> set[bpy.types.Object]:
    before = set(bpy.data.objects)
    gltf = bpy.ops.import_scene.gltf
    importer = {".fbx": bpy.ops.import_scene.fbx, ".glb": gltf, ".gltf": gltf}
    importer[path.suffix](filepath=str(path))
    imported = set(bpy.data.objects) - before
    # A Poly glTF brings its viewer's camera and lights, which would light the scene.
    # glTF nodes outside the file's scene land in an excluded collection: matrix_world
    # never updates there, and copies would render.
    shown = bpy.context.view_layer.objects
    # A skinned glTF's bones are drawn with a sphere the importer adds to the scene,
    # where it renders and enters the fit.
    shapes = {
        bone.custom_shape
        for o in imported
        if o.type == "ARMATURE"
        for bone in o.pose.bones
    }
    for obj in [
        o
        for o in imported
        if o.type in {"CAMERA", "LIGHT"} or o.name not in shown or o in shapes
    ]:
        imported.remove(obj)
        bpy.data.objects.remove(obj)
    return imported


def _import(name: str, band: Band) -> list[bpy.types.Object]:
    """Import an asset and fit it; returns the root parts.

    An asset arrives in its author's units, off-origin, in many parts.
    """
    imported = _load(fetch(name))
    asset = manifest()[name]
    # Measure everything, move the roots. An FBX keeps meshes under empties, and
    # measuring only the roots would leave them out of the fit.
    parts = [o for o in imported if o.parent is None]

    corners = _corners(imported)
    fit = _fit(corners, asset)
    for part in parts:
        part.matrix_world = fit @ part.matrix_world
    rotor = None
    if isinstance(asset, Turbine):
        named = [o for o in imported if o.name == asset.rotor.node]
        if len(named) != 1:
            raise ValueError(f"{name}: {len(named)} objects named {asset.rotor.node!r}")
        (rotor,) = named
    # Fitted bow to +Y, so the beam is the width along x and the length along y.
    fitted = [fit @ c for c in corners]
    if band == "ir":
        _zone_slots(name, _meshes(parts), asset.zones)
        if rotor is not None:
            # Joined apart from the rest, so it still turns about its hub.
            bpy.context.view_layer.update()
            hub = rotor.matrix_world.copy()
            rotor.parent = None
            rotor.matrix_world = hub
            rotor = _joined([rotor], hub)
        parts = [_joined(parts)]
        # Frees the import's object name, so the next asset's objects keep theirs,
        # which number its islands.
        parts[0].name = parts[0].data.name = f"{name}_hull"
        if rotor is not None:
            rotor.name = rotor.data.name = f"{name}_rotor"
            parts.append(rotor)
        _unslotted(parts)
    if rotor is not None:
        rotor["rotor"] = True
    parts[0]["beam_m"] = max(c.x for c in fitted) - min(c.x for c in fitted)
    parts[0]["length_m"] = max(c.y for c in fitted) - min(c.y for c in fitted)
    # Above the waterline, which the fit puts at z = 0.
    parts[0]["height_m"] = max(c.z for c in fitted)
    return parts


def _images(tree: bpy.types.NodeTree) -> set[bpy.types.Image]:
    found = set()
    for node in tree.nodes:
        if node.type == "TEX_IMAGE" and node.image:
            found.add(node.image)
        elif node.type == "GROUP" and node.node_tree:
            found |= _images(node.node_tree)
    return found


def measure(path: Path) -> tuple[int, tuple[int, ...]]:
    """A mesh file's triangles and the longest side of each image, as imported."""
    return measured(str(path), [obj for obj in _load(path) if obj.type == "MESH"])


def measured(name: str, meshes: list[bpy.types.Object]) -> tuple[int, tuple[int, ...]]:
    """`measure` of `name`'s meshes, already imported."""
    for mesh in meshes:
        mesh.data.calc_loop_triangles()
    images = set().union(
        *(
            _images(material.node_tree)
            for mesh in meshes
            for material in mesh.data.materials
            if material and material.node_tree
        )
    )
    if missing := sorted(image.name for image in images if 0 in image.size):
        raise ValueError(f"{name}: no pixels in {missing}")
    return (
        sum(len(mesh.data.loop_triangles) for mesh in meshes),
        tuple(sorted((max(image.size) for image in images), reverse=True)),
    )


def _vessel(
    name: str,
    t_k: float | None,
    engine_off_min: float,
    paint_absorptance: float | None,
    sky: Sky,
    heat: _Heat | None,
    hulls: dict[str, list[bpy.types.Object]],
) -> bpy.types.Object:
    """A hull fitted and anchored at the origin under an empty. `heat` is the LWIR
    build's, and None in EO.

    `hulls` holds each asset's first import for the rest of the build: the FBX
    importer slows with every material already in the file.
    """
    if name in hulls:
        parts = [_copy_tree(part, None) for part in hulls[name]]
    else:
        parts = hulls[name] = _import(name, "eo" if heat is None else "ir")

    if heat is not None:
        heat = heat.stopped(engine_off_min)
        uniform = None if t_k is None else _thermal_skin(f"{name}_ir", t_k, sky)
        for mesh in _meshes(parts):
            for slot, zone in zip(mesh.material_slots, mesh["zones"], strict=True):
                # Copies share mesh data, and a hull given its own t_k differs.
                slot.link = "OBJECT"
                slot.material = uniform or _zone_skin(zone, heat, paint_absorptance)

    anchor = bpy.data.objects.new(name, None)
    for size in ("beam_m", "length_m", "height_m"):
        anchor[size] = parts[0][size]
    anchor["engine_off_min"] = engine_off_min
    # Pitch and roll go here, so the anchor keeps the pose labels read.
    attitude = bpy.data.objects.new(f"{name}_attitude", None)
    for obj in (anchor, attitude):
        bpy.context.collection.objects.link(obj)
    attitude.parent = anchor
    for part in parts:
        part.parent = attitude
    place(anchor, 0.0, 0.0, 0.0)
    place(attitude, 0.0, 0.0, 0.0)
    # YXZ euler is Rz @ Rx @ Ry: roll about the keel, innermost.
    attitude.rotation_mode = "YXZ"
    return anchor


def _ownship(
    ownship: Ownship,
    sky: Sky,
    heat: _Heat | None,
    rigs: bpy.types.Object,
    hulls: dict[str, list[bpy.types.Object]],
    outputs: Outputs,
    tall: tuple[waves.Wave, ...],
) -> bpy.types.Object:
    """At the origin, bow to +Y, carrying the rigs: their offsets are in this frame.
    It heaves on the `tall` waves, which are geometry, and its script."""
    if ownship.asset is None:
        anchor = bpy.data.objects.new("ownship", None)
        bpy.context.collection.objects.link(anchor)
        place(anchor, 0.0, 0.0, 0.0)
    else:
        anchor = _vessel(ownship.asset, ownship.t_k, 0.0, None, sky, heat, hulls)
    anchor.name = "ownship"
    rigs.parent = anchor
    # YXZ euler is Rz @ Rx @ Ry: roll about the keel, innermost.
    anchor.rotation_mode = "YXZ"
    anchor.rotation_euler = (
        math.radians(ownship.pitch_deg),
        math.radians(ownship.roll_deg),
        0.0,
    )
    for index, mean_deg, swing in (
        (0, ownship.pitch_deg, ownship.pitch),
        (1, ownship.roll_deg, ownship.roll),
    ):
        if swing is not None:
            value_at = sine(
                math.radians(mean_deg),
                math.radians(swing.amplitude_deg),
                outputs.period_s(swing.period_s),
            )
            animate(anchor, "rotation_euler", outputs.times_s, value_at, index)
    scripted = sine(anchor.location.z, 0.0, 1.0)
    if (heave := ownship.heave) is not None:
        scripted = sine(
            anchor.location.z, heave.amplitude_m, outputs.period_s(heave.period_s)
        )
    if heave is not None or tall:
        # Bow to +Y, a heading of 0; a rig without a hull rides the water under it.
        length_m, beam_m = _waterplane(anchor) if ownship.asset else (0.0, 0.0)

        def value_at(t_s: float) -> float:
            return scripted(t_s) + waves.heave_m(
                tall, 0.0, 0.0, 0.0, length_m, beam_m, t_s
            )

        animate(anchor, "location", outputs.times_s, value_at, index=2)
    return anchor


def _pose(
    anchor: bpy.types.Object,
    range_m: float,
    bearing_deg: float,
    heading_deg: float,
    speed_mps: float,
    drift: Drift | None,
    radius_m: float,
    outputs: Outputs,
) -> None:
    """Put a hull on the sea at a bearing and range at t = 0, underway along its
    heading and drifting about that pose.

    A hull left at z = 0 flies above the curved sea.

    Not tilted to the local vertical: at any range the sea reaches, range / R moves a
    hull's ends far less than its draught.
    """
    bearing, heading = math.radians(bearing_deg), math.radians(heading_deg)
    sway = surge = sine(0.0, 0.0, 1.0)
    if drift is not None:
        period_s = outputs.period_s(drift.period_s)
        sway = sine(0.0, drift.sway_m, period_s)
        surge = sine(0.0, drift.surge_m, period_s / 2.0)

    def at(t_s: float) -> tuple[float, float, float]:
        along, across = speed_mps * t_s + surge(t_s), sway(t_s)
        # Starboard of the heading is (cos, -sin).
        east = (
            range_m * math.sin(bearing)
            + along * math.sin(heading)
            + across * math.cos(heading)
        )
        north = (
            range_m * math.cos(bearing)
            + along * math.cos(heading)
            - across * math.sin(heading)
        )
        return east, north, waves.sea_z_m(east, north, radius_m)

    place(anchor, *at(0.0))
    animate(anchor, "location", outputs.times_s, at)
    anchor.rotation_euler = (0.0, 0.0, yaw(heading_deg))


def _orbit(
    anchor: bpy.types.Object,
    range_m: float,
    bearing_deg: float,
    lap_s: float,
    radius_m: float,
    times_s: Sequence[float],
) -> None:
    def bearing_at_deg(t_s: float) -> float:
        return bearing_deg + 360.0 * t_s / lap_s

    def at(t_s: float) -> tuple[float, float, float]:
        bearing = math.radians(bearing_at_deg(t_s))
        east, north = range_m * math.sin(bearing), range_m * math.cos(bearing)
        return east, north, waves.sea_z_m(east, north, radius_m)

    place(anchor, *at(0.0))
    animate(anchor, "location", times_s, at)
    # Clockwise, the bow runs a right angle ahead of the bearing.
    animate(
        anchor,
        "rotation_euler",
        times_s,
        lambda t: yaw(bearing_at_deg(t) + 90),
        2,
    )


def _copy_tree(
    obj: bpy.types.Object, parent: bpy.types.Object | None
) -> bpy.types.Object:
    """Duplicate an object tree. `copy()` shares `data`, so N targets cost one mesh.

    The parent inverse copies too, so the tree keeps its shape rather than flattening.
    """
    clone = obj.copy()
    bpy.context.collection.objects.link(clone)
    clone.parent = parent
    for child in obj.children:
        _copy_tree(child, clone)
    return clone


def _targets(
    spec: Targets,
    radius_m: float,
    sky: Sky,
    heat: _Heat | None,
    hulls: dict[str, list[bpy.types.Object]],
    outputs: Outputs,
    trails: list[wakes.Wake],
) -> dict[str, list[bpy.types.Object]]:
    """The ring's hulls, by asset."""
    anchors: dict[str, list[bpy.types.Object]] = {}
    for i, (asset, bearing_deg, heading_deg) in enumerate(spec.poses()):
        anchor = _vessel(
            asset,
            spec.t_k,
            spec.engine_off_min,
            spec.paint_absorptance,
            sky,
            heat,
            hulls,
        )
        anchor.name = f"target_{i}"
        _pose(
            anchor,
            spec.range_m,
            bearing_deg,
            heading_deg,
            spec.speed_mps,
            spec.drift,
            radius_m,
            outputs,
        )
        trails.extend(_wake(spec, anchor, bearing_deg, heading_deg))
        anchors.setdefault(asset, []).append(anchor)
    return anchors


def _object(
    spec: Object,
    radius_m: float,
    sky: Sky,
    heat: _Heat | None,
    hulls: dict[str, list[bpy.types.Object]],
    outputs: Outputs,
    trails: list[wakes.Wake],
) -> list[bpy.types.Object]:
    """`spec`'s hulls, each wake under way added to `trails`."""
    anchor = _vessel(
        spec.asset,
        spec.t_k,
        spec.engine_off_min,
        spec.paint_absorptance,
        sky,
        heat,
        hulls,
    )
    orbit = spec.orbit
    if orbit is None:
        _pose(
            anchor,
            spec.range_m,
            spec.bearing_deg,
            spec.course_deg,
            spec.speed_mps,
            spec.drift,
            radius_m,
            outputs,
        )
        trails.extend(_wake(spec, anchor, spec.bearing_deg, spec.course_deg))
        return [anchor]
    anchors = [anchor, *(_copy_tree(anchor, None) for _ in range(orbit.count - 1))]
    lap_s = orbit.count * outputs.period_s(orbit.period_s / orbit.count)
    for i, hull in enumerate(anchors):
        bearing_deg = spec.bearing_deg + 360.0 * i / orbit.count
        _orbit(hull, spec.range_m, bearing_deg, lap_s, radius_m, outputs.times_s)
    trails.append(
        wakes.Wake(
            speed_mps=2 * math.pi * spec.range_m / lap_s,
            length_m=anchor["length_m"],
            beam_m=anchor["beam_m"],
            orbit_m=spec.range_m,
            start_bearing_rad=math.radians(spec.bearing_deg),
            lap_s=lap_s,
            count=orbit.count,
        )
    )
    return anchors


def _turbine(
    farm: Farm,
    range_m: float,
    bearing_deg: float,
    scenario: Scenario,
    radius_m: float,
    heat: _Heat | None,
    hulls: dict[str, list[bpy.types.Object]],
) -> bpy.types.Object:
    """A turbine facing the wind."""
    anchor = _vessel(
        farm.asset, None, 0.0, farm.paint_absorptance, scenario.sky, heat, hulls
    )
    _pose(
        anchor,
        range_m,
        bearing_deg,
        scenario.sea.wind_from_deg,
        0.0,
        None,
        radius_m,
        scenario.outputs,
    )
    return anchor


def _wake(
    spec: Object | Targets,
    anchor: bpy.types.Object,
    bearing_deg: float,
    heading_deg: float,
) -> list[wakes.Wake]:
    """The wake `_pose` leaves, if the hull is under way.

    ponytail: along the heading from the start, without the drift's figure-eight; give
    the wake the drift's offset if a drifting hull under way shows its wake beside it.
    """
    if spec.speed_mps <= 0.0:
        return []
    bearing = math.radians(bearing_deg)
    return [
        wakes.Wake(
            speed_mps=spec.speed_mps,
            length_m=anchor["length_m"],
            beam_m=anchor["beam_m"],
            start_m=(
                spec.range_m * math.sin(bearing),
                spec.range_m * math.cos(bearing),
            ),
            heading_rad=math.radians(heading_deg),
        )
    ]


def _waterplane(anchor: bpy.types.Object) -> tuple[float, float]:
    """A hull's length and beam, in its own frame, bow to +Y."""
    # The parts' matrix_world is stale until the depsgraph runs.
    bpy.context.view_layer.update()
    corners = _corners(_meshes([anchor]), anchor.matrix_world.inverted())
    xs, ys = [c.x for c in corners], [c.y for c in corners]
    return max(ys) - min(ys), max(xs) - min(xs)


def _ride(
    anchors: Sequence[bpy.types.Object],
    field: tuple[waves.Wave, ...],
    tall: tuple[waves.Wave, ...],
    outputs: Outputs,
) -> None:
    """Key each hull's pitch and roll to the sea under its keyed pose, and its heave
    to the `tall` waves, which are geometry: the rest are shading, which a heaving
    hull would break its waterline on."""
    sc = bpy.context.scene
    hulls = []
    for anchor in anchors:
        (attitude,) = anchor.children
        hulls.append((anchor, attitude, *_waterplane(anchor)))
    keyed = len(outputs.times_s) > 1
    for frame, t_s in enumerate(outputs.times_s):
        sc.frame_set(frame)
        for anchor, attitude, length_m, beam_m in hulls:
            east, north, _ = anchor.matrix_world.translation
            heading = heading_rad(anchor)
            pitch, roll = waves.attitude(
                field, east, north, heading, length_m, beam_m, t_s
            )
            # +x turns the bow up; +y turns starboard down.
            attitude.rotation_euler = (pitch, -roll, 0.0)
            attitude.location.z = waves.heave_m(
                tall, east, north, heading, length_m, beam_m, t_s
            )
            if keyed:
                attitude.keyframe_insert("rotation_euler", index=0, frame=frame)
                attitude.keyframe_insert("rotation_euler", index=1, frame=frame)
                attitude.keyframe_insert("location", index=2, frame=frame)


JPEG_QUALITY = 95

# ISO 12232's saturation-based speed puts mid-grey at 10/78 of saturation; the
# log-average luminance stands for a scene's mid-grey (Reinhard et al. 2002).
MID_GREY = 10 / 78
# A lens scatters in Harvey's form, (1 + (r / r0)^2)^(-g / 2) off the direct light
# (Harvey 1976, the ABg model). Judgements, for a generic lens: the share of the light
# it scatters, r0 in half the image's longer side, and g.
GLARE_SHARE, GLARE_SHOULDER, GLARE_SLOPE = 0.02, 0.01, 3.0
# Judgement: the Blur node's size, in the image's longer side; a pixel of a 4K frame.
BLUR_OF_SIDE = 1 / 3840

# Blender's identifier and bit depth.
FORMATS: dict[ImageFormat, tuple[str, str]] = {
    "exr": ("OPEN_EXR", "32"),
    "png": ("PNG", "8"),
    "jpg": ("JPEG", "8"),
}


def enable_gpu() -> str | None:
    """Point Cycles at a GPU and name its backend; None leaves it on the CPU.

    Without `refresh_devices()` it stays on the CPU.
    """
    preferences = bpy.context.preferences.addons["cycles"].preferences
    for backend in ("METAL", "OPTIX", "CUDA", "HIP", "ONEAPI"):
        try:
            preferences.compute_device_type = backend
        except TypeError:
            continue  # not compiled into this build
        preferences.refresh_devices()
        # `devices` lists a card once per backend; CPU alongside the GPU wins nothing.
        if any(device.type == backend for device in preferences.devices):
            for device in preferences.devices:
                device.use = device.type == backend
            return backend
    return None


def _output(
    outputs: Outputs, band: Band, largest_px: int, graded: bool = False
) -> None:
    """Render and display settings, in the .blend, so F12 renders what `render` does.
    `largest_px` is the longest side of any of the band's cameras, and `graded` whether
    any has a sensor."""
    sc = bpy.context.scene
    # Not EEVEE: it darkens rough reflections at grazing view.
    sc.render.engine = "CYCLES"
    # Caps reflected light at 10, and LWIR radiance is tens of W m^-2 sr^-1.
    sc.eevee.clamp_surface_indirect = 0.0
    sc.cycles.device = "GPU" if enable_gpu() else "CPU"
    sc.cycles.samples = getattr(outputs.samples, band)
    # On by default. OIDN breaks the ir frame's R=G=B and blurs the waves.
    sc.cycles.use_denoising = False
    # Cycles merges tiles through a file on disk; a frame in one tile renders the same
    # pixels sooner. RNA clamps a side past its maximum, which then tiles.
    sc.cycles.tile_size = largest_px
    # The factory AgX is a film curve; a camera clips, and LWIR radiance is no picture.
    view = sc.view_settings
    view.view_transform, view.look = "Standard", "None"
    view.exposure, view.gamma = 0.0, 1.0
    # 8-bit radiance is not radiance; `render` maps 8-bit ir from the exr.
    file_format, depth = FORMATS[outputs.format if band == "eo" else "exr"]
    _compositor(outputs.exposure_compensation_ev if depth == "8" else None, graded)
    sc.render.image_settings.file_format = file_format
    sc.render.image_settings.color_depth = depth
    sc.render.image_settings.quality = JPEG_QUALITY
    sc.render.use_persistent_data = True
    # The GPU compositor, the factory's, keeps memory from every render, and a clip
    # runs out of it.
    sc.render.compositor_device = "CPU"
    # A fixed seed would hold the sample noise still while the scene moves under it.
    sc.cycles.use_animated_seed = True


def _compositor(compensation_ev: float | None, graded: bool = False) -> None:
    """The render as a camera takes it, `compensation_ev` over auto-exposure, through
    a sensor if `graded`; as rendered for None."""
    tree = bpy.data.node_groups.new("compositor", "CompositorNodeTree")
    tree.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    bpy.context.scene.compositing_node_group = tree
    image = tree.nodes.new("CompositorNodeRLayers").outputs["Image"]
    if compensation_ev is not None:
        image = _camera(tree, image, compensation_ev, graded)
    tree.links.new(image, tree.nodes.new("NodeGroupOutput").inputs["Image"])


def _camera(
    tree: bpy.types.NodeTree,
    image: bpy.types.NodeSocket,
    compensation_ev: float,
    with_sensor: bool = False,
) -> bpy.types.NodeSocket:
    """A lens's falloff, glare and blur, then auto-exposure, and with `with_sensor`
    the sensor and ISP. Without one the Standard view clips at white, as a sensor
    does. `respond` sets the camera.

    ponytail: each frame meters itself; a real camera's exposure lags, so add a time
    constant if a sequence flickers.
    """
    nodes, links = tree.nodes, tree.links
    coords, info = (
        nodes.new(kind)
        for kind in ("CompositorNodeImageCoordinates", "CompositorNodeImageInfo")
    )
    links.new(image, coords.inputs["Image"])
    links.new(image, info.inputs["Image"])
    # Uniform spans -1 to 1 along the longer side, so the kernel is a fixed share of
    # the frame at any resolution. Fog Glow and Bloom change shape with it.
    radius = sea._vector(tree, "LENGTH", coords.outputs["Uniform"])
    # Natural vignetting: irradiance falls as cos^4 of the field angle, 1 / (1 +
    # tan^2)^2 (Kerr 2007). `lens.vignette` is the tangent at Uniform's 1, or 0.
    tangent = sea._math(tree, "MULTIPLY", radius, sensor._value(tree, "lens.vignette"))
    falloff = sea._math(
        tree,
        "POWER",
        sea._math(tree, "MULTIPLY_ADD", tangent, tangent, 1.0),
        -2.0,
    )
    image = sea._vector(tree, "SCALE", image, falloff)
    shoulder = sea._math(tree, "DIVIDE", radius, GLARE_SHOULDER)
    harvey = sea._math(
        tree,
        "POWER",
        sea._math(tree, "MULTIPLY_ADD", shoulder, shoulder, 1.0),
        -GLARE_SLOPE / 2,
    )
    glare = nodes.new("CompositorNodeGlare")
    glare.inputs["Type"].default_value = "Kernel"
    # Half the resolution: Harvey's glow is smooth, so only the sun's rim shows it.
    glare.inputs["Quality"].default_value = "Medium"
    # A lens scatters all the light, not only what clips.
    glare.inputs["Threshold"].default_value = 0.0
    # Glare adds its glow to the image; normalized, the glow is this share of the sum.
    glare.inputs["Strength"].default_value = GLARE_SHARE / (1 - GLARE_SHARE)
    kernel = next(s for s in glare.inputs if s.name == "Kernel" and s.type == "VALUE")
    links.new(harvey, kernel)
    links.new(image, glare.inputs["Image"])
    side = nodes.new("ShaderNodeSeparateXYZ")
    links.new(info.outputs["Resolution"], side.inputs["Vector"])
    longer_px = sea._math(tree, "MAXIMUM", side.outputs["X"], side.outputs["Y"])
    # A share of the frame, not a pixel count, so it blurs alike at any resolution.
    size_px = sea._math(tree, "MULTIPLY", longer_px, BLUR_OF_SIDE)
    size = nodes.new("ShaderNodeCombineXYZ")
    links.new(size_px, size.inputs["X"])
    links.new(size_px, size.inputs["Y"])
    blur = nodes.new("CompositorNodeBlur")
    blur.inputs["Type"].default_value = "Gaussian"
    links.new(size.outputs["Vector"], blur.inputs["Size"])
    links.new(glare.outputs["Image"], blur.inputs["Image"])
    image = blur.outputs["Image"]
    luminance = nodes.new("CompositorNodeRGBToBW")
    links.new(image, luminance.inputs["Image"])
    # Blender's log of 0 is 0, which would count black as a luminance of 1.
    lit = sea._math(tree, "MAXIMUM", luminance.outputs["Val"], 1e-6)
    levels = nodes.new("CompositorNodeLevels")
    levels.inputs["Channel"].default_value = "Luminance"
    links.new(sea._math(tree, "LOGARITHM", lit, math.e), levels.inputs["Image"])
    log_average = sea._math(tree, "EXPONENT", levels.outputs["Mean"])
    gain = sea._math(
        tree, "DIVIDE", 2**compensation_ev * MID_GREY, log_average, name="exposure"
    )
    processing = sensor._value(tree, "processing.gain", 1.0)
    gain = sea._math(tree, "MULTIPLY", gain, processing)
    exposed = sea._vector(tree, "SCALE", image, gain)
    if not with_sensor:
        return exposed
    # A camera without a full well shares the tree, and takes the ideal frame.
    grey = sea._math(tree, "MULTIPLY", processing, 2**compensation_ev * MID_GREY)
    read = sensor.chain(tree, exposed, coords.outputs["Pixel"], grey)
    return sensor._switch(tree, "sensor.on", exposed, read)


def respond(mount: Mount, noise: np.random.Generator) -> None:
    """Set Cycles' pixel filter and the compositor's camera to `mount`'s, its sensor
    noise seeded from `noise`; a build without a compositor takes only the filter."""
    # A microbolometer integrates over its pixel: Cycles' box filter, a pixel wide.
    cycles = bpy.context.scene.cycles
    props = cycles.bl_rna.properties
    if mount.camera.microbolometer.pitch_um is not None:
        cycles.pixel_filter_type, cycles.filter_width = "BOX", 1.0
    else:
        cycles.pixel_filter_type = props["pixel_filter_type"].default
        cycles.filter_width = props["filter_width"].default
    tree = bpy.context.scene.compositing_node_group
    if tree is None or "lens.vignette" not in tree.nodes:
        return
    camera = mount.camera
    longer_of_width = max(camera.width_px, camera.height_px) / camera.width_px
    tangent = longer_of_width * math.tan(math.radians(camera.hfov_deg) / 2)
    sensor._set(tree, "lens.vignette", tangent if camera.lens.vignetting else 0.0)
    sensor._set(tree, "processing.gain", 2**camera.processing.exposure_ev)
    if "sensor.on" not in tree.nodes:
        return
    sensor._toggle(tree, "sensor.on", camera.sensor.full_well_e is not None)
    sensor.respond(tree, camera, noise)


class OverlapError(ValueError):
    """Two targets' hulls meet."""


def _overlaps(anchors: list[bpy.types.Object], frames: int) -> list[tuple[str, str]]:
    """Pairs of hulls whose footprints meet in any frame: each a circle of half its
    larger horizontal extent about its anchor, where the frame has moved it."""
    radius_m = {a.name: max(a["length_m"], a["beam_m"]) / 2 for a in anchors}
    sc = bpy.context.scene
    found = set()
    for frame in range(frames):
        sc.frame_set(frame)
        for a, b in combinations(anchors, 2):
            apart_m = (a.matrix_world.translation - b.matrix_world.translation).xy
            if apart_m.length < radius_m[a.name] + radius_m[b.name]:
                found.add((a.name, b.name))
    return sorted(found)


def _viewport(near_m: float, far_m: float) -> None:
    """Blender's view clip defaults to 0.01-1000 m, which cuts a sea reaching tens
    of km."""
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type != "VIEW_3D":
                continue
            for space in area.spaces:
                if space.type == "VIEW_3D":
                    space.clip_start, space.clip_end = near_m, far_m


def _sightline_m(scenario: Scenario, anchors: Iterable[bpy.types.Object]) -> float:
    """Past the camera's horizon plus the tallest hull's own, every hull is below the
    horizon."""
    # The parts' matrix_world is stale until the depsgraph runs.
    bpy.context.view_layer.update()
    tops_m = [
        max(c.z for c in _corners(_meshes([anchor])))
        - anchor.matrix_world.translation.z
        for anchor in anchors
    ]
    if not tops_m:
        return 0.0
    heave = scenario.ownship.heave
    eye_m = max(rig.height_m for rig in scenario.rigs.values()) + (
        heave.amplitude_m if heave is not None else 0.0
    )
    k = scenario.sea.refraction_k
    return waves.horizon_m(eye_m, k) + waves.horizon_m(max(tops_m), k)


def _finest_footprint_m(
    scenario: Scenario, band: Band, cameras: dict[str, bpy.types.Object]
) -> float:
    """The finest pixel any of the band's cameras lays on the sea in any frame: its
    height over the sea below it, which the curved sea's crests come no nearer than,
    times the angle of a pixel."""
    sc = bpy.context.scene
    radius_m = waves.earth_radius_m(scenario.sea.refraction_k)
    tall, _ = sea.displaced((wind_waves(scenario), swell_waves(scenario)))
    crest_m = sum(w.amplitude_m for w in tall)
    finest = math.inf
    for frame in range(len(scenario.outputs.times_s)):
        sc.frame_set(frame)
        # matrix_world is stale until the depsgraph runs.
        bpy.context.view_layer.update()
        for mount in scenario.mounts:
            if mount.camera.band != band:
                continue
            camera = cameras[mount.name]
            east, north, up = camera.matrix_world.translation
            height_m = max(up - waves.sea_z_m(east, north, radius_m) - crest_m, 0.0)
            pixel_rad = camera.data.angle_x / mount.camera.width_px
            finest = min(finest, height_m * pixel_rad)
    sc.frame_set(0)
    return finest


def _rays(
    scenario: Scenario, mounts: list[Mount], cameras: dict[str, bpy.types.Object]
) -> list[tuple[Vector, Vector, float]]:
    """(origin, unit direction, the camera's pixel in rad) of rays across each camera's
    frame, in every frame, on the foam memory's drifting grid."""
    sc = bpy.context.scene
    drift = sea.surface_drift(scenario.sea, scenario.outputs)
    rays = []
    for frame, t_s in enumerate(scenario.outputs.times_s):
        sc.frame_set(frame)
        # matrix_world is stale until the depsgraph runs.
        bpy.context.view_layer.update()
        for mount in mounts:
            camera = cameras[mount.name]
            pixel_rad = camera.data.angle_x / mount.camera.width_px
            turn = camera.matrix_world.to_3x3()
            # Not `view_frame`: it takes the scene's aspect, which the build sets last.
            across = math.tan(camera.data.angle_x / 2)
            up = across * mount.camera.height_px / mount.camera.width_px
            corners = [
                Vector((x * across, y * up, -1.0))
                for x, y in ((1, 1), (1, -1), (-1, -1), (-1, 1))
            ]
            for a in np.linspace(0.0, 1.0, FOAM_RAYS):
                for b in np.linspace(0.0, 1.0, FOAM_RAYS):
                    toward = (corners[2].lerp(corners[1], a)).lerp(
                        corners[3].lerp(corners[0], a), b
                    )
                    # Water at x is on the grid at x - v t.
                    origin = camera.matrix_world.translation - Vector(
                        (drift[0] * t_s, drift[1] * t_s, 0.0)
                    )
                    rays.append((origin, (turn @ toward).normalized(), pixel_rad))
    sc.frame_set(0)
    return rays


def _box(
    rays: list[tuple[Vector, Vector, float]], cell_m: float, shrink: float
) -> tuple[float, float, float, float]:
    """(west, south, width, height) on the lattice of `cell_m` round where the rays meet
    the sea, or pass over it, out to where their camera's pixel spans `2 * shrink`
    cells, plus the grid's fading edge."""
    xs, ys = [], []
    for origin, ray, pixel_rad in rays:
        flat = Vector((ray.x, ray.y))
        run = 2 * shrink * cell_m / pixel_rad
        if ray.z < 0.0:
            run = min(run, origin.z / -ray.z * flat.length)
        flat.normalize()
        xs += [origin.x, origin.x + flat.x * run]
        ys += [origin.y, origin.y + flat.y * run]
    margin = (sea.FOAM_SEAM_CELLS + 1) * cell_m
    west = math.floor((min(xs) - margin) / cell_m) * cell_m
    south = math.floor((min(ys) - margin) / cell_m) * cell_m
    return west, south, max(xs) + margin - west, max(ys) + margin - south


def _foam_box(
    rays: list[tuple[Vector, Vector, float]], cell_m: float
) -> tuple[float, float, float, float, float] | None:
    """`_box` at the largest shrink up to 1 that holds `FOAM_MAX_CELLS` a side, and the
    shrink; None if the cameras stand too far apart for even their origins to fit."""
    most_m = FOAM_MAX_CELLS * cell_m
    least_m = max(_box(rays, cell_m, 0.0)[2:])
    if least_m >= most_m:
        return None
    shrink = 1.0
    box = _box(rays, cell_m, shrink)
    while max(box[2:]) > most_m:
        # The excess over the origins' spread shrinks at most in proportion, so this
        # may repeat.
        shrink *= (most_m - least_m) / (max(box[2:]) - least_m)
        box = _box(rays, cell_m, shrink)
    return (*box, shrink)


def _foam_grids(
    scenario: Scenario,
    band: Band,
    cameras: dict[str, bpy.types.Object],
    wind: tuple[waves.Wave, ...],
) -> tuple[waves.FoamGrid, ...]:
    """Grids over every stretch of sea the band's cameras see in any frame, each on a
    lattice of its own cells, so a cell remembers the same foam whatever else is in
    view. The finest has `FOAM_NEAR_CELL_M` cells, out to where a camera's pixel spans
    one; each next is `FOAM_LEVEL_RATIO` coarser, out to the horizon or to cells too
    coarse for any breaker. A grid too wide to hold reaches less far."""
    if not wind:
        return ()
    mounts = [m for m in scenario.mounts if m.camera.band == band]
    rays = _rays(scenario, mounts, cameras)
    horizon = waves.horizon_m(
        max(origin.z for origin, _, _ in rays), scenario.sea.refraction_k
    )
    longest_m = max(2 * math.pi / w.k_rad_m for w in wind)
    finest_rad = min(pixel_rad for _, _, pixel_rad in rays)
    grids = []
    cell_m = FOAM_NEAR_CELL_M
    while True:
        fit = _foam_box(rays, cell_m)
        reach_m = 0.0
        if fit is not None:
            west, south, width, height, shrink = fit
            columns, rows = math.ceil(width / cell_m), math.ceil(height / cell_m)
            grids.append(waves.FoamGrid(west, south, cell_m, columns, rows))
            reach_m = shrink * cell_m / finest_rad
        cell_m *= FOAM_LEVEL_RATIO
        shortest_m = waves.FOAM_CELLS_PER_WAVE * cell_m
        if reach_m >= horizon or shortest_m > min(longest_m, FOAM_LONGEST_BREAKER_M):
            return tuple(grids)


def _placed(
    scenario: Scenario,
    radius_m: float,
    heat: _Heat | None,
    hulls: dict[str, list[bpy.types.Object]],
    trails: list[wakes.Wake],
) -> tuple[dict[str, list[bpy.types.Object]], dict[str, list[bpy.types.Object]]]:
    """Every object and ring hull, then every turbine, by asset; each wake under way
    added to `trails`."""
    sky, outputs = scenario.sky, scenario.outputs
    vessels: defaultdict[str, list[bpy.types.Object]] = defaultdict(list)
    for spec in scenario.objects:
        vessels[spec.asset] += _object(
            spec, radius_m, sky, heat, hulls, outputs, trails
        )
    if scenario.targets is not None:
        ring = _targets(scenario.targets, radius_m, sky, heat, hulls, outputs, trails)
        for asset, anchors in ring.items():
            vessels[asset] += anchors
    turbines: defaultdict[str, list[bpy.types.Object]] = defaultdict(list)
    for farm in scenario.farms:
        for range_m, bearing_deg in farm.positions():
            turbines[farm.asset].append(
                _turbine(farm, range_m, bearing_deg, scenario, radius_m, heat, hulls)
            )
    return vessels, turbines


def build(scenario: Scenario, band: Band = "eo", cull: bool = False) -> Built:
    """Replace the current Blender session's contents with `scenario` in one band.

    `cull` draws only the waves the scenario's own cameras resolve, at their own
    resolution: a camera zoomed, enlarged or added later loses its finest ripples."""
    if not any(mount.camera.band == band for mount in scenario.mounts):
        raise ValueError(f"no rig has an {band} camera to build an {band} scene for")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sea.forget_foam()
    sides = [
        max(m.camera.width_px, m.camera.height_px)
        for m in scenario.mounts
        if m.camera.band == band
    ]
    graded = any(
        m.camera.sensor.full_well_e is not None
        for m in scenario.mounts
        if m.camera.band == band
    )
    _output(scenario.outputs, band, max(sides), graded)
    bpy.context.scene.world = _sky(scenario.sky, band)
    reach_m = sea.sea_reach_m(scenario)
    far_m = 1.5 * reach_m  # the sea's corner is reach * sqrt(2) away
    outputs = scenario.outputs
    wind, swell = wind_waves(scenario), swell_waves(scenario)
    rigs = _rigs(scenario, far_m)
    hulls: dict[str, list[bpy.types.Object]] = {}
    tall, _ = sea.displaced((wind, swell))
    heat = _heat(scenario) if band == "ir" else None
    vessel = _ownship(
        scenario.ownship, scenario.sky, heat, rigs.root, hulls, outputs, tall
    )
    radius_m = waves.earth_radius_m(scenario.sea.refraction_k)
    trails: list[wakes.Wake] = []
    vessels, turbines = _placed(scenario, radius_m, heat, hulls, trails)
    targets = {**vessels, **turbines}  # no asset is both
    far_m = max(far_m, _sightline_m(scenario, chain(*targets.values())))
    for camera in rigs.cameras.values():
        camera.data.clip_end = far_m
    # After the hulls, whose poses and beams set the wakes.
    material = sea.material(
        scenario.sea,
        wind,
        swell,
        outputs,
        (
            substream(scenario.seed, "sea/gust"),
            substream(scenario.seed, "sea/slick"),
            substream(scenario.seed, "sea/foam"),
        ),
        tuple(trails),
        _light(scenario.sky, band),
        sea.Seen(
            _finest_footprint_m(scenario, band, rigs.cameras) if cull else 0.0,
            _foam_grids(scenario, band, rigs.cameras, wind),
        ),
    )
    sea.water(scenario.sea, reach_m, material)
    # After the last material.
    _haze(scenario.sky, band, far_m)
    # The object-index pass reads 0 for sky and sea. The ownship takes the index after
    # the targets: no box, and no target's background.
    anchors = [*chain(*targets.values()), vessel]
    for index, anchor in enumerate(anchors, start=1):
        for part in [anchor, *anchor.children_recursive]:
            part.pass_index = index
    # The first camera may belong to the other band, whose optics this build lacks.
    first = next(mount for mount in scenario.mounts if mount.camera.band == band)
    sc = bpy.context.scene
    sc.camera = rigs.cameras[first.name]
    sc.render.resolution_x, sc.render.resolution_y = (
        first.camera.width_px,
        first.camera.height_px,
    )
    respond(first, substream(scenario.seed, f"sensor/{first.name}"))
    _viewport(min(rig.near_clip_m for rig in scenario.rigs.values()), far_m)
    # After the last import, which sets fps and fps_base to the file's own. The
    # factory scene starts at frame 1.
    sc.frame_start = sc.frame_current = 0
    sc.frame_end = len(outputs.times_s) - 1
    sc.render.fps, sc.render.fps_base = outputs.fps, 1.0
    # ponytail: a floating turbine stands still; give it its platform's motion
    # if a sea rocks one visibly.
    _ride(list(chain(*vessels.values())), wind + swell, tall, outputs)
    solid = list(chain(*targets.values()))
    if scenario.ownship.asset is not None:
        solid.append(vessel)
    if meeting := _overlaps(solid, len(outputs.times_s)):
        raise OverlapError(f"hulls inside each other: {meeting}")
    sc.frame_set(0)
    # Until the depsgraph runs, every child still reports its pre-parenting
    # matrix_world, so anything measuring the scene reads the wrong place.
    bpy.context.view_layer.update()
    return Built(vessel, rigs.rigs, rigs.cameras, targets)
