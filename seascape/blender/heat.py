"""An asset's LWIR skin: each zone's emission and the sky its glass and metal mirror."""

import math
from dataclasses import replace
from typing import NamedTuple

import bpy
import numpy as np

from seascape.blender.blend import (
    _math,
    curve_image,
    lookup,
)
from seascape.blender.sky import _sun_vector
from seascape.config import (
    Scenario,
    Sky,
)
from seascape.physics import lwir, thermal

# Judgement: glass and bare metal mirror the sky they do not emit; paint, rubber,
# fabric and wood scatter it. ponytail: dull stainless, as thermal takes it, is rougher
# than a mirror.
SPECULAR_ZONES = frozenset({"glass", "bare_metal"})


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
    zone_name: thermal.ZoneName,
    heat: _Heat,
    paint_absorptance: float | None = None,
    wet: bool = True,
) -> bpy.types.Material:
    """A zone's eps emitted at its heat balance and 1 - eps reflected, shared by every
    hull: each face looks up its temperature by max(0, n . s) and the normal's z.
    Below the waterline, the bulk water's, unless the part never meets it.
    `paint_absorptance` repaints the zones of `thermal.PAINTED`.

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
    if not wet:
        name += "_dry"
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
    y = _math(
        tree,
        "MULTIPLY_ADD",
        _math(tree, "MULTIPLY_ADD", normal.outputs["Z"], 0.5, 0.5),
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
        lit = _math(tree, "MAXIMUM", facing.outputs["Value"], 0.0)
        at = _math(
            tree,
            "MULTIPLY_ADD",
            lit,
            (len(sun_cos) - 1) / len(sun_cos),
            0.5 / len(sun_cos),
        )
    emission = tree.nodes.new("ShaderNodeEmission")
    link(lookup(tree, image, at, y), emission.inputs["Color"])
    skin = _reflecting(tree, emission.outputs["Emission"], zone.emissivity, zone_name)
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    link(_wetted(tree, skin, heat) if wet else skin, output.inputs["Surface"])
    return material


def _wetted(
    tree: bpy.types.NodeTree, skin: bpy.types.NodeSocket, heat: _Heat
) -> bpy.types.NodeSocket:
    """`skin` above the waterline, the bulk water below it."""
    link = tree.links.new
    water = tree.nodes.new("ShaderNodeEmission")
    water.inputs["Strength"].default_value = lwir.band_radiance(heat.t_water_k)
    wetted = _reflecting(tree, water.outputs["Emission"], thermal.SEA_EMISSIVITY, None)
    # The fit puts the waterline at z = 0; the join keeps the hull's frame in metres.
    coords = tree.nodes.new("ShaderNodeTexCoord")
    height = tree.nodes.new("ShaderNodeSeparateXYZ")
    link(coords.outputs["Object"], height.inputs["Vector"])
    under = _math(tree, "LESS_THAN", height.outputs["Z"], 0.0)
    mix = tree.nodes.new("ShaderNodeMixShader")
    link(under, mix.inputs["Factor"])
    link(skin, mix.inputs["Shader"])
    link(wetted, mix.inputs["Shader_001"])
    return mix.outputs["Shader"]


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
