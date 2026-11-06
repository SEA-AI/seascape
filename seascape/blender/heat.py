"""An asset's LWIR skin: each zone's emission and the sky its glass and metal mirror."""

import math
from dataclasses import replace
from typing import NamedTuple

import bpy
import numpy as np

from seascape.assets import Material
from seascape.blender import wear
from seascape.blender.blend import (
    Num,
    curve_image,
    lookup,
    mix_shader,
    node,
    scalar,
    separate,
    vector,
)
from seascape.blender.sky import cloud_fraction
from seascape.config import (
    Scenario,
    Sky,
)
from seascape.physics import aerosol, lwir, thermal

# Judgement: glass and bare metal mirror the sky they do not emit; paint, rubber,
# fabric and wood scatter it. Limitation: dull stainless, as thermal takes it, is
# rougher than a mirror.
SPECULAR_ZONES: frozenset[thermal.ZoneName] = frozenset({"glass", "bare_metal"})


def thermal_skin(name: str, t_k: float, sky: Sky) -> bpy.types.Material:
    """eps of a painted hull emitted, the remaining 1 - eps reflected from the sky.

    Diffuse: flat marine paint is near-Lambertian in this band.
    """
    material = bpy.data.materials.new(name)
    tree = material.node_tree
    tree.nodes.clear()
    emissivity = thermal.ZONES["paint_hull"].emissivity
    skin = _reflecting(tree, _sunlit_emission(tree, t_k, sky), emissivity, None)
    node(tree, "ShaderNodeOutputMaterial", {"Surface": skin})
    return material


def _sunlit_emission(
    tree: bpy.types.NodeTree, t_k: float, sky: Sky
) -> bpy.types.NodeSocket:
    """Emission graded from shaded to sunlit by Lambert's cosine on a sun above the
    horizon.

    Two emissions mixed by `max(0, n . sun)`, so both ends are the exact band radiance
    and only the middle interpolates: no shader node can evaluate the Planck integral
    at the blended temperature.
    """
    shaded = node(tree, "ShaderNodeEmission", {"Strength": lwir.band_radiance(t_k)})
    elevation = sky.sun_elevation_deg
    if elevation is None or elevation <= 0:
        return shaded.outputs["Emission"]
    # The gain is linear in the beam a surface absorbs.
    gain_k = sky.solar_gain_k * sky.sun_transmittance
    sunlit = node(
        tree, "ShaderNodeEmission", {"Strength": lwir.band_radiance(t_k + gain_k)}
    )

    geometry = tree.nodes.new("ShaderNodeNewGeometry")
    facing = vector(tree, "DOT_PRODUCT", geometry.outputs["Normal"], sky.sun_vector)
    # Clamped, or a surface turned away from the sun would cool below shaded.
    lit = scalar(tree, "MAXIMUM", facing, 0.0)
    return mix_shader(tree, lit, shaded.outputs["Emission"], sunlit.outputs["Emission"])


class Heat(NamedTuple):
    """What a hull's zones settle against, and the water its wetted faces sit in."""

    env: thermal.Environment
    t_water_k: float
    sun: tuple[float, float, float] | None


def surroundings(scenario: Scenario) -> Heat:
    """What the scenario's hulls exchange heat with: air, sea, sun and sky."""
    sky = scenario.sky
    elevation = sky.sun_elevation_deg
    env = thermal.Environment(
        t_air_k=sky.t_air_k,
        t_sea_k=scenario.sea.t_skin_k(sky.sun_up),
        # Limitation: the true wind; a hull underway convects in its own way through the
        # air, and every hull shares one skin per zone.
        wind_m_s=scenario.sea.wind_speed_mps,
        sun_elevation_rad=None if elevation is None else math.radians(elevation),
        cloud_fraction=cloud_fraction(sky),
        sun_diffuse_fraction=sky.sun_diffuse_fraction,
        cloud_base_m=sky.cloud_base_m,
        atmosphere=sky.atmosphere,
        fog_cover=aerosol.fog_cover(sky.visibility_km, sky.atmosphere),
        beam_transmittance=sky.sun_transmittance,
        fog_sunlight_transmittance=aerosol.diffuse_transmittance(sky.visibility_km),
    )
    sun = sky.sun_vector if elevation is not None and elevation > 0 else None
    return Heat(env, scenario.sea.t_sea_k, sun)


def zone_skin(
    zone_name: thermal.ZoneName,
    heat: Heat,
    paint_absorptance: float | None = None,
    wet: bool = True,
    wear_material: Material | None = None,
    crevice: bpy.types.Image | None = None,
) -> bpy.types.Material:
    """A zone's eps emitted at its heat balance and 1 - eps reflected, shared by every
    hull: each face looks up its temperature by max(0, n . s) and the normal's z.

    No face shades another: a shadow on a hull is narrower than a pixel at the ranges
    a camera finds it, and the balance is steady, so a shadow would be as sharp and
    as deep as one that had sat for hours.

    Args:
        zone_name: A thermal zone, which sets the skin's emissivity, solar
            absorptance and the interior behind it.
        heat: The environment the balance is struck in, and the bulk water's
            temperature.
        paint_absorptance: Solar absorptance of the paint on
            `thermal.PAINTED`, 0 to 1.
        wet: Below the waterline, the bulk water's; False for a part that never meets
            it.
        wear_material: What the hull is built of, which mixes in the wear layers.
        crevice: The asset's crevice mask; required with `wear_material`."""
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
    steel = wear_material == "steel"
    if wear_material is not None:
        assert crevice is not None
        name += f"_worn{'_steel' if steel else ''}/{crevice.name}"
    if (found := bpy.data.materials.get(name)) is not None:
        return found
    material = bpy.data.materials.new(name)
    tree = material.node_tree
    tree.nodes.clear()

    clean = _emission(tree, zone, heat, name)
    skin = _reflecting(tree, clean, zone.emissivity, zone_name)
    water = _water(tree, heat) if wet or wear_material is not None else None
    if wear_material is not None:
        assert water is not None
        masks = node(tree, "ShaderNodeGroup", node_tree=wear.masks(crevice))
        skin = _worn(masks, skin, clean, zone, heat, name, steel, water)
    if wet:
        assert water is not None
        skin = _wetted(tree, skin, water)
    node(tree, "ShaderNodeOutputMaterial", {"Surface": skin})
    return material


def _emission(
    tree: bpy.types.NodeTree, zone: thermal.Zone, heat: Heat, name: str
) -> bpy.types.NodeSocket:
    sun_cos, up, table = thermal.temperature_table_k(zone, heat.env)
    image = curve_image(name, np.vectorize(lwir.band_radiance)(table))
    geometry = tree.nodes.new("ShaderNodeNewGeometry")
    _, _, z = separate(tree, geometry.outputs["Normal"])
    # At texel centres: the lookup interpolates between them.
    y = Num(tree, z).madd(0.5, 0.5).madd((len(up) - 1) / len(up), 0.5 / len(up))
    if heat.sun is None:
        at = tree.nodes.new("ShaderNodeValue").outputs["Value"]
        at.default_value = 0.5 / len(sun_cos)
    else:
        facing = vector(tree, "DOT_PRODUCT", geometry.outputs["Normal"], heat.sun)
        lit = Num(tree, scalar(tree, "MAXIMUM", facing, 0.0))
        at = lit.madd((len(sun_cos) - 1) / len(sun_cos), 0.5 / len(sun_cos))
    color = lookup(tree, image, at, y)
    return node(tree, "ShaderNodeEmission", {"Color": color}).outputs["Emission"]


def _water(tree: bpy.types.NodeTree, heat: Heat) -> bpy.types.NodeSocket:
    water = node(
        tree, "ShaderNodeEmission", {"Strength": lwir.band_radiance(heat.t_water_k)}
    )
    return _reflecting(tree, water.outputs["Emission"], thermal.SEA_EMISSIVITY, None)


def _worn(
    masks: bpy.types.ShaderNodeGroup,
    skin: bpy.types.NodeSocket,
    clean: bpy.types.NodeSocket,
    zone: thermal.Zone,
    heat: Heat,
    name: str,
    steel: bool,
    water: bpy.types.NodeSocket,
) -> bpy.types.NodeSocket:
    tree = masks.id_data
    for layer in wear.COVERS:
        if layer == "stain":
            # Judgement: the stain is the splash zone, kept wet at the bulk water's
            # temperature.
            covered = water
        elif steel:
            # Steel spreads a strip's heat sideways over L = sqrt(k t / h), 0.16 m for
            # 12 mm plate at k = 60.5 W m^-1 K^-1 (Incropera & DeWitt Table A.1, plain
            # carbon steel) and h = 30 W m^-2 K^-1 (Judgement: wind and radiation). A
            # strip w wide keeps 1 - exp(-w / 2L) of its own balance (after
            # Incropera & DeWitt section 3.6, infinite fin), a few percent for a
            # streak: so on steel a layer keeps the hull's temperature and changes
            # only its emissivity. GRP's L is 1-2 cm.
            emissivity = thermal.under(zone, layer).emissivity
            if emissivity == zone.emissivity:
                continue
            covered = _reflecting(tree, clean, emissivity, None)
        else:
            layered = thermal.under(zone, layer)
            emission = _emission(tree, layered, heat, f"{name}_{layer}")
            covered = _reflecting(tree, emission, layered.emissivity, None)
        skin = mix_shader(tree, masks.outputs[layer], skin, covered)
    return skin


def _wetted(
    tree: bpy.types.NodeTree, skin: bpy.types.NodeSocket, water: bpy.types.NodeSocket
) -> bpy.types.NodeSocket:
    """`skin` above the waterline, the bulk water below it."""
    # The fit puts the waterline at z = 0; the join keeps the hull's frame in metres.
    coords = tree.nodes.new("ShaderNodeTexCoord")
    _, _, height_m = separate(tree, coords.outputs["Object"])
    rise = node(
        tree, "ShaderNodeAttribute", attribute_type="OBJECT", attribute_name="rise_m"
    )
    above_water_m = scalar(tree, "ADD", height_m, rise.outputs["Factor"])
    under = scalar(tree, "LESS_THAN", above_water_m, 0.0)
    return mix_shader(tree, under, skin, water)


def _reflecting(
    tree: bpy.types.NodeTree,
    emission: bpy.types.NodeSocket,
    emissivity: float,
    zone_name: str | None,
) -> bpy.types.NodeSocket:
    """An emission weighted by its emissivity, the rest a white reflection: the Mix
    Shader already weights it by 1 - eps, so a grey would absorb part of it twice.

    Args:
        tree: The hull skin's node tree.
        emission: The surface's radiance as a blackbody would emit it.
        emissivity: Fraction of that the surface emits, 0 to 1; by Kirchhoff the
            rest is what it reflects.
        zone_name: A zone of `SPECULAR_ZONES` reflects as a sharp mirror; any other,
            or None, scatters."""
    white = (1.0, 1.0, 1.0, 1.0)
    if zone_name in SPECULAR_ZONES:
        scatter = node(
            tree,
            "ShaderNodeBsdfGlossy",
            {"Color": white, "Roughness": 0.0},
            distribution="GGX",
        )
    else:
        scatter = node(tree, "ShaderNodeBsdfDiffuse", {"Color": white})
    return mix_shader(tree, emissivity, scatter.outputs["BSDF"], emission)
