"""The sea's mesh and materials: waves as shading on a grid curved to the earth."""

import math

import bpy
import numpy as np

from seascape import lwir
from seascape.blend import CURVE_SAMPLES, animate, curve_image, place, sine
from seascape.config import Band, Outputs, Rig, Sea
from seascape.waves import (
    NOISE_DETAIL,
    NOISE_ROUGHNESS,
    NOISE_SLOPE_PER_UNIT,
    NOISE_SLOPE_PER_UNIT_4D,
    bump_slope,
    earth_radius_m,
    horizon_m,
    sea_z_m,
    specular_roughness,
    unresolved_slope,
    wave_length_m,
    wave_period_s,
)

# Cells per side: enough for the tangent point to land on a face, not an accuracy
# knob. A cell's sagitta, width^2 / 8R, is far under a pixel at the horizon.
SEA_CELLS = 128

# Margin on the horizon, or the grid's own edge becomes the horizon.
SEA_MARGIN = 1.5


def sea_reach_m(rig: Rig, sea: Sea) -> float:
    """Half-width of the sea, a margin past the horizon."""
    return SEA_MARGIN * horizon_m(rig.height_m, sea.refraction_k)


def _emissivity_image(t_sea_k: float, slope_sigma: float) -> bpy.types.Image:
    """`lwir.emissivity_curve` baked against cos(theta), which is what the shader has.

    The curve is sampled uniformly in angle; the shader's dot product is uniform in its
    cosine, so it is resampled here rather than corrected in nodes.
    """
    theta, eps = lwir.emissivity_curve(t_sea_k=t_sea_k, slope_sigma=slope_sigma)
    mu = np.cos(theta)[::-1]
    return curve_image(
        "sea_emissivity",
        np.interp(np.linspace(0.0, 1.0, CURVE_SAMPLES), mu, eps[::-1]),
    )


def _wave_normals(
    tree: bpy.types.NodeTree, sea: Sea, rng: np.random.Generator, outputs: Outputs
) -> bpy.types.NodeSocket:
    """Wave normals from world position.

    Shading, not geometry. A bump normal is evaluated per pixel and varies
    continuously, so distant water averages smooth; displaced geometry at any
    affordable spacing goes sub-pixel before the horizon and aliases instead.
    """
    length_m = wave_length_m(sea.wind_speed_mps)
    # z multiplier 0: seed and time own that axis, so the sea curving under it cannot
    # slide the wave field. A unit of z decorrelates the isotropic noise as a wavelength
    # of x does, so time advances z a unit per dominant period.
    # Vector Math names all three inputs "Vector"; identifiers tell them apart.
    scale = tree.nodes.new("ShaderNodeVectorMath")
    scale.operation = "MULTIPLY_ADD"
    scale.inputs["Vector_001"].default_value = (1.0 / length_m, 1.0 / length_m, 0.0)
    geometry = tree.nodes.new("ShaderNodeNewGeometry")
    noise = tree.nodes.new("ShaderNodeTexNoise")
    # Scale stays 1 so the vector above carries the wavelength in metres.
    noise.inputs["Scale"].default_value = 1.0
    noise.inputs["Detail"].default_value = NOISE_DETAIL
    noise.inputs["Roughness"].default_value = NOISE_ROUGHNESS

    phase = rng.random() * 1e3
    period_s = wave_period_s(sea.wind_speed_mps)
    times_s = outputs.times_s
    offset = scale.inputs["Vector_002"]
    slope_per_unit = NOISE_SLOPE_PER_UNIT
    if outputs.loop:
        # A line in z never returns; a circle in (z, W) does, at the same speed.
        noise.noise_dimensions = "4D"
        slope_per_unit = NOISE_SLOPE_PER_UNIT_4D
        span_s = outputs.span_s
        radius = span_s / (2.0 * math.pi * period_s)
        animate(offset, "default_value", times_s, sine(phase, radius, span_s), 2)
        animate(
            noise.inputs["W"],
            "default_value",
            times_s,
            lambda t: radius * (1.0 - math.cos(2.0 * math.pi * t / span_s)),
        )
    else:
        animate(offset, "default_value", times_s, lambda t: phase + t / period_s, 2)

    bump = tree.nodes.new("ShaderNodeBump")
    bump.inputs["Distance"].default_value = (
        bump_slope(sea.wind_speed_mps) * length_m / slope_per_unit
    )

    link = tree.links.new
    link(geometry.outputs["Position"], scale.inputs["Vector"])
    link(scale.outputs["Vector"], noise.inputs["Vector"])
    link(noise.outputs["Fac"], bump.inputs["Height"])
    return bump.outputs["Normal"]


def _incidence_lookup(
    tree: bpy.types.NodeTree,
    curve: bpy.types.Image,
    normal: bpy.types.NodeSocket,
) -> bpy.types.NodeSocket:
    """Sample `curve` at |cos(theta)| between the wave normal and the viewing ray.

    Against the wave normal, not the plane's, or a flat sea's emissivity gets applied
    to water that is visibly not flat.
    """
    geometry = tree.nodes.new("ShaderNodeNewGeometry")
    dot = tree.nodes.new("ShaderNodeVectorMath")
    dot.operation = "DOT_PRODUCT"
    facing = tree.nodes.new("ShaderNodeMath")
    facing.operation = "ABSOLUTE"
    lookup = tree.nodes.new("ShaderNodeCombineXYZ")
    texture = tree.nodes.new("ShaderNodeTexImage")
    texture.image = curve
    texture.extension = "EXTEND"

    link = tree.links.new
    link(geometry.outputs["Incoming"], dot.inputs[0])
    link(normal, dot.inputs["Vector_001"])
    link(dot.outputs["Value"], facing.inputs[0])
    link(facing.outputs["Value"], lookup.inputs["X"])
    link(lookup.outputs["Vector"], texture.inputs["Vector"])
    return texture.outputs["Color"]


def _thermal_sea(
    sea: Sea, rng: np.random.Generator, outputs: Outputs
) -> bpy.types.Material:
    """eps(theta) of the sea emitted, the remaining 1 - eps reflected from the sky.

    Complements, so the two very nearly cancel and the sea holds close to ambient at
    every angle.
    """
    material = bpy.data.materials.new("sea")
    tree = material.node_tree
    tree.nodes.clear()
    mirror = tree.nodes.new("ShaderNodeBsdfGlossy")
    # The same unresolved slope the emissivity curve is averaged over.
    mirror.inputs["Roughness"].default_value = specular_roughness(sea.wind_speed_mps)
    # Glossy BSDF ships at 0.8 grey. The Mix Shader already applies the 1 - eps
    # weighting, so anything but white here absorbs reflected sky and cuts a dark
    # notch along the horizon.
    mirror.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.inputs["Strength"].default_value = lwir.band_radiance(sea.t_sea_k)
    mix = tree.nodes.new("ShaderNodeMixShader")
    output = tree.nodes.new("ShaderNodeOutputMaterial")

    link = tree.links.new
    normal = _wave_normals(tree, sea, rng, outputs)
    link(normal, mirror.inputs["Normal"])
    # Mix Shader names both shader inputs "Shader", so they can only be indexed. Factor
    # is emissivity: 0 at grazing incidence takes the mirror, 1 head-on takes emission.
    link(mirror.outputs["BSDF"], mix.inputs[1])
    link(emission.outputs["Emission"], mix.inputs[2])
    link(
        _incidence_lookup(
            tree,
            _emissivity_image(sea.t_sea_k, unresolved_slope(sea.wind_speed_mps)),
            normal,
        ),
        mix.inputs["Factor"],
    )
    link(mix.outputs["Shader"], output.inputs["Surface"])
    return material


def _water_material(
    sea: Sea, rng: np.random.Generator, outputs: Outputs
) -> bpy.types.Material:
    """Daylight water, refracting at seawater's IOR."""
    material = bpy.data.materials.new("sea")
    tree = material.node_tree
    principled = tree.nodes["Principled BSDF"]
    principled.inputs["Base Color"].default_value = (0.004, 0.02, 0.035, 1.0)
    principled.inputs["Roughness"].default_value = specular_roughness(
        sea.wind_speed_mps
    )
    principled.inputs["IOR"].default_value = 1.33
    tree.links.new(_wave_normals(tree, sea, rng, outputs), principled.inputs["Normal"])
    return material


def water(
    sea: Sea, rng: np.random.Generator, reach_m: float, band: Band, outputs: Outputs
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
    water.data.materials.append(material(sea, rng, outputs))
    return water
