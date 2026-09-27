"""The sea's mesh and materials: waves as shading on a grid curved to the earth."""

import math

import bpy
import numpy as np

from seascape import lwir
from seascape.blend import CURVE_SAMPLES, animate, curve_image, place
from seascape.config import Band, Outputs, Rig, Sea
from seascape.waves import (
    Wave,
    earth_radius_m,
    horizon_m,
    sea_z_m,
    specular_roughness,
    unresolved_slope,
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


def _wave_normals(
    tree: bpy.types.NodeTree, field: tuple[Wave, ...], outputs: Outputs
) -> bpy.types.NodeSocket:
    """The field's analytic slope, added to the curved grid's normal."""
    link = tree.links.new
    geometry = tree.nodes.new("ShaderNodeNewGeometry")
    position = tree.nodes.new("ShaderNodeSeparateXYZ")
    # Height is left out, so the sea curving under the field cannot slide it.
    xyt = tree.nodes.new("ShaderNodeCombineXYZ")
    link(geometry.outputs["Position"], position.inputs["Vector"])
    link(position.outputs["X"], xyt.inputs["X"])
    link(position.outputs["Y"], xyt.inputs["Y"])
    link(_sea_time(tree, outputs), xyt.inputs["Z"])

    relief = tree.nodes.new("ShaderNodeVectorMath")
    relief.operation = "SCALE"
    relief.name = "wave_relief"
    relief.inputs["Scale"].default_value = 1.0
    gradient = None
    for i, wave in enumerate(field):
        dot = tree.nodes.new("ShaderNodeVectorMath")
        dot.operation = "DOT_PRODUCT"
        dot.name = f"wave_{i}"
        dot.inputs["Vector_001"].default_value = (
            wave.k_east_rad_m,
            wave.k_north_rad_m,
            -wave.omega_rad_s,
        )
        phase = tree.nodes.new("ShaderNodeMath")
        phase.operation = "ADD"
        phase.name = f"wave_{i}_phase"
        phase.inputs["Value_001"].default_value = wave.phase_rad
        sine = tree.nodes.new("ShaderNodeMath")
        sine.operation = "SINE"
        # -d height / dx of a cos(phase) is a k_x sin(phase).
        term = tree.nodes.new("ShaderNodeVectorMath")
        term.operation = "MULTIPLY_ADD"
        term.name = f"wave_{i}_slope"
        term.inputs["Vector"].default_value = (
            wave.amplitude_m * wave.k_east_rad_m,
            wave.amplitude_m * wave.k_north_rad_m,
            0.0,
        )
        link(xyt.outputs["Vector"], dot.inputs["Vector"])
        link(dot.outputs["Value"], phase.inputs["Value"])
        link(phase.outputs["Value"], sine.inputs["Value"])
        link(sine.outputs["Value"], term.inputs["Vector_001"])
        if gradient is not None:
            link(gradient, term.inputs["Vector_002"])
        gradient = term.outputs["Vector"]
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
    return normal.outputs["Vector"]


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
    sea: Sea, field: tuple[Wave, ...], outputs: Outputs
) -> bpy.types.Material:
    """eps(theta) of the sea emitted, the remaining 1 - eps reflected from the sky.

    Complements, so the two very nearly cancel and the sea holds close to ambient at
    every angle.
    """
    material = bpy.data.materials.new("sea")
    tree = material.node_tree
    tree.nodes.clear()
    unresolved = unresolved_slope(sea.wind_speed_mps, field)
    mirror = tree.nodes.new("ShaderNodeBsdfGlossy")
    # The same unresolved slope the emissivity curve is averaged over.
    mirror.inputs["Roughness"].default_value = specular_roughness(unresolved)
    # Glossy BSDF ships at 0.8 grey. The Mix Shader already applies the 1 - eps
    # weighting, so anything but white here absorbs reflected sky and cuts a dark
    # notch along the horizon.
    mirror.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.inputs["Strength"].default_value = lwir.band_radiance(sea.t_sea_k)
    mix = tree.nodes.new("ShaderNodeMixShader")
    output = tree.nodes.new("ShaderNodeOutputMaterial")

    link = tree.links.new
    normal = _wave_normals(tree, field, outputs)
    link(normal, mirror.inputs["Normal"])
    # Mix Shader names both shader inputs "Shader", so they can only be indexed. Factor
    # is emissivity: 0 at grazing incidence takes the mirror, 1 head-on takes emission.
    link(mirror.outputs["BSDF"], mix.inputs[1])
    link(emission.outputs["Emission"], mix.inputs[2])
    link(
        _incidence_lookup(
            tree,
            _emissivity_image(sea.t_sea_k, unresolved),
            normal,
        ),
        mix.inputs["Factor"],
    )
    link(mix.outputs["Shader"], output.inputs["Surface"])
    return material


def _water_material(
    sea: Sea, field: tuple[Wave, ...], outputs: Outputs
) -> bpy.types.Material:
    """Daylight water, refracting at seawater's IOR."""
    material = bpy.data.materials.new("sea")
    tree = material.node_tree
    principled = tree.nodes["Principled BSDF"]
    principled.inputs["Base Color"].default_value = (0.004, 0.02, 0.035, 1.0)
    principled.inputs["Roughness"].default_value = specular_roughness(
        unresolved_slope(sea.wind_speed_mps, field)
    )
    principled.inputs["IOR"].default_value = 1.33
    tree.links.new(_wave_normals(tree, field, outputs), principled.inputs["Normal"])
    return material


def water(
    sea: Sea, field: tuple[Wave, ...], reach_m: float, band: Band, outputs: Outputs
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
    water.data.materials.append(material(sea, field, outputs))
    return water
