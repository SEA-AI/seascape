"""A pixel's footprint on the sea, along and across."""

import math
from collections.abc import Callable
from typing import NamedTuple

import bpy
import numpy as np

from seascape.blender.blend import (
    CURVE_SAMPLES,
    curve_image,
    drive,
    lookup,
    scalar,
    vector,
)

# A judgement: the footprints a sea pixel spans.
FOOTPRINT_RANGE_M = (1e-3, 1e4)


class _Pixel(NamedTuple):
    across_m: bpy.types.NodeSocket
    along_m: bpy.types.NodeSocket
    along_dir: bpy.types.NodeSocket  # unit, horizontal


def _pixel(
    tree: bpy.types.NodeTree, incoming: bpy.types.NodeSocket | None = None
) -> _Pixel:
    """The footprint of a pixel of the scene's perspective camera at the render's
    resolution, Resolution % included. `incoming` stands in for the Geometry node's,
    which a displacement shader lacks."""
    camera = tree.nodes.new("ShaderNodeCameraData")
    geometry = tree.nodes.new("ShaderNodeNewGeometry")
    if incoming is None:
        incoming = geometry.outputs["Incoming"]
    facing = vector(tree, "DOT_PRODUCT", incoming, geometry.outputs["Normal"])
    across = scalar(tree, "MULTIPLY", camera.outputs["View Distance"], 0.0)
    # ponytail: the build's cameras fit their angle to the width; a camera added in
    # Blender with another sensor fit gets the wrong pixel. Read `sensor_fit` when
    # such a camera must render.
    drive(
        across.node.inputs[1],
        "default_value",
        "angle / (width * percent / 100)",
        bpy.context.scene,
        angle="camera.data.angle_x",
        width="render.resolution_x",
        percent="render.resolution_percentage",
    )
    # Grazing stretches the pixel by 1 / cos along the view; the floor keeps it finite.
    cosine = scalar(tree, "MAXIMUM", scalar(tree, "ABSOLUTE", facing), 1e-3)
    tilt = vector(tree, "SCALE", geometry.outputs["Normal"])
    tree.links.new(facing, tilt.node.inputs["Scale"])
    level = vector(tree, "SUBTRACT", incoming, tilt)
    along = vector(tree, "NORMALIZE", level)
    return _Pixel(across, scalar(tree, "DIVIDE", across, cosine), along)


def _instance(
    tree: bpy.types.NodeTree,
    group: bpy.types.NodeTree,
    pixel: _Pixel,
    calm: bpy.types.NodeSocket | None,
) -> bpy.types.Node:
    """`group` fed the pixel's footprint. Cycles merges the identical nodes each call
    adds."""
    link = tree.links.new
    across_sq = scalar(tree, "MULTIPLY", pixel.across_m, pixel.across_m)
    along_sq = scalar(tree, "MULTIPLY", pixel.along_m, pixel.along_m)
    node = tree.nodes.new("ShaderNodeGroup")
    node.node_tree = group
    link(pixel.along_dir, node.inputs["Along"])
    link(across_sq, node.inputs["Across Sq"])
    link(scalar(tree, "SUBTRACT", along_sq, across_sq), node.inputs["Stretch"])
    if calm is not None:
        link(calm, node.inputs["Calm"])
    return node


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
    x = scalar(
        tree,
        "MULTIPLY_ADD",
        scalar(tree, "LOGARITHM", footprint_m, 10.0),
        1 / decades,
        -math.log10(low) / decades,
    )
    value = lookup(tree, table, x)
    value.node.name = name
    return value
