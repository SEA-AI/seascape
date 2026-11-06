"""A pixel's footprint on the sea, along and across."""

import math
from collections.abc import Callable
from typing import NamedTuple

import bpy
import numpy as np

from seascape.blender.blend import (
    Num,
    curve_image,
    drive,
    lookup,
    node,
    scalar,
    separate,
    texels,
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
    resolution, Resolution % included.

    Args:
        tree: The sea material's shader tree.
        incoming: Stands in for the Geometry node's Incoming, which a displacement
            shader lacks; None takes the Geometry node's.
    """
    camera = tree.nodes.new("ShaderNodeCameraData")
    geometry = tree.nodes.new("ShaderNodeNewGeometry")
    if incoming is None:
        incoming = geometry.outputs["Incoming"]
    facing = vector(tree, "DOT_PRODUCT", incoming, geometry.outputs["Normal"])
    across = scalar(tree, "MULTIPLY", camera.outputs["View Distance"], 0.0)
    # Limitation: the build's cameras fit their angle to the width; a camera added in
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
    cosine = Num(tree, facing).op("ABSOLUTE").op("MAXIMUM", 1e-3)
    tilt = vector(tree, "SCALE", geometry.outputs["Normal"], facing)
    along = vector(tree, "NORMALIZE", vector(tree, "SUBTRACT", incoming, tilt))
    return _Pixel(across, (Num(tree, across) / cosine).socket, along)


def _instance(
    tree: bpy.types.NodeTree,
    group: bpy.types.NodeTree,
    pixel: _Pixel,
    calm: bpy.types.NodeSocket | None,
) -> bpy.types.Node:
    """A group node fed the pixel's footprint. Cycles merges the identical nodes each
    call adds.

    Args:
        tree: The sea material's shader tree.
        group: A wave group with footprint inputs.
        pixel: The footprint the group fades its waves by.
        calm: Feeds the group's Calm input, which scales the damped waves; None for no
            slick.
    """
    across_sq = Num(tree, pixel.across_m) * pixel.across_m
    along_sq = Num(tree, pixel.along_m) * pixel.along_m
    instance = node(
        tree,
        "ShaderNodeGroup",
        {
            "Along": pixel.along_dir,
            "Across Sq": across_sq.socket,
            "Stretch": (along_sq - across_sq).socket,
        },
        node_tree=group,
    )
    if calm is not None:
        tree.links.new(calm, instance.inputs["Calm"])
    return instance


def _footprint_table(name: str, value_at: Callable[[float], float]) -> bpy.types.Image:
    """A function of the footprint baked against log footprint: one lookup, not a sum
    over the waves per sample.

    Args:
        name: The table image's name in `bpy.data`.
        value_at: Its value at a footprint in metres, sampled log-spaced across
            `FOOTPRINT_RANGE_M`.
    """
    low, high = FOOTPRINT_RANGE_M
    return curve_image(
        name, np.array([value_at(f) for f in low * (high / low) ** texels()])
    )


def _at_footprint(
    tree: bpy.types.NodeTree,
    table: bpy.types.Image,
    footprint_m: bpy.types.NodeSocket,
    name: str,
) -> bpy.types.NodeSocket:
    low, high = FOOTPRINT_RANGE_M
    decades = math.log10(high / low)
    x = (
        Num(tree, footprint_m)
        .op("LOGARITHM", 10.0)
        .madd(1 / decades, -math.log10(low) / decades)
    )
    value = lookup(tree, table, x.socket)
    value.node.name = name
    return value


def _redraw(
    tree: bpy.types.NodeTree,
    cell: bpy.types.NodeSocket,
    time_s: bpy.types.NodeSocket,
    rate: bpy.types.NodeSocket | float,
    name: str,
) -> tuple[Num, Num]:
    """White noise per cell, drawn again at its own phase: ln of its first uniform,
    floored off 1e-7, and its second.

    Args:
        tree: The sea material's shader tree.
        cell: Integer cell coordinates; every point in a cell draws the same noise.
        time_s: The sea's time.
        rate: Draws per second.
        name: Names the node that counts the draws.
    """
    offset = node(
        tree, "ShaderNodeTexWhiteNoise", {"Vector": cell}, noise_dimensions="3D"
    )
    draw = Num(tree, time_s).madd(rate, offset.outputs["Value"]).op("FLOOR")
    draw.socket.node.name = name
    noise = node(
        tree,
        "ShaderNodeTexWhiteNoise",
        {"Vector": cell, "W": draw.socket},
        noise_dimensions="4D",
    )
    first, second, _ = separate(tree, noise.outputs["Color"])
    # Numerics: ln 0 is -inf.
    ln_u = Num(tree, first).op("MAXIMUM", 1e-7).op("LOGARITHM", math.e)
    return ln_u, Num(tree, second)
