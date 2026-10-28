"""The waves a pixel does not draw, as the slope variance of its lobe."""

import math
from collections.abc import Callable

import bpy

from seascape.blender.blend import (
    _math,
    _vector,
)
from seascape.blender.sea.pixel import _at_footprint, _footprint_table, _Pixel
from seascape.blender.sea.wind import _gusty
from seascape.config import Sea
from seascape.physics.waves import (
    GRAVITY_MS2,
    Wave,
    cox_munk_slick_variances,
    cox_munk_variances,
    mss_spread,
    unresolved_slope_variances,
)


def _grouped(
    tree: bpy.types.NodeTree,
    sea: Sea,
    where: bpy.types.NodeSocket,
    pixel: _Pixel,
    unresolved: tuple[bpy.types.NodeSocket, bpy.types.NodeSocket],
) -> tuple[bpy.types.NodeSocket, bpy.types.NodeSocket]:
    """`unresolved` scaled per cell by a lognormal of mean 1, so a pixel's footprint
    holds `mss_spread` of its mean square slope as the short waves' groups do; at
    `where`, (east, north, time). Rough footprints read warm at grazing views.

    A grazing footprint spans several cells along and part of one across. Each cell is
    redrawn when a group of the waves a footprint its own size fades crosses it, at
    their group velocity.

    ponytail: EO keeps the mean, as the glitter carries its own spread. Scale its lobe
    too if its sea reads too even.
    """
    if sea.wind_speed_mps <= 0.0:
        return unresolved
    link = tree.links.new
    footprint = _math(
        tree, "SQRT", _math(tree, "MULTIPLY", pixel.along_m, pixel.across_m)
    )
    # Judgement: square cells, a power of two across, one to two to the footprint's
    # geometric mean, so few fall in a pixel and their spread stays narrow.
    level = _math(tree, "FLOOR", _math(tree, "LOGARITHM", footprint, 2.0))
    cell_m = _math(tree, "POWER", 2.0, level, name="sea_group_cell")
    spread = _at_footprint(
        tree,
        _footprint_table(
            "sea_mss_spread",
            lambda f: mss_spread(sea.wind_speed_mps, sea.fetch_m, f),
        ),
        footprint,
        "sea_mss_spread",
    )
    kept = _math(
        tree,
        "MULTIPLY",
        _box_kept(tree, _math(tree, "DIVIDE", pixel.along_m, cell_m)),
        _box_kept(tree, _math(tree, "DIVIDE", pixel.across_m, cell_m)),
    )
    per_cell = _math(tree, "DIVIDE", spread, _math(tree, "SQRT", kept))
    sigma_sq = _math(
        tree,
        "LOGARITHM",
        _math(tree, "MULTIPLY_ADD", per_cell, per_cell, 1.0),
        math.e,
    )
    split = tree.nodes.new("ShaderNodeSeparateXYZ")
    link(where, split.inputs["Vector"])
    cell = tree.nodes.new("ShaderNodeCombineXYZ")
    for axis in "XY":
        link(
            _math(tree, "FLOOR", _math(tree, "DIVIDE", split.outputs[axis], cell_m)),
            cell.inputs[axis],
        )
    link(level, cell.inputs["Z"])
    offset = tree.nodes.new("ShaderNodeTexWhiteNoise")
    offset.noise_dimensions = "3D"
    link(cell.outputs["Vector"], offset.inputs["Vector"])
    # A cell L across leaves the waves up to L long, `waves.visibility`'s zero. Deep
    # water's group velocity is half sqrt(g lambda / 2 pi), so a group of them crosses
    # it at sqrt(g / (8 pi L)) per second. Read off the cell, every sample in it agrees.
    rate = _math(
        tree,
        "SQRT",
        _math(
            tree,
            "DIVIDE",
            GRAVITY_MS2 / (8 * math.pi),
            cell_m,
        ),
    )
    draw = _math(
        tree,
        "FLOOR",
        _math(
            tree,
            "MULTIPLY_ADD",
            split.outputs["Z"],
            rate,
            offset.outputs["Value"],
        ),
        name="sea_group_draw",
    )
    noise = tree.nodes.new("ShaderNodeTexWhiteNoise")
    noise.noise_dimensions = "4D"
    link(cell.outputs["Vector"], noise.inputs["Vector"])
    link(draw, noise.inputs["W"])
    uniform = tree.nodes.new("ShaderNodeSeparateXYZ")
    link(noise.outputs["Color"], uniform.inputs["Vector"])
    # Box-Muller.
    normal = _math(
        tree,
        "MULTIPLY",
        _math(
            tree,
            "SQRT",
            _math(
                tree,
                "MULTIPLY",
                _math(
                    tree,
                    "LOGARITHM",
                    _math(tree, "MAXIMUM", uniform.outputs["X"], 1e-7),
                    math.e,
                ),
                -2.0,
            ),
        ),
        _math(
            tree,
            "COSINE",
            _math(tree, "MULTIPLY", uniform.outputs["Y"], 2 * math.pi),
        ),
    )
    # Judgement: a lognormal, so the scale stays positive. exp(sigma z - sigma^2 / 2)
    # has mean 1.
    scale = _math(
        tree,
        "EXPONENT",
        _math(
            tree,
            "MULTIPLY_ADD",
            normal,
            _math(tree, "SQRT", sigma_sq),
            _math(tree, "MULTIPLY", sigma_sq, -0.5),
        ),
        name="sea_group_scale",
    )
    along, across = unresolved
    return _math(tree, "MULTIPLY", along, scale), _math(tree, "MULTIPLY", across, scale)


def _box_kept(
    tree: bpy.types.NodeTree, cells: bpy.types.NodeSocket
) -> bpy.types.NodeSocket:
    """The share of a cell's variance a box `cells` cells long keeps, at a random
    offset on the grid: the mean over the box of the triangle 1 - |d|, which is
    (1 - 1 / 3m) / m from one cell up and 1 - m / 3 below."""
    over = _math(
        tree,
        "DIVIDE",
        _math(tree, "MULTIPLY_ADD", cells, 3.0, -1.0),
        _math(tree, "MULTIPLY", _math(tree, "MULTIPLY", cells, cells), 3.0),
    )
    under = _math(tree, "MULTIPLY_ADD", cells, -1 / 3, 1.0)
    return _math(
        tree,
        "MULTIPLY_ADD",
        _math(tree, "GREATER_THAN", cells, 1.0),
        _math(tree, "SUBTRACT", over, under),
        under,
    )


def _unresolved(
    tree: bpy.types.NodeTree,
    sea: Sea,
    wind: tuple[Wave, ...],
    swell: tuple[Wave, ...],
    survivors: tuple[Wave, ...],
    pixel: _Pixel,
    gust: bpy.types.NodeSocket | None,
    smooth: bpy.types.NodeSocket | None,
) -> tuple[bpy.types.NodeSocket, bpy.types.NodeSocket]:
    """The slope variance each pixel does not draw, along and across its view: the
    gusty sea's, and a slick's where `smooth` is 1. Each is twice its axis's, the total
    an isotropic sea of that slope would hold.

    Cox & Munk's axes are the wind's. ponytail: the view's take their variances by cos^2
    of the angle between and leave out the covariance, so a lobe 45 degrees off the
    wind stays aligned with the view. Rotate the tangent if a glitter shows it.
    """
    speed = sea.wind_speed_mps
    # From or toward, the axis is the same.
    wind_axis = math.radians(sea.wind_from_deg)
    facing = _vector(
        tree,
        "DOT_PRODUCT",
        pixel.along_dir,
        (math.sin(wind_axis), math.cos(wind_axis), 0),
    )
    cos2 = _math(tree, "MULTIPLY", facing, facing, name="sea_wind_facing")
    footprints = (pixel.along_m, pixel.across_m)

    def tabled(
        name: str,
        field: tuple[Wave, ...],
        fit: Callable[[float], tuple[float, float]],
    ) -> tuple[bpy.types.NodeSocket, bpy.types.NodeSocket]:
        axes = tuple(
            _footprint_table(
                f"{name}_{axis}",
                lambda f, i=i: unresolved_slope_variances(
                    speed, wind_axis, field, swell, f, fit
                )[i],
            )
            for i, axis in enumerate(("upwind", "crosswind"))
        )
        out = []
        for f, on_up in zip(
            footprints, (cos2, _math(tree, "SUBTRACT", 1.0, cos2)), strict=True
        ):
            up, cross = (
                _at_footprint(tree, t, f, f"{name}_{axis}")
                for t, axis in zip(axes, ("upwind", "crosswind"), strict=True)
            )
            mixed = _math(
                tree, "MULTIPLY_ADD", _math(tree, "SUBTRACT", up, cross), on_up, cross
            )
            out.append(_math(tree, "MULTIPLY", mixed, 2.0))
        return out[0], out[1]

    unresolved = tabled("sea_unresolved_variance", wind, cox_munk_variances)
    if gust is not None:
        unresolved = _gusty(tree, sea, gust, unresolved)
    if smooth is None:
        return unresolved
    slick = tabled("sea_unresolved_slick_variance", survivors, cox_munk_slick_variances)
    along, across = (
        _math(tree, "MULTIPLY_ADD", smooth, _math(tree, "SUBTRACT", s, v), v)
        for s, v in zip(slick, unresolved, strict=True)
    )
    return along, across
