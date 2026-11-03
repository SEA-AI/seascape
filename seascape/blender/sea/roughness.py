"""The waves a pixel does not draw, as the slope variance of its lobe."""

import math
from collections.abc import Callable

import bpy

from seascape.blender.blend import (
    curve_image,
    lookup,
    scalar,
    vector,
)
from seascape.blender.sea.pixel import (
    _at_footprint,
    _footprint_aspect_coords,
    _footprint_table,
    _Pixel,
    footprint_aspect_grid,
)
from seascape.blender.sea.wind import _gusty
from seascape.config import Sea
from seascape.physics.earth import GRAVITY_MS2
from seascape.physics.waves import (
    Wave,
    cox_munk_slick_variances,
    cox_munk_variances,
    mss_spread,
    unresolved_view_slope_variances,
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

    Limitation: EO keeps the mean, as the glitter carries its own spread. Scale its lobe
    too if its sea reads too even.
    """
    if sea.wind_speed_mps <= 0.0:
        return unresolved
    link = tree.links.new
    footprint = scalar(
        tree, "SQRT", scalar(tree, "MULTIPLY", pixel.along_m, pixel.across_m)
    )
    # Judgement: square cells, a power of two across, one to two to the footprint's
    # geometric mean, so few fall in a pixel and their spread stays narrow.
    level = scalar(tree, "FLOOR", scalar(tree, "LOGARITHM", footprint, 2.0))
    cell_m = scalar(tree, "POWER", 2.0, level, name="sea_group_cell")
    spread = _at_footprint(
        tree,
        _footprint_table(
            "sea_mss_spread",
            lambda f: mss_spread(sea.wind_speed_mps, sea.fetch_m, f),
        ),
        footprint,
        "sea_mss_spread",
    )
    kept = scalar(
        tree,
        "MULTIPLY",
        _box_kept(tree, scalar(tree, "DIVIDE", pixel.along_m, cell_m)),
        _box_kept(tree, scalar(tree, "DIVIDE", pixel.across_m, cell_m)),
    )
    per_cell = scalar(tree, "DIVIDE", spread, scalar(tree, "SQRT", kept))
    sigma_sq = scalar(
        tree,
        "LOGARITHM",
        scalar(tree, "MULTIPLY_ADD", per_cell, per_cell, 1.0),
        math.e,
    )
    split = tree.nodes.new("ShaderNodeSeparateXYZ")
    link(where, split.inputs["Vector"])
    cell = tree.nodes.new("ShaderNodeCombineXYZ")
    for axis in "XY":
        link(
            scalar(tree, "FLOOR", scalar(tree, "DIVIDE", split.outputs[axis], cell_m)),
            cell.inputs[axis],
        )
    link(level, cell.inputs["Z"])
    offset = tree.nodes.new("ShaderNodeTexWhiteNoise")
    offset.noise_dimensions = "3D"
    link(cell.outputs["Vector"], offset.inputs["Vector"])
    # A cell L across leaves the waves up to L long, `waves.visibility`'s zero. Deep
    # water's group velocity is half sqrt(g lambda / 2 pi), so a group of them crosses
    # it at sqrt(g / (8 pi L)) per second. Read off the cell, every sample in it agrees.
    rate = scalar(
        tree,
        "SQRT",
        scalar(
            tree,
            "DIVIDE",
            GRAVITY_MS2 / (8 * math.pi),
            cell_m,
        ),
    )
    draw = scalar(
        tree,
        "FLOOR",
        scalar(
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
    normal = scalar(
        tree,
        "MULTIPLY",
        scalar(
            tree,
            "SQRT",
            scalar(
                tree,
                "MULTIPLY",
                scalar(
                    tree,
                    "LOGARITHM",
                    scalar(tree, "MAXIMUM", uniform.outputs["X"], 1e-7),
                    math.e,
                ),
                -2.0,
            ),
        ),
        scalar(
            tree,
            "COSINE",
            scalar(tree, "MULTIPLY", uniform.outputs["Y"], 2 * math.pi),
        ),
    )
    # Judgement: a lognormal, so the scale stays positive. exp(sigma z - sigma^2 / 2)
    # has mean 1.
    scale = scalar(
        tree,
        "EXPONENT",
        scalar(
            tree,
            "MULTIPLY_ADD",
            normal,
            scalar(tree, "SQRT", sigma_sq),
            scalar(tree, "MULTIPLY", sigma_sq, -0.5),
        ),
        name="sea_group_scale",
    )
    along, across = unresolved
    return scalar(tree, "MULTIPLY", along, scale), scalar(
        tree, "MULTIPLY", across, scale
    )


def _box_kept(
    tree: bpy.types.NodeTree, cells: bpy.types.NodeSocket
) -> bpy.types.NodeSocket:
    """The share of a cell's variance a box `cells` cells long keeps, at a random
    offset on the grid: the mean over the box of the triangle 1 - |d|, which is
    (1 - 1 / 3m) / m from one cell up and 1 - m / 3 below."""
    over = scalar(
        tree,
        "DIVIDE",
        scalar(tree, "MULTIPLY_ADD", cells, 3.0, -1.0),
        scalar(tree, "MULTIPLY", scalar(tree, "MULTIPLY", cells, cells), 3.0),
    )
    under = scalar(tree, "MULTIPLY_ADD", cells, -1 / 3, 1.0)
    return scalar(
        tree,
        "MULTIPLY_ADD",
        scalar(tree, "GREATER_THAN", cells, 1.0),
        scalar(tree, "SUBTRACT", over, under),
        under,
    )


# The views the unresolved tables are baked for, as quarter turns off the wind.
_TABLED_VIEWS = (("downwind", 0.0), ("crosswind", math.pi / 2))


def _tabled(
    tree: bpy.types.NodeTree,
    name: str,
    sea: Sea,
    field: tuple[Wave, ...],
    swell: tuple[Wave, ...],
    fit: Callable[[float], tuple[float, float]],
    pixel: _Pixel,
    cos2: bpy.types.NodeSocket,
) -> tuple[bpy.types.NodeSocket, bpy.types.NodeSocket]:
    """`unresolved_view_slope_variances` along and across the pixel's view, each
    twice its axis's: baked for a view down and across the wind against the pixel's
    across footprint and along / across, and mixed by `cos2`, the view's cos^2 to
    the wind."""
    wind_axis = math.radians(sea.wind_from_deg)
    x, y = _footprint_aspect_coords(tree, pixel)
    across_m, along_m = footprint_aspect_grid()
    read = {}
    for view, turn in _TABLED_VIEWS:
        variances = unresolved_view_slope_variances(
            sea.wind_speed_mps,
            wind_axis,
            field,
            swell,
            across_m,
            along_m,
            wind_axis + turn,
            fit,
        )
        for axis, variance in zip(("along", "across"), variances, strict=True):
            table = f"{name}_{view}_{axis}"
            read[view, axis] = lookup(tree, curve_image(table, variance), x, y)
            read[view, axis].node.name = table
    along, across = (
        scalar(
            tree,
            "MULTIPLY",
            scalar(
                tree,
                "MULTIPLY_ADD",
                scalar(
                    tree,
                    "SUBTRACT",
                    read["downwind", axis],
                    read["crosswind", axis],
                ),
                cos2,
                read["crosswind", axis],
            ),
            2.0,
        )
        for axis in ("along", "across")
    )
    return along, across


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

    Limitation: between the views down and across the wind, the tables mix by cos^2 of
    the angle and leave out the covariance, so a lobe 45 degrees off the wind stays
    aligned with the view. Rotate the tangent if a glitter shows it.
    """
    # From or toward, the axis is the same.
    wind_axis = math.radians(sea.wind_from_deg)
    facing = vector(
        tree,
        "DOT_PRODUCT",
        pixel.along_dir,
        (math.sin(wind_axis), math.cos(wind_axis), 0),
    )
    cos2 = scalar(tree, "MULTIPLY", facing, facing, name="sea_wind_facing")
    unresolved = _tabled(
        tree,
        "sea_unresolved_variance",
        sea,
        wind,
        swell,
        cox_munk_variances,
        pixel,
        cos2,
    )
    if gust is not None:
        unresolved = _gusty(tree, sea, gust, unresolved)
    if smooth is None:
        return unresolved
    slick = _tabled(
        tree,
        "sea_unresolved_slick_variance",
        sea,
        survivors,
        swell,
        cox_munk_slick_variances,
        pixel,
        cos2,
    )
    along, across = (
        scalar(tree, "MULTIPLY_ADD", smooth, scalar(tree, "SUBTRACT", s, v), v)
        for s, v in zip(slick, unresolved, strict=True)
    )
    return along, across
