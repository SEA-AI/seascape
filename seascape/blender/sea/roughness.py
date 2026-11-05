"""The waves a pixel does not draw, as the slope variance of its lobe."""

import math
from collections.abc import Callable

import bpy

from seascape.blender.blend import (
    Num,
    combine,
    scalar,
    separate,
    vector,
)
from seascape.blender.sea.pixel import (
    _at_footprint,
    _footprint_table,
    _Pixel,
    _redraw,
)
from seascape.blender.sea.wind import _gusty
from seascape.config import Sea
from seascape.physics.earth import GRAVITY_MS2
from seascape.physics.waves import (
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
    """Unresolved slope variances scaled per cell by a lognormal of mean 1, so a
    pixel's footprint holds `mss_spread` of its mean square slope as the short waves'
    groups do. Rough footprints read warm at grazing views.

    A grazing footprint spans several cells along and part of one across. Each cell is
    redrawn when a group of the waves a footprint its own size fades crosses it, at
    their group velocity.

    Limitation: EO keeps the mean, as the glitter carries its own spread. Scale its lobe
    too if its sea reads too even.

    Args:
        tree: Where the nodes go.
        sea: The wind and fetch the spread is taken at; no wind keeps the mean.
        where: The flat sea's (east, north, time) the cells are drawn at.
        pixel: The footprint the cells are sized to.
        unresolved: The variances along and across the view.
    """
    if sea.wind_speed_mps <= 0.0:
        return unresolved
    along_m, across_m = Num(tree, pixel.along_m), Num(tree, pixel.across_m)
    footprint = (along_m * across_m).op("SQRT")
    # Judgement: square cells, a power of two across, one to two to the footprint's
    # geometric mean, so few fall in a pixel and their spread stays narrow.
    level = footprint.op("LOGARITHM", 2.0).op("FLOOR")
    cell_m = level.rop("POWER", 2.0)
    cell_m.socket.node.name = "sea_group_cell"
    spread = _at_footprint(
        tree,
        _footprint_table(
            "sea_mss_spread",
            lambda f: mss_spread(sea.wind_speed_mps, sea.fetch_m, f),
        ),
        footprint.socket,
        "sea_mss_spread",
    )
    kept = _box_kept(along_m / cell_m) * _box_kept(across_m / cell_m)
    per_cell = Num(tree, spread) / kept.op("SQRT")
    sigma_sq = per_cell.madd(per_cell, 1.0).op("LOGARITHM", math.e)
    east, north, time_s = separate(tree, where)
    cell = combine(
        tree,
        *((Num(tree, axis) / cell_m).op("FLOOR").socket for axis in (east, north)),
        level.socket,
    )
    # A cell L across leaves the waves up to L long, `waves.visibility`'s zero. Deep
    # water's group velocity is half sqrt(g lambda / 2 pi), so a group of them crosses
    # it at sqrt(g / (8 pi L)) per second. Read off the cell, every sample in it agrees.
    rate = (GRAVITY_MS2 / (8 * math.pi) / cell_m).op("SQRT")
    log_u, angle_u = _redraw(tree, cell, time_s, rate.socket, "sea_group_draw")
    # Box-Muller.
    normal = (log_u * -2.0).op("SQRT") * (angle_u * (2 * math.pi)).op("COSINE")
    # Judgement: a lognormal, so the scale stays positive. exp(sigma z - sigma^2 / 2)
    # has mean 1.
    scale = normal.madd(sigma_sq.op("SQRT"), sigma_sq * -0.5).op("EXPONENT")
    scale.socket.node.name = "sea_group_scale"
    along, across = (Num(tree, v) * scale for v in unresolved)
    return along.socket, across.socket


def _box_kept(cells: Num) -> Num:
    """The share of a cell's variance a box keeps, at a random offset on the grid: the
    mean over the box of the triangle 1 - |d|, which is (1 - 1 / 3m) / m from one cell
    up and 1 - m / 3 below.

    Args:
        cells: The box's length m, in cells."""
    over = cells.madd(3.0, -1.0) / (cells * cells * 3.0)
    under = cells.madd(-1 / 3, 1.0)
    return cells.op("GREATER_THAN", 1.0).madd(over - under, under)


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
    gusty sea's, and a slick's where it is smooth. Each is twice its axis's, the total
    an isotropic sea of that slope would hold.

    Cox & Munk's axes are the wind's. Limitation: the view's take their variances by
    cos^2 of the angle between and leave out the covariance, so a lobe 45 degrees off
    the wind stays aligned with the view. Rotate the tangent if a glitter shows it.

    Args:
        tree: Where the nodes go.
        sea: The wind the variances are fitted at.
        wind: The wind sea's shaded waves.
        swell: The swell's shaded waves.
        survivors: The wind sea's waves a slick leaves.
        pixel: The footprint and view direction.
        gust: The wind over its mean, less one; None for a steady wind.
        smooth: Where the variance is a slick's, under a slick or in a wake, 1 at
            full; None for neither.
    """
    speed = sea.wind_speed_mps
    # From or toward, the axis is the same.
    wind_axis = math.radians(sea.wind_from_deg)
    facing = vector(
        tree,
        "DOT_PRODUCT",
        pixel.along_dir,
        (math.sin(wind_axis), math.cos(wind_axis), 0),
    )
    cos2 = Num(tree, scalar(tree, "MULTIPLY", facing, facing, name="sea_wind_facing"))
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
        for f, on_up in zip(footprints, (cos2, 1.0 - cos2), strict=True):
            up, cross = (
                Num(tree, _at_footprint(tree, t, f, f"{name}_{axis}"))
                for t, axis in zip(axes, ("upwind", "crosswind"), strict=True)
            )
            out.append(((up - cross).madd(on_up, cross) * 2.0).socket)
        return out[0], out[1]

    unresolved = tabled("sea_unresolved_variance", wind, cox_munk_variances)
    if gust is not None:
        unresolved = _gusty(tree, sea, gust, unresolved)
    if smooth is None:
        return unresolved
    slick = tabled("sea_unresolved_slick_variance", survivors, cox_munk_slick_variances)
    along, across = (
        Num(tree, smooth).madd(Num(tree, s) - v, v).socket
        for s, v in zip(slick, unresolved, strict=True)
    )
    return along, across
