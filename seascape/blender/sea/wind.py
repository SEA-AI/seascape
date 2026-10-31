"""What the wind draws on the surface: gusts, Langmuir rows, slicks and drift."""

import math
from collections.abc import Callable
from functools import cache
from statistics import NormalDist
from typing import NamedTuple

import bpy
import numpy as np

from seascape.blender.blend import (
    clamped,
    curve_image,
    scalar,
    vector,
)
from seascape.config import Outputs, Sea
from seascape.physics.waves import (
    BEAUFORT_7_MPS,
    BEAUFORT_8_MPS,
    FINE_WINDROW_SPACING_M,
    WINDROW_ASPECT,
    WINDROW_MIN_WIND_MPS,
    WINDROW_SPACING_S,
    foam_drift_mps,
    gaussian_ranks,
    gust_slope_variance,
    turbulence_intensity,
    von_karman_field,
)

# A judgement: gusts to about ten metres, and a tile of dozens of integral scales, so
# its repeat is lost in the distance. Blender's Noise Texture would not repeat, but
# carries no Kaimal spectrum.
GUST_CELLS = 512
GUST_SPACING_M = 4.0
GUST_TILE_M = GUST_CELLS * GUST_SPACING_M
# Numerics: the offsets within a texel, per axis, a lookup's spread is averaged over.
LOOKUP_SHARES = 8
# A judgement: the finer rows' tile, a few texels to a band 3-5 m wide (Zedel & Farmer
# 1991).
FINE_ROWS_TEXEL_M = 0.5


def _drifting(
    tree: bpy.types.NodeTree,
    where: bpy.types.NodeSocket,
    outputs: Outputs,
    tile: bpy.types.Image,
    axes: tuple[tuple[float, float], tuple[float, float]],
    velocity_mps: tuple[float, float],
    start: float,
    name: str,
) -> bpy.types.NodeSocket:
    """`tile` laid on the sea along `axes`, each an (east, north) tile axis in tiles
    per metre, offset `start` tiles, frozen and carried at `velocity_mps`, (east,
    north); read at `where`, (east, north, time).

    A loop crossfades two layers, each carried for one span and back while its weight
    is 0 (Vlachos, "Water flow in Portal 2", SIGGRAPH 2010 course), over the root of
    their squared weights so the variance holds.
    """
    split = tree.nodes.new("ShaderNodeSeparateXYZ")
    tree.links.new(where, split.inputs["Vector"])
    time_s = split.outputs["Z"]
    flat = tree.nodes.new("ShaderNodeCombineXYZ")
    tree.links.new(split.outputs["X"], flat.inputs["X"])
    tree.links.new(split.outputs["Y"], flat.inputs["Y"])
    here = tree.nodes.new("ShaderNodeCombineXYZ")
    here.name = f"{name}_axes"
    for axis, out in zip(axes, ("X", "Y"), strict=True):
        along = vector(tree, "DOT_PRODUCT", flat.outputs["Vector"])
        along.node.inputs["Vector_001"].default_value = (*axis, 0.0)
        tree.links.new(along, here.inputs[out])
    # The tile read at x - v t is the tile at x carried v t.
    carry = [-(velocity_mps[0] * e + velocity_mps[1] * n) for e, n in axes]

    def layer(carried_s: bpy.types.NodeSocket, shift: float) -> bpy.types.NodeSocket:
        at = vector(tree, "MULTIPLY_ADD", name=f"{name}_carry")
        at.node.inputs["Vector"].default_value = (*carry, 0.0)
        tree.links.new(carried_s, at.node.inputs["Vector_001"])
        at.node.inputs["Vector_002"].default_value = (shift, shift, 0.0)
        texture = tree.nodes.new("ShaderNodeTexImage")
        texture.image = tile
        texture.extension = "REPEAT"
        tree.links.new(
            vector(tree, "ADD", here.outputs["Vector"], at),
            texture.inputs["Vector"],
        )
        return texture.outputs["Color"]

    # A tile that holds still loops as it is.
    if not outputs.loop or velocity_mps == (0.0, 0.0):
        return layer(time_s, start)
    span_s = outputs.span_s
    weighted = []
    # Half a tile apart, so the two layers are independent.
    for i, lag in enumerate((0.0, 0.5)):
        cycle = scalar(
            tree,
            "FRACT",
            scalar(tree, "MULTIPLY_ADD", time_s, 1 / span_s, lag),
        )
        weight = scalar(
            tree,
            "SUBTRACT",
            1.0,
            scalar(tree, "ABSOLUTE", scalar(tree, "MULTIPLY_ADD", cycle, 2.0, -1.0)),
            name=f"{name}_weight_{i}",
        )
        carried_s = scalar(tree, "MULTIPLY", cycle, span_s, name=f"{name}_carried_{i}")
        weighted.append((weight, layer(carried_s, start + lag)))
    (w0, n0), (w1, n1) = weighted
    norm = scalar(
        tree,
        "SQRT",
        scalar(tree, "MULTIPLY_ADD", w0, w0, scalar(tree, "MULTIPLY", w1, w1)),
    )
    return scalar(
        tree,
        "DIVIDE",
        scalar(tree, "MULTIPLY_ADD", w0, n0, scalar(tree, "MULTIPLY", w1, n1)),
        norm,
    )


def surface_drift(sea: Sea, outputs: Outputs) -> tuple[float, float]:
    """(east, north) m/s the surface drifts at, carrying foam and windrows alike; a
    loop's holds still, as the foam memory's drift would jump at its seam."""
    if outputs.loop:
        return 0.0, 0.0
    east, north = _downwind(sea)
    return foam_drift_mps(sea.wind_speed_mps, math.atan2(east, north))


def _downwind(sea: Sea) -> tuple[float, float]:
    # Wind is named for where it blows from.
    toward = math.radians(sea.wind_from_deg + 180.0)
    return math.sin(toward), math.cos(toward)


def _gust(
    tree: bpy.types.NodeTree,
    sea: Sea,
    where: bpy.types.NodeSocket,
    outputs: Outputs,
    tile: bpy.types.Image,
) -> bpy.types.NodeSocket:
    """The wind over its mean, less one, carried downwind at the mean wind; read at
    `where`, (east, north, time)."""
    per_m = 1 / GUST_TILE_M
    east, north = _downwind(sea)
    speed = sea.wind_speed_mps
    unit = _drifting(
        tree,
        where,
        outputs,
        tile,
        ((per_m, 0.0), (0.0, per_m)),
        (speed * east, speed * north),
        0.0,
        "gust",
    )
    return scalar(tree, "MULTIPLY", unit, turbulence_intensity(speed), name="gust")


class _RowRead(NamedTuple):
    """A row field's unit Gaussian convergence, made when first read, as an image
    nothing reads is not saved; and the rows' spacing."""

    read: Callable[[], bpy.types.NodeSocket]
    spacing_m: float


class _Rows(NamedTuple):
    windrows: _RowRead
    fine: _RowRead


def _langmuir_rows(
    tree: bpy.types.NodeTree,
    sea: Sea,
    where: bpy.types.NodeSocket,
    outputs: Outputs,
    rng: np.random.Generator,
) -> _Rows:
    """Langmuir circulation's convergence, which gathers slicks and aged foam alike:
    the windrows and the finer rows between them, each Gaussian at unit sigma as
    Cycles reads it, so a
    symmetric clip keeps its mean and a quantile its share. `rng` draws the windrows
    first, as every reader reads them first.

    Limitation: the integral scale stands for the windrow spacing, a judgement, and
    there are rows at every wind, where below the cutoff real slicks lie in patches.
    Give the slicks their own spectrum if the rows read wrong.
    """
    spacing_m = WINDROW_SPACING_S * max(sea.wind_speed_mps, WINDROW_MIN_WIND_MPS)
    fine_tile_m = GUST_CELLS * FINE_ROWS_TEXEL_M

    def windrows() -> bpy.types.NodeSocket:
        tile = _unit(
            gaussian_ranks(von_karman_field(rng, GUST_CELLS, GUST_SPACING_M, spacing_m))
        )
        return _rows(tree, sea, where, outputs, tile, GUST_TILE_M, "windrows")

    def fine() -> bpy.types.NodeSocket:
        tile = _unit(
            gaussian_ranks(
                von_karman_field(
                    rng, GUST_CELLS, FINE_ROWS_TEXEL_M, FINE_WINDROW_SPACING_M
                )
            )
        )
        return _rows(tree, sea, where, outputs, tile, fine_tile_m, "fine_rows")

    return _Rows(
        _RowRead(cache(windrows), spacing_m),
        _RowRead(cache(fine), FINE_WINDROW_SPACING_M),
    )


def _looked_up_sigma(tile: np.ndarray) -> float:
    """`tile`'s spread as Cycles' linear lookup with repeat reads it, between texels as
    on them: interpolation averages neighbours, so it is under the texels' own.

    A lookup a share along a texel blends the same four neighbours, with the same
    weights, wherever it reads: its variance over the tile is those weights' quadratic
    form in the neighbours' covariance."""
    neighbours = np.stack(
        [np.roll(tile, (-dy, -dx), (0, 1)).ravel() for dy in (0, 1) for dx in (0, 1)]
    )
    covariance = np.cov(neighbours, bias=True)
    shares = (np.arange(LOOKUP_SHARES) + 0.5) / LOOKUP_SHARES
    blend = np.stack([1 - shares, shares], axis=-1)
    # (share up, share across) x (neighbour up, neighbour across), as `neighbours`.
    weights = np.einsum("ai,bj->abij", blend, blend).reshape(-1, 4)
    variance = np.einsum("si,ij,sj->s", weights, covariance, weights).mean()
    return float(np.sqrt(variance))


def _unit(tile: np.ndarray) -> np.ndarray:
    """`tile` at unit sigma as Cycles' linear lookup reads it."""
    return tile / _looked_up_sigma(tile)


def _rows(
    tree: bpy.types.NodeTree,
    sea: Sea,
    where: bpy.types.NodeSocket,
    outputs: Outputs,
    tile: np.ndarray,
    tile_m: float,
    name: str,
) -> bpy.types.NodeSocket:
    """`tile`, `tile_m` a side, stretched along the wind and drifting with the surface;
    read at `where`, (east, north, time)."""
    east, north = _downwind(sea)
    across = 1 / tile_m
    along = across / WINDROW_ASPECT
    return _drifting(
        tree,
        where,
        outputs,
        curve_image(f"sea_{name}", tile),
        ((north * across, -east * across), (east * along, north * along)),
        surface_drift(sea, outputs),
        0.0,
        name,
    )


def _slick(
    tree: bpy.types.NodeTree, sea: Sea, rows: _Rows
) -> bpy.types.NodeSocket | None:
    """1 under a slick: where the windrows' convergence is in its top `slick_cover`;
    None with no slicks.

    Limitation: whitecaps break as on the clean sea; simulate the foam memory on the
    survivors under a slick if slicks foam too often.
    """
    if sea.slick_cover <= 0.0:
        return None
    # The tile reads as a unit Gaussian, so its top `slick_cover` lies above this.
    threshold = NormalDist().inv_cdf(1.0 - sea.slick_cover)
    return scalar(tree, "GREATER_THAN", rows.windrows.read(), threshold, name="slick")


def _streaking(sea: Sea) -> float:
    """How far foam is blown into streaks: 0 below Beaufort 7, 1 from force 8."""
    # A judgement: streaks from none at force 7 to full strength at force 8.
    share = (sea.wind_speed_mps - BEAUFORT_7_MPS) / (BEAUFORT_8_MPS - BEAUFORT_7_MPS)
    return min(max(share, 0.0), 1.0)


def _gathered(
    tree: bpy.types.NodeTree,
    sea: Sea,
    footprint_m: bpy.types.NodeSocket,
    rows: _Rows,
) -> tuple[bpy.types.NodeSocket, ...]:
    """How hard each of `rows` pulls aged foam into its convergence g: s g, s rising
    from Beaufort 7's streaks to 8's; none below force 7."""
    share = _streaking(sea)
    if share == 0.0:
        return ()
    pulls = []
    for read, spacing_m in rows:
        # A pixel wider than half the rows' spacing cannot draw them: they fade, gone
        # at a whole spacing.
        fade = clamped(
            tree, scalar(tree, "MULTIPLY_ADD", footprint_m, -2 / spacing_m, 2.0)
        )
        pulls.append(
            scalar(tree, "MULTIPLY", read(), scalar(tree, "MULTIPLY", fade, share))
        )
    return tuple(pulls)


def _gusty(
    tree: bpy.types.NodeTree,
    sea: Sea,
    gust: bpy.types.NodeSocket,
    unresolved: tuple[bpy.types.NodeSocket, bpy.types.NodeSocket],
) -> tuple[bpy.types.NodeSocket, bpy.types.NodeSocket]:
    """`unresolved` with Cox & Munk's variance at the local wind for the mean's.

    Limitation: drawn waves and swell hold the mean wind, as waves long enough to draw
    grow too slowly to follow a gust (Plant 1982); a close camera that draws its
    capillaries sees no gust in them. Scale the drawn slopes below the growth cutoff if
    that shows. The tile is not filtered to the footprint either: past the pixel the
    samples average it, until too few fall in one; filter it to the footprint if
    distant gusts shimmer.
    """
    offset = scalar(
        tree,
        "MULTIPLY",
        gust,
        gust_slope_variance(sea.wind_speed_mps),
        name="sea_gust_variance",
    )
    # A lull can take more than the pixel leaves.
    return tuple(
        scalar(tree, "MAXIMUM", scalar(tree, "ADD", variance, offset), 0.0)
        for variance in unresolved
    )
