"""Whitecaps: their lace and lattice, and the foam a build remembers on its grids."""

import math
from statistics import NormalDist
from typing import NamedTuple

import bpy
import numpy as np

from seascape.blender.blend import (
    CURVE_SAMPLES,
    clamped,
    curve_image,
    lookup,
    pixels,
    scalar,
)
from seascape.blender.sea.wind import _downwind, _drifting, _streaking, surface_drift
from seascape.config import Outputs, Sea
from seascape.physics.wakes import (
    FRESH_FOAM_REFLECTANCE,
)
from seascape.physics.waves import (
    FOAM_VISIBLE_COVER,
    THIN_FOAM_SHARE,
    WHITECAP_EFFECTIVE_REFLECTANCE,
    FoamGrid,
    foam_lattice_field,
    whitecap_fraction,
)

# A judgement: a whitecap outline's ramp, per sigma of its lace, fresh and aged.
FRESH_EDGE = 6.0
AGED_EDGE = 2.0
# A judgement: foam's albedo swings this share of fresh foam's per sigma of density.
FOAM_ALBEDO_SWING = 0.2
# A judgement: how much the lattice frays a whitecap's outline.
LACE_RAGGED_WEIGHT = 0.4
# Numerics: the offsets within a texel, per axis, a lookup's spread is averaged over.
FOAM_LOOKUP_SHARES = 8
# Numerics: a finer foam grid fades in over this many of its cells from its edge.
FOAM_SEAM_CELLS = 8

# A judgement: foam patches a metre or two across, streaked 4:1 along the wake.
FOAM_CELLS = 512
FOAM_SPACING_M = 0.25
FOAM_LENGTH_M = 1.0
FOAM_TILE_M = FOAM_CELLS * FOAM_SPACING_M
FOAM_STREAK = 4.0
# A judgement: aged foam's lattice, the walls of cells 1.3 m and 40 cm across among
# openings of 2 m, over a 16 m tile.
LATTICE_CELLS = 512
LATTICE_TILE_M = 16.0
LATTICE_CELL_SIZES_M = (16 / 12, 16 / 40)
LATTICE_OPENING_M = 2.0
# A judgement: the width of a cell wall, in metres.
LATTICE_WALL_M = 0.15
# A judgement: a lattice edge's ramp, per sigma of the field.
LATTICE_SHARPNESS = 4.0
# A judgement: a second, finer octave of the lattice, weighted so.
LATTICE_OCTAVE = 4.0
LATTICE_FINE_WEIGHT = 0.8
# A judgement: how long a streak of aged foam is for its width at force 8.
LATTICE_STREAK = 4.0


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
    shares = (np.arange(FOAM_LOOKUP_SHARES) + 0.5) / FOAM_LOOKUP_SHARES
    blend = np.stack([1 - shares, shares], axis=-1)
    # (share up, share across) x (neighbour up, neighbour across), as `neighbours`.
    weights = np.einsum("ai,bj->abij", blend, blend).reshape(-1, 4)
    variance = np.einsum("si,ij,sj->s", weights, covariance, weights).mean()
    return float(np.sqrt(variance))


def _remembered(
    tree: bpy.types.NodeTree,
    sea: Sea,
    outputs: Outputs,
    position: bpy.types.NodeSocket,
    footprint_m: bpy.types.NodeSocket,
    textures: "_FoamTexture",
    memory: list[tuple[FoamGrid, bpy.types.Image]],
    pulls: tuple[bpy.types.NodeSocket, ...],
) -> tuple[bpy.types.NodeSocket, bpy.types.NodeSocket, bpy.types.NodeSocket]:
    """How much of the pixel foam covers; that, as a share of fresh foam's brightness;
    and the share of the sea under foam, where its bubbles are: read off the finest
    foam memory that holds `position`, (east, north, time), aged foam `gathered` by the
    windrows, the lace above its quantile covering that share."""
    lace = textures.lace
    split = tree.nodes.new("ShaderNodeSeparateXYZ")
    tree.links.new(position, split.inputs["Vector"])
    drift = surface_drift(sea, outputs)
    # The memory's grid drifts: water at x is at grid x - v t.
    on_grid = [
        scalar(tree, "MULTIPLY_ADD", split.outputs["Z"], -v, split.outputs[axis])
        for axis, v in zip("XY", drift, strict=True)
    ]
    # Past the coarsest grid the foam is finer than a pixel: Monahan's cover, as bright
    # on average as Koepke measured.
    cover: bpy.types.NodeSocket | float = whitecap_fraction(sea.wind_speed_mps)
    breaking: bpy.types.NodeSocket | float = (
        WHITECAP_EFFECTIVE_REFLECTANCE / FRESH_FOAM_REFLECTANCE - THIN_FOAM_SHARE
    ) / (1 - THIN_FOAM_SHARE)
    # Coarsest first, so each finer grid overwrites where it holds.
    for grid, sequence in sorted(memory, key=lambda m: -m[0].cell_m):
        u = scalar(
            tree,
            "MULTIPLY_ADD",
            on_grid[0],
            1 / (grid.columns * grid.cell_m),
            -grid.west_m / (grid.columns * grid.cell_m),
        )
        w = scalar(
            tree,
            "MULTIPLY_ADD",
            on_grid[1],
            1 / (grid.rows * grid.cell_m),
            -grid.south_m / (grid.rows * grid.cell_m),
        )
        at = tree.nodes.new("ShaderNodeCombineXYZ")
        tree.links.new(u, at.inputs["X"])
        tree.links.new(w, at.inputs["Y"])
        texture = tree.nodes.new("ShaderNodeTexImage")
        texture.name = f"sea_foam_{grid.cell_m:g}m"
        texture.image = sequence
        texture.extension = "CLIP"
        # Linear lookup lays the cells' squares into the lace's edge.
        texture.interpolation = "Cubic"
        texture.image_user.frame_duration = len(outputs.times_s)
        texture.image_user.frame_start = 0
        texture.image_user.use_auto_refresh = True
        tree.links.new(at.outputs["Vector"], texture.inputs["Vector"])
        channels = tree.nodes.new("ShaderNodeSeparateColor")
        tree.links.new(texture.outputs["Color"], channels.inputs["Color"])
        # Taken where the pixel spans under a cell, fading out by two, and faded in from
        # its edge, so neither a finer grid's reach nor its border leaves a seam.
        fine_enough = clamped(
            tree, scalar(tree, "MULTIPLY_ADD", footprint_m, -1 / grid.cell_m, 2.0)
        )
        inside = scalar(
            tree,
            "MULTIPLY",
            *(
                clamped(
                    tree,
                    scalar(
                        tree,
                        "MULTIPLY",
                        scalar(tree, "MINIMUM", x, scalar(tree, "SUBTRACT", 1.0, x)),
                        cells / FOAM_SEAM_CELLS,
                    ),
                )
                for x, cells in ((u, grid.columns), (w, grid.rows))
            ),
        )
        inside = scalar(tree, "MULTIPLY", inside, fine_enough)
        cover = scalar(
            tree,
            "MULTIPLY_ADD",
            inside,
            scalar(tree, "SUBTRACT", channels.outputs["Red"], cover),
            cover,
        )
        breaking = scalar(
            tree,
            "MULTIPLY_ADD",
            inside,
            scalar(tree, "SUBTRACT", channels.outputs["Green"], breaking),
            breaking,
        )
    assert isinstance(cover, bpy.types.NodeSocket)
    if pulls:
        # Each pull s g is clipped to [-a, a]: g is Gaussian, so 1 + clamp(s g) averages
        # 1 and Monahan's cover holds; (1 + a)^2 under 1 / cover keeps it under full.
        # The floor keeps INVERSE_SQRT, which reads 0 at 0, off an empty cell.
        headroom = scalar(
            tree,
            "MINIMUM",
            scalar(
                tree,
                "SUBTRACT",
                scalar(tree, "INVERSE_SQRT", scalar(tree, "MAXIMUM", cover, 1e-6)),
                1.0,
            ),
            1.0,
        )
        floor = scalar(tree, "MULTIPLY", headroom, -1.0)
        densities = [
            scalar(
                tree,
                "ADD",
                scalar(tree, "MINIMUM", scalar(tree, "MAXIMUM", pull, floor), headroom),
                1.0,
            )
            for pull in pulls
        ]
        gathered = densities[0]
        for density in densities[1:]:
            gathered = scalar(tree, "MULTIPLY", gathered, density)
        gathered.node.name = "whitecap_gathered"
        # Foam still breaking has not been gathered yet.
        kept = scalar(
            tree,
            "MULTIPLY_ADD",
            breaking,
            scalar(tree, "SUBTRACT", 1.0, gathered),
            gathered,
        )
        cover = scalar(tree, "MULTIPLY", cover, kept)
    levels = (np.arange(CURVE_SAMPLES) + 0.5) / CURVE_SAMPLES
    ppf = curve_image("normal_ppf", np.vectorize(NormalDist().inv_cdf)(1 - levels))
    # Above its 1 - cover quantile, the lace covers `cover` of the pixel.
    laced = scalar(
        tree,
        "MULTIPLY_ADD",
        scalar(tree, "SUBTRACT", lace, lookup(tree, ppf, cover)),
        # Foam still breaking is dense to its edge; aged foam thins out to it.
        scalar(tree, "MULTIPLY_ADD", breaking, FRESH_EDGE - AGED_EDGE, AGED_EDGE),
        0.5,
    )
    laced.node.use_clamp = True
    laced = scalar(
        tree,
        "MULTIPLY",
        laced,
        scalar(tree, "GREATER_THAN", cover, FOAM_VISIBLE_COVER),
        name="whitecaps",
    )
    bright = scalar(
        tree,
        "MULTIPLY_ADD",
        breaking,
        1 - THIN_FOAM_SHARE,
        THIN_FOAM_SHARE,
        name="whitecap_bright",
    )
    foam = _latticed(tree, laced, bright, textures, lookup(tree, ppf, bright))
    under = scalar(tree, "MAXIMUM", cover, 0.0, name="whitecap_bubbles")
    return laced, foam, under


def _latticed(
    tree: bpy.types.NodeTree,
    laced: bpy.types.NodeSocket,
    bright: bpy.types.NodeSocket,
    texture: "_FoamTexture",
    threshold: bpy.types.NodeSocket,
) -> bpy.types.NodeSocket:
    """Foam over the `laced` area at `bright` of fresh foam's brightness, drawn as
    fresh foam's white over `bright` of that area, the lattice above `threshold`, its
    quantile, not a grey film: the same mean. Its albedo, varying with the texture's
    density about fresh foam's, is the node `whitecap_albedo`."""
    held = scalar(
        tree,
        "MULTIPLY_ADD",
        scalar(tree, "SUBTRACT", texture.lattice, threshold),
        LATTICE_SHARPNESS,
        0.5,
    )
    held.node.use_clamp = True
    # A pixel longer than the lattice's walls draws their mean: from one wall's width,
    # gone at two. Its longer side, as a mipmap without anisotropic filtering takes.
    fade = clamped(
        tree, scalar(tree, "MULTIPLY_ADD", texture.pixel_m, 1 / LATTICE_WALL_M, -1.0)
    )
    held = scalar(
        tree, "MULTIPLY_ADD", fade, scalar(tree, "SUBTRACT", bright, held), held
    )
    # The density's symmetric clip keeps the mean albedo fresh foam's; it fades with
    # the walls, and with it the bump it lifts.
    swing = scalar(
        tree,
        "MULTIPLY",
        scalar(tree, "MINIMUM", scalar(tree, "MAXIMUM", texture.shade, -2.0), 2.0),
        scalar(tree, "SUBTRACT", 1.0, fade),
    )
    scalar(
        tree,
        "MULTIPLY_ADD",
        swing,
        FRESH_FOAM_REFLECTANCE * FOAM_ALBEDO_SWING,
        FRESH_FOAM_REFLECTANCE,
        name="whitecap_albedo",
    )
    return scalar(tree, "MULTIPLY", laced, held, name="whitecap_foam")


def _bare(
    tree: bpy.types.NodeTree,
) -> tuple[bpy.types.NodeSocket, bpy.types.NodeSocket, bpy.types.NodeSocket]:
    """No whitecaps: a calm sea, or one no camera sees."""
    area, foam, under = (
        scalar(tree, "MAXIMUM", 0.0, 0.0, name=name)
        for name in ("whitecaps", "whitecap_foam", "whitecap_bubbles")
    )
    return area, foam, under


class _FoamTexture(NamedTuple):
    """Unit Gaussians drifting with the surface: whitecaps' outline, aged foam's
    lattice, and its density; and the pixel's longer side, which fades the lattice."""

    lace: bpy.types.NodeSocket
    lattice: bpy.types.NodeSocket
    shade: bpy.types.NodeSocket
    pixel_m: bpy.types.NodeSocket


def _foam_texture(
    tree: bpy.types.NodeTree,
    sea: Sea,
    where: bpy.types.NodeSocket,
    outputs: Outputs,
    tile: bpy.types.Image,
    rng: np.random.Generator,
    pixel_m: bpy.types.NodeSocket,
) -> _FoamTexture:
    """Whitecap foam's texture read at `where`, (east, north, time): the lace off
    `tile`, frayed by a lattice drawn from `rng`, and that lattice twice more, for a
    pixel `pixel_m` long."""
    lattice = curve_image(
        "sea_foam_lattice",
        foam_lattice_field(
            rng,
            LATTICE_CELLS,
            LATTICE_TILE_M,
            LATTICE_CELL_SIZES_M,
            LATTICE_OPENING_M,
        ),
    )
    # A quarter tile apart, each read is independent of the others.
    ragged, holes, shade = (
        _foam_lattice(tree, sea, where, outputs, lattice, start)
        for start in (0.25, 0.0, 0.5)
    )
    return _FoamTexture(
        _foam_lace(tree, sea, where, outputs, tile, ragged), holes, shade, pixel_m
    )


def _foam_lace(
    tree: bpy.types.NodeTree,
    sea: Sea,
    where: bpy.types.NodeSocket,
    outputs: Outputs,
    tile: bpy.types.Image,
    ragged: bpy.types.NodeSocket,
) -> bpy.types.NodeSocket:
    """Whitecap foam's lace, a unit Gaussian drifting with the surface, read at `where`,
    (east, north, time): a read of `tile` and the `ragged` lattice, rescaled to one."""
    sigma = _looked_up_sigma(pixels(tile)[..., 0])
    east, north = _downwind(sea)
    per_m = 1 / FOAM_TILE_M
    along = per_m / FOAM_STREAK
    coarse = _drifting(
        tree,
        where,
        outputs,
        tile,
        ((along * east, along * north), (per_m * north, -per_m * east)),
        surface_drift(sea, outputs),
        0.0,
        "whitecap_lace",
    )
    coarse = scalar(tree, "MULTIPLY", coarse, 1 / sigma)
    weight = LACE_RAGGED_WEIGHT
    summed = scalar(tree, "MULTIPLY_ADD", ragged, weight, coarse)
    return scalar(tree, "MULTIPLY", summed, 1 / math.sqrt(1 + weight**2))


def _foam_lattice(
    tree: bpy.types.NodeTree,
    sea: Sea,
    where: bpy.types.NodeSocket,
    outputs: Outputs,
    tile: bpy.types.Image,
    start: float = 0.0,
) -> bpy.types.NodeSocket:
    """Aged foam's lattice, a unit Gaussian drifting with the surface, read at
    `where`, (east, north, time), offset `start` tiles."""
    sigma = _looked_up_sigma(pixels(tile)[..., 0])
    east, north = _downwind(sea)
    layers = []
    # From force 7 the lattice is blown into streaks along the wind.
    stretch = 1 + (LATTICE_STREAK - 1) * _streaking(sea)
    for scale, shift in ((1.0, start), (LATTICE_OCTAVE, start + 0.5)):
        per_m = scale / LATTICE_TILE_M
        along = per_m / stretch
        layers.append(
            _drifting(
                tree,
                where,
                outputs,
                tile,
                ((along * east, along * north), (per_m * north, -per_m * east)),
                surface_drift(sea, outputs),
                shift,
                "whitecap_lattice",
            )
        )
    weight = LATTICE_FINE_WEIGHT
    summed = scalar(tree, "MULTIPLY_ADD", layers[1], weight, layers[0])
    return scalar(tree, "MULTIPLY", summed, 1 / (sigma * math.sqrt(1 + weight**2)))
