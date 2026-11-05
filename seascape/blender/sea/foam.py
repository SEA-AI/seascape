"""Whitecaps: their lace and lattice, shaded from the foam memory."""

import math
from statistics import NormalDist
from typing import NamedTuple

import bpy
import numpy as np

from seascape.blender.blend import (
    Num,
    clamped,
    combine,
    curve_image,
    lookup,
    node,
    scalar,
    separate,
    texels,
)
from seascape.blender.sea.wind import (
    _downwind,
    _drifting,
    _streaking,
    _Surface,
    _unit,
)
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


def _remembered(
    tree: bpy.types.NodeTree,
    surface: _Surface,
    footprint_m: bpy.types.NodeSocket,
    textures: "_FoamTexture",
    memory: list[tuple[FoamGrid, bpy.types.Image]],
    pulls: tuple[bpy.types.NodeSocket, ...],
) -> "_Whitecaps":
    """The whitecaps read off the finest foam memory that holds the `surface`'s
    point, aged foam `gathered` by the windrows, the lace above its quantile covering
    that share."""
    ppf = textures.ppf
    lace = textures.lace
    east, north, time_s = separate(tree, surface.where)
    # The memory's grid drifts: water at x is at grid x - v t.
    on_grid = [
        Num(tree, time_s).madd(-v, axis)
        for axis, v in zip((east, north), surface.drift, strict=True)
    ]
    # Past the coarsest grid the foam is finer than a pixel: Monahan's cover, as bright
    # on average as Koepke measured.
    cover: Num | float = whitecap_fraction(surface.sea.wind_speed_mps)
    breaking: Num | float = (
        WHITECAP_EFFECTIVE_REFLECTANCE / FRESH_FOAM_REFLECTANCE - THIN_FOAM_SHARE
    ) / (1 - THIN_FOAM_SHARE)
    # Coarsest first, so each finer grid overwrites where it holds.
    for grid, sequence in sorted(memory, key=lambda m: -m[0].cell_m):
        u = on_grid[0].madd(
            1 / (grid.columns * grid.cell_m),
            -grid.west_m / (grid.columns * grid.cell_m),
        )
        w = on_grid[1].madd(
            1 / (grid.rows * grid.cell_m), -grid.south_m / (grid.rows * grid.cell_m)
        )
        texture = node(
            tree,
            "ShaderNodeTexImage",
            {"Vector": combine(tree, u.socket, w.socket)},
            name=f"sea_foam_{grid.cell_m:g}m",
            image=sequence,
            extension="CLIP",
            # Linear lookup lays the cells' squares into the lace's edge.
            interpolation="Cubic",
        )
        texture.image_user.frame_duration = len(surface.outputs.times_s)
        texture.image_user.frame_start = 0
        texture.image_user.use_auto_refresh = True
        channels = node(
            tree, "ShaderNodeSeparateColor", {"Color": texture.outputs["Color"]}
        )
        # Taken where the pixel spans under a cell, fading out by two, and faded in from
        # its edge, so neither a finer grid's reach nor its border leaves a seam.
        fine_enough = Num(
            tree,
            clamped(
                tree,
                Num(tree, footprint_m).madd(-1 / grid.cell_m, 2.0).socket,
            ),
        )
        inside_u, inside_w = (
            Num(
                tree,
                clamped(
                    tree,
                    (x.op("MINIMUM", 1.0 - x) * (cells / FOAM_SEAM_CELLS)).socket,
                ),
            )
            for x, cells in ((u, grid.columns), (w, grid.rows))
        )
        inside = inside_u * inside_w * fine_enough
        cover = inside.madd(Num(tree, channels.outputs["Red"]) - cover, cover)
        breaking = inside.madd(
            Num(tree, channels.outputs["Green"]) - breaking, breaking
        )
    assert isinstance(cover, Num) and isinstance(breaking, Num)
    if pulls:
        # Each pull s g is clipped to [-a, a]: g is Gaussian, so 1 + clamp(s g) averages
        # 1 and Monahan's cover holds; (1 + a)^2 under 1 / cover keeps it under full.
        # The floor keeps INVERSE_SQRT, which reads 0 at 0, off an empty cell.
        headroom = (cover.op("MAXIMUM", 1e-6).op("INVERSE_SQRT") - 1.0).op(
            "MINIMUM", 1.0
        )
        floor = headroom * -1.0
        densities = [
            Num(tree, pull).op("MAXIMUM", floor).op("MINIMUM", headroom) + 1.0
            for pull in pulls
        ]
        gathered = densities[0]
        for density in densities[1:]:
            gathered = gathered * density
        gathered.socket.node.name = "whitecap_gathered"
        # Foam still breaking has not been gathered yet.
        cover = cover * breaking.madd(1.0 - gathered, gathered)
    # Above its 1 - cover quantile, the lace covers `cover` of the pixel.
    laced = (Num(tree, lace) - lookup(tree, ppf, cover.socket)).madd(
        # Foam still breaking is dense to its edge; aged foam thins out to it.
        breaking.madd(FRESH_EDGE - AGED_EDGE, AGED_EDGE),
        0.5,
    )
    laced.socket.node.use_clamp = True
    laced = scalar(
        tree,
        "MULTIPLY",
        laced.socket,
        cover.op("GREATER_THAN", FOAM_VISIBLE_COVER).socket,
        name="whitecaps",
    )
    bright = scalar(
        tree,
        "MULTIPLY_ADD",
        breaking.socket,
        1 - THIN_FOAM_SHARE,
        THIN_FOAM_SHARE,
        name="whitecap_bright",
    )
    foam, albedo = _latticed(tree, laced, bright, textures, lookup(tree, ppf, bright))
    under = scalar(tree, "MAXIMUM", cover.socket, 0.0, name="whitecap_bubbles")
    return _Whitecaps(laced, foam, under, albedo)


def _latticed(
    tree: bpy.types.NodeTree,
    laced: bpy.types.NodeSocket,
    bright: bpy.types.NodeSocket,
    texture: "_FoamTexture",
    threshold: bpy.types.NodeSocket,
) -> tuple[bpy.types.NodeSocket, bpy.types.NodeSocket]:
    """Foam over the `laced` area at `bright` of fresh foam's brightness, drawn as
    fresh foam's white over `bright` of that area, the lattice above `threshold`, its
    quantile, not a grey film: the same mean; and its albedo, varying with the
    texture's density about fresh foam's."""
    held = (Num(tree, texture.lattice) - threshold).madd(LATTICE_SHARPNESS, 0.5)
    held.socket.node.use_clamp = True
    # A pixel longer than the lattice's walls draws their mean: from one wall's width,
    # gone at two. Its longer side, as a mipmap without anisotropic filtering takes.
    fade = Num(
        tree,
        clamped(
            tree,
            Num(tree, texture.pixel_m).madd(1 / LATTICE_WALL_M, -1.0).socket,
        ),
    )
    held = fade.madd(Num(tree, bright) - held, held)
    # The density's symmetric clip keeps the mean albedo fresh foam's; it fades with
    # the walls, and with it the bump it lifts.
    swing = Num(tree, texture.shade).op("MAXIMUM", -2.0).op("MINIMUM", 2.0) * (
        1.0 - fade
    )
    albedo = scalar(
        tree,
        "MULTIPLY_ADD",
        swing.socket,
        FRESH_FOAM_REFLECTANCE * FOAM_ALBEDO_SWING,
        FRESH_FOAM_REFLECTANCE,
        name="whitecap_albedo",
    )
    return scalar(tree, "MULTIPLY", laced, held.socket, name="whitecap_foam"), albedo


class _Whitecaps(NamedTuple):
    """The share of the pixel foam covers; that as a share of fresh foam's brightness;
    the share bubbles lie under; and the foam's albedo, None with no whitecaps."""

    area: bpy.types.NodeSocket
    foam: bpy.types.NodeSocket
    bubbles: bpy.types.NodeSocket
    albedo: bpy.types.NodeSocket | None


def _bare(tree: bpy.types.NodeTree) -> _Whitecaps:
    """No whitecaps: a calm sea, or one no camera sees."""
    area, foam, under = (
        scalar(tree, "MAXIMUM", 0.0, 0.0, name=name)
        for name in ("whitecaps", "whitecap_foam", "whitecap_bubbles")
    )
    return _Whitecaps(area, foam, under, None)


def _quantiles() -> np.ndarray:
    """The unit normal's 1 - p quantile, sampled in p."""
    return np.vectorize(NormalDist().inv_cdf)(1 - texels())


class _FoamTexture(NamedTuple):
    """Unit Gaussians drifting with the surface: whitecaps' outline, aged foam's
    lattice, and its density; the pixel's longer side, which fades the lattice; and the
    unit normal's quantiles, which threshold them."""

    lace: bpy.types.NodeSocket
    lattice: bpy.types.NodeSocket
    shade: bpy.types.NodeSocket
    pixel_m: bpy.types.NodeSocket
    ppf: bpy.types.Image


def _foam_texture(
    tree: bpy.types.NodeTree,
    surface: _Surface,
    tile: bpy.types.Image,
    rng: np.random.Generator,
    pixel_m: bpy.types.NodeSocket,
    ppf: bpy.types.Image,
) -> _FoamTexture:
    """Whitecap foam's texture on the `surface`: the lace off `tile`, frayed by a
    lattice drawn from `rng`, and that lattice twice more, for a pixel `pixel_m`
    long."""
    lattice = curve_image(
        "sea_foam_lattice",
        _unit(
            foam_lattice_field(
                rng,
                LATTICE_CELLS,
                LATTICE_TILE_M,
                LATTICE_CELL_SIZES_M,
                LATTICE_OPENING_M,
            )
        ),
    )
    # A quarter tile apart, each read is independent of the others.
    ragged, holes, shade = (
        _foam_lattice(tree, surface, lattice, start) for start in (0.25, 0.0, 0.5)
    )
    return _FoamTexture(
        _foam_lace(tree, surface, tile, ragged), holes, shade, pixel_m, ppf
    )


def _foam_lace(
    tree: bpy.types.NodeTree,
    surface: _Surface,
    tile: bpy.types.Image,
    ragged: bpy.types.NodeSocket,
) -> bpy.types.NodeSocket:
    """Whitecap foam's lace, a unit Gaussian drifting with the `surface`: a read of
    the unit `tile` summed with the `ragged` lattice, rescaled to one."""
    east, north = _downwind(surface.sea)
    per_m = 1 / FOAM_TILE_M
    along = per_m / FOAM_STREAK
    coarse = _drifting(
        tree,
        surface,
        tile,
        ((along * east, along * north), (per_m * north, -per_m * east)),
        surface.drift,
        0.0,
        "whitecap_lace",
    )
    weight = LACE_RAGGED_WEIGHT
    summed = Num(tree, ragged).madd(weight, coarse)
    return (summed * (1 / math.sqrt(1 + weight**2))).socket


def _foam_lattice(
    tree: bpy.types.NodeTree,
    surface: _Surface,
    tile: bpy.types.Image,
    start: float,
) -> bpy.types.NodeSocket:
    """Aged foam's lattice, a unit Gaussian drifting with the `surface`, offset
    `start` tiles."""
    east, north = _downwind(surface.sea)
    layers = []
    # From force 7 the lattice is blown into streaks along the wind.
    stretch = 1 + (LATTICE_STREAK - 1) * _streaking(surface.sea)
    for scale, shift in ((1.0, start), (LATTICE_OCTAVE, start + 0.5)):
        per_m = scale / LATTICE_TILE_M
        along = per_m / stretch
        layers.append(
            _drifting(
                tree,
                surface,
                tile,
                ((along * east, along * north), (per_m * north, -per_m * east)),
                surface.drift,
                shift,
                "whitecap_lattice",
            )
        )
    weight = LATTICE_FINE_WEIGHT
    summed = Num(tree, layers[1]).madd(weight, layers[0])
    return (summed * (1 / math.sqrt(1 + weight**2))).socket
