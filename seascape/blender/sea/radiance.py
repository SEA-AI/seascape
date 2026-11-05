"""The LWIR sea: its emissivity, its skin's radiance and the sky it reflects."""

import math
import os
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import NamedTuple

import bpy
import numpy as np

from seascape.blender.blend import (
    Num,
    curve_image,
    lookup,
    node,
    scalar,
    separate,
    texels,
    vector,
)
from seascape.blender.sea.daylight import _lobe, _stretchable, _widest
from seascape.blender.sea.pixel import _Pixel
from seascape.blender.sea.wakes import _Wakes
from seascape.config import Sea
from seascape.physics import lwir

# A judgement: emissivity and the reflected sky move slowly with slope.
SLOPE_ROWS = 16
# Measured against `lwir.Sky.reflected` evaluated exactly: bilinear in these, the
# sea's brightness temperature stays within 0.25 K, grazing views included.
VIEW_ROWS = 32
MIRROR_SAMPLES = 256
# A judgement: the bearing resolution of a cloud's reflection.
CLOUD_BEARINGS = 64
# Measured against `lwir.Sky.reflected` evaluated exactly under the photo
# kloofendal_48d_partly_cloudy: interpolated in these, the clouds' share moves the
# sea under 0.3 K.
CLOUD_SLOPES = 16
CLOUD_VIEWS = 8
CLOUD_MIRRORS = 64

# Elevation (rad, ascending) and the IR sky's radiance there.
type SkyCurve = tuple[np.ndarray, np.ndarray]


class Clouds(NamedTuple):
    """The ir sky by bearing: radiance at the ascending `elevation` (rad), in columns
    laid out as the world's photo, which the world turns by `turn_rad`."""

    elevation: np.ndarray
    radiance: np.ndarray
    turn_rad: float


class Light(NamedTuple):
    """The ir sky round the horizon, or the unit vector an eo sea glints towards. A sky
    makes the sea ir, its cool skin thinner with the sun up; with `clouds`, the sea
    reflects that sky at each bearing."""

    sky: SkyCurve | None = None
    sun: tuple[float, float, float] | None = None
    sun_up: bool = True
    clouds: Clouds | None = None


def _row_sigmas(slope_max: float, rows: int = SLOPE_ROWS) -> np.ndarray:
    """Per-axis unresolved RMS slope up a table's rows, at texel centres.

    Args:
        slope_max: Total RMS slope the rows run up to.
        rows: How many rows the table has."""
    return texels(rows) * slope_max / math.sqrt(2)


def _emissivity_image(t_sea_k: float, slope_max: float) -> bpy.types.Image:
    """`lwir.emissivity_curve` baked against cos(theta) along a row and the unresolved
    RMS slope up the rows, both at texel centres, which is what the shader samples.

    The curve is sampled uniformly in angle; the shader's dot product is uniform in its
    cosine, so it is resampled here rather than corrected in nodes.
    """
    mu = texels()
    rows = []
    for sigma in _row_sigmas(slope_max):
        theta, eps = lwir.emissivity_curve(t_sea_k=t_sea_k, slope_sigma=sigma)
        rows.append(np.interp(mu, np.cos(theta)[::-1], eps[::-1]))
    return curve_image("sea_emissivity", np.array(rows))


def _foam_emissivity_image() -> bpy.types.Image:
    """`lwir.foam_emissivity_excess` against cos(theta) at texel centres."""
    return curve_image(
        "sea_foam_emissivity", lwir.foam_emissivity_excess(np.arccos(texels()))
    )


def _view_rows(rows: int = VIEW_ROWS) -> np.ndarray:
    """|cos| of the view off the wave normal up a tile's rows, at texel centres: even
    in its square root, so the rows crowd toward grazing, where the reflection turns
    fastest."""
    return texels(rows) ** 2


def _mirror_columns() -> np.ndarray:
    """sin(mirror elevation) along a row, at texel centres: down to -1, as a wave
    tilted away steeply mirrors the sea."""
    return texels(MIRROR_SAMPLES) * 2.0 - 1.0


def _reflection_image(
    t_sea_k: float, sky: SkyCurve, slope_max: float
) -> bpy.types.Image:
    """`lwir.Sky.reflected` baked against sin(mirror elevation) along a row
    (`_mirror_columns`), the view off the wave normal up a tile's rows
    (`_view_rows`), and the unresolved RMS slope from tile to tile, all at texel
    centres."""
    elevation, radiance = (np.ascontiguousarray(a, np.float64).tobytes() for a in sky)
    return curve_image(
        "sea_reflection", _reflection_table(t_sea_k, elevation, radiance, slope_max)
    )


# The bake is most of an IR build, and builds in one process often share a sea.
# Arrays do not hash, so the sky comes as float64 bytes.
@lru_cache(maxsize=8)
def _reflection_table(
    t_sea_k: float, elevation: bytes, radiance: bytes, slope_max: float
) -> np.ndarray:
    mirror = np.arcsin(_mirror_columns())
    sky = lwir.Sky(np.frombuffer(elevation), np.frombuffer(radiance))
    return np.array(
        _each(
            lambda sigma, view: sky.reflected(
                mirror, t_sea_k=t_sea_k, slope_sigma=sigma, view_rad=view
            ),
            _tiles(slope_max, SLOPE_ROWS, VIEW_ROWS),
        )
    )


def _tiles(slope_max: float, slopes: int, views: int) -> list[tuple[float, float]]:
    """(unresolved slope, view off the wave normal) per tile, views nested in slopes."""
    return [
        (sigma, view)
        for sigma in _row_sigmas(slope_max, slopes)
        for view in np.arcsin(_view_rows(views))
    ]


def _each[T](
    bake: Callable[[float, float], T], tiles: Iterable[tuple[float, float]]
) -> list[T]:
    """`bake` of each tile, in order; numpy releases the GIL, so threads share it."""
    # CPU-bound: a thread past the cores only holds another tile's arrays.
    with ThreadPoolExecutor(os.process_cpu_count()) as pool:
        return list(pool.map(lambda tile: bake(*tile), tiles))


def _cloud_image(t_sea_k: float, clouds: Clouds, slope_max: float) -> bpy.types.Image:
    """The sky reflected at each bearing over the reflection of the sky averaged round
    the horizon, which the shader multiplies into the reflection. Columns span the
    photo's bearings. Rows run up sin(mirror elevation), -1 to 1, evenly in its signed
    square root; tiles of view off the wave normal, as `_view_rows`, nest in tiles of
    unresolved slope."""
    root = texels(CLOUD_MIRRORS) * 2.0 - 1.0
    mirror = np.arcsin(np.sign(root) * root**2)
    edges = np.linspace(0, clouds.radiance.shape[1], CLOUD_BEARINGS + 1).astype(int)
    widths = np.diff(edges)
    bearings = np.add.reduceat(clouds.radiance, edges[:-1], axis=1) / widths
    round_ = clouds.radiance.mean(axis=1)
    skies = [lwir.Sky(clouds.elevation, sky) for sky in (bearings, round_)]

    def ratio(sigma: float, view: float) -> np.ndarray:
        seen, mean = (
            sky.reflected(mirror, t_sea_k=t_sea_k, slope_sigma=sigma, view_rad=view)
            for sky in skies
        )
        return seen / mean[:, None]

    rows = _each(ratio, _tiles(slope_max, CLOUD_SLOPES, CLOUD_VIEWS))
    return curve_image("sea_clouds", np.concatenate(rows))


def _incidence(
    tree: bpy.types.NodeTree, normal: bpy.types.NodeSocket
) -> bpy.types.NodeSocket:
    """|cos(theta)| between the wave normal and the viewing ray: against the wave
    normal, not the plane's, or a flat sea's emissivity lands on water visibly not
    flat."""
    geometry = tree.nodes.new("ShaderNodeNewGeometry")
    facing = vector(tree, "DOT_PRODUCT", geometry.outputs["Incoming"], normal)
    return scalar(tree, "ABSOLUTE", facing)


def _thermal(
    tree: bpy.types.NodeTree,
    sea: Sea,
    normal: bpy.types.NodeSocket,
    unresolved: tuple[bpy.types.NodeSocket, bpy.types.NodeSocket],
    slope_max: float,
    light: Light,
    skin: bpy.types.NodeSocket | float,
    foam: bpy.types.NodeSocket,
) -> tuple[bpy.types.NodeSocket, bpy.types.NodeSocket]:
    """The sea's emission plus the 1 - eps(theta) of sky it reflects, and eps(theta).

    Complements, so the two very nearly cancel and the sea holds close to ambient at
    every angle. The reflection is a lookup in the wave normal's mirror direction, not
    a BSDF: Cycles sampling a lobe across a sky cold overhead leaves the sea grainy.
    Only the slope along the view enters: the slope across turns the reflection
    sideways, which lowers it only at second order. Derivation: a tilt d across the
    view scales the mirror's sin(elevation) by (1 - d^2) / (1 + d^2).

    The unresolved facets lean about the wave normal, so the reflection is tabulated
    against the view off the wave normal as well as the mirror off the horizon.

    Clouds reflect at the mirror's bearing, blurred in elevation only: a facet tilted
    across the view by d turns the mirror's bearing by about 2 d sin(depression), which
    vanishes at the grazing views the sea is seen at.

    The sky alone: `_traced` adds the hulls.

    Args:
        tree: The sea material's node tree.
        sea: Its bulk temperature sets the tables' optical constants.
        normal: The wave normal the pixel draws, in world space.
        unresolved: Variances of the slope finer than the pixel draws, along and
            across the view.
        slope_max: Total RMS slope the tables span.
        light: The sky, and any clouds, the sea reflects.
        skin: Band radiance the sea emits. The tables keep the bulk's optical
            constants, which a skin a few kelvin off barely moves.
        foam: Share of the pixel under foam; raises eps by that share of foam's
            excess."""
    sky = light.sky
    assert sky is not None
    geometry = tree.nodes.new("ShaderNodeNewGeometry")
    # Incoming points at the camera; REFLECT takes the ray.
    ray = vector(tree, "SCALE", geometry.outputs["Incoming"], -1.0)
    reflected = vector(tree, "REFLECT", ray, normal)
    _, _, mirror_z = separate(tree, reflected)
    along = scalar(
        tree,
        "DIVIDE",
        scalar(tree, "SQRT", unresolved[0]),
        slope_max,
        name="sea_slope_along",
    )
    incidence = _incidence(tree, normal)
    sky_seen = _tiled(
        tree,
        _reflection_image(sea.t_sea_k, sky, slope_max),
        scalar(tree, "MULTIPLY_ADD", mirror_z, 0.5, 0.5),
        scalar(tree, "SQRT", incidence),
        VIEW_ROWS,
        ((along, SLOPE_ROWS),),
    )
    if light.clouds is not None:
        tiles = (
            (along, CLOUD_SLOPES),
            (scalar(tree, "SQRT", incidence), CLOUD_VIEWS),
        )
        ratio = _cloud_ratio(tree, sea, light.clouds, slope_max, reflected, tiles)
        sky_seen = scalar(tree, "MULTIPLY", sky_seen, ratio)
    sky_seen.node.name = "sea_reflection"
    mean = (Num(tree, unresolved[0]) + unresolved[1]) * 0.5
    fraction = mean.op("SQRT") / slope_max
    bare = lookup(
        tree, _emissivity_image(sea.t_sea_k, slope_max), incidence, fraction.socket
    )
    excess = lookup(tree, _foam_emissivity_image(), incidence)
    emissivity = scalar(
        tree,
        "MINIMUM",
        scalar(tree, "MULTIPLY_ADD", foam, excess, bare),
        1.0,
        name="sea_emissivity",
    )
    emitted = scalar(tree, "SUBTRACT", skin, sky_seen, name="sea_skin")
    radiance = scalar(
        tree, "MULTIPLY_ADD", emissivity, emitted, sky_seen, name="sea_radiance"
    )
    emission = node(tree, "ShaderNodeEmission", {"Color": radiance})
    return emission.outputs["Emission"], emissivity


def _traced(
    tree: bpy.types.NodeTree,
    sky: bpy.types.NodeSocket,
    emissivity: bpy.types.NodeSocket,
    normal: bpy.types.NodeSocket,
    pixel: _Pixel,
    unresolved: tuple[bpy.types.NodeSocket, bpy.types.NodeSocket],
) -> bpy.types.NodeSocket:
    """The sea reflecting the sky, plus what a lobe of its unresolved slopes meets
    besides: hulls, and waves that hide the sky.

    Args:
        tree: The sea material's node tree.
        sky: The sea's emission and the sky it reflects, as `_thermal` draws them.
        emissivity: The pixel's, not each facet's own; the lobe is weighted 1 minus it.
        normal: The wave normal the lobe leans about.
        pixel: The footprint whose axes orient the lobe.
        unresolved: Variances of the slope finer than the pixel draws, along and
            across the view, which shape the lobe."""
    wide, narrow, tangent = _widest(tree, pixel, unresolved)
    roughness, anisotropy = _lobe(tree, wide, _stretchable(tree, wide, narrow))
    lobe = node(
        tree,
        "ShaderNodeBsdfGlossy",
        {
            "Normal": normal,
            "Tangent": tangent,
            "Roughness": roughness,
            "Anisotropy": anisotropy,
            "Color": scalar(tree, "SUBTRACT", 1.0, emissivity),
        },
        name="sea_traced",
        distribution="BECKMANN",
    )
    both = tree.nodes.new("ShaderNodeAddShader")
    tree.links.new(sky, both.inputs[0])
    tree.links.new(lobe.outputs["BSDF"], both.inputs[1])
    return both.outputs["Shader"]


def _cloud_ratio(
    tree: bpy.types.NodeTree,
    sea: Sea,
    clouds: Clouds,
    slope_max: float,
    reflected: bpy.types.NodeSocket,
    tiles: tuple[tuple[bpy.types.NodeSocket, int], ...],
) -> bpy.types.NodeSocket:
    """`_cloud_image` at the reflection's bearing and elevation, its bearing turned as
    the world turns its photo.

    Args:
        tree: The sea material's node tree.
        sea: Its bulk temperature sets the table's reflectance.
        clouds: The photo's sky radiance by elevation and bearing, and its turn.
        slope_max: Total RMS slope the table spans.
        reflected: The view ray mirrored in the wave normal, in world space.
        tiles: (coordinate, count) per tile axis, as `_tiled` takes them."""
    turn = node(
        tree,
        "ShaderNodeMapping",
        {"Rotation": (0.0, 0.0, clouds.turn_rad), "Vector": reflected},
    )
    x, y, z = separate(tree, turn.outputs["Vector"])
    # Blender's equirectangular u: 0.5 - atan2(y, x) / 2 pi.
    bearing = scalar(
        tree,
        "MULTIPLY_ADD",
        scalar(tree, "ARCTAN2", y, x),
        -1 / (2 * math.pi),
        0.5,
        name="sea_cloud_bearing",
    )
    up = Num(tree, z)
    root = up.op("SIGN") * up.op("ABSOLUTE").op("SQRT")
    elevation = root.madd(0.5, 0.5).socket
    ratio = _tiled(
        tree,
        _cloud_image(sea.t_sea_k, clouds, slope_max),
        bearing,
        elevation,
        CLOUD_MIRRORS,
        tiles,
        extension="REPEAT",
    )
    ratio.node.name = "sea_cloud_ratio"
    return ratio


def _skin_radiance(
    tree: bpy.types.NodeTree, sea: Sea, sun_up: bool, wake: _Wakes | None
) -> bpy.types.NodeSocket | float:
    """The band radiance the sea emits: its skin's, or its bulk's where a wake's
    turbulence mixes the skin away. A wake reads warm under a cool skin and cold where
    a warm layer lifts the skin above the bulk (Nylund et al. 2021, Ocean Sci. 17 1285).

    Limitation: the wake mixes down to the bulk only; colder water it raises from below
    the bulk is not modelled."""
    skin = lwir.band_radiance(sea.t_skin_k(sun_up))
    if wake is None:
        return skin
    bulk = lwir.band_radiance(sea.t_sea_k)
    return scalar(
        tree, "MULTIPLY_ADD", wake.band, bulk - skin, skin, name="sea_skin_radiance"
    )


def _tiled(
    tree: bpy.types.NodeTree,
    image: bpy.types.Image,
    x: bpy.types.NodeSocket,
    row: bpy.types.NodeSocket,
    rows: int,
    tiles: tuple[tuple[bpy.types.NodeSocket, int], ...],
    extension: str = "EXTEND",
) -> bpy.types.NodeSocket:
    """A lookup in a tiled table. The image interpolates within a tile and lookups mix
    across tiles, so no tile bleeds into the next.

    Args:
        tree: The sea material's node tree.
        image: The table, its tiles stacked up the rows.
        x: Position along a row, in [0, 1].
        row: Position up a tile, in [0, 1].
        rows: Rows per tile.
        tiles: (coordinate, count) per axis, each picking a tile along it, coordinate
            in [0, 1]; the last axis varies fastest.
        extension: Blender's image extension along a row: EXTEND holds the edge
            texel, REPEAT wraps, as a bearing does round the horizon."""
    # Held to texel centres, as EXTEND holds the image's own edges.
    texel = (Num(tree, row) * rows).op("MAXIMUM", 0.5).op("MINIMUM", rows - 0.5)
    height = rows * math.prod(count for _, count in tiles)
    axes = []
    for coord, count in tiles:
        at = (
            Num(tree, coord)
            .madd(count, -0.5)
            .op("MAXIMUM", 0.0)
            .op("MINIMUM", count - 1.0)
        )
        lower = at.op("FLOOR")
        upper = (lower + 1.0).op("MINIMUM", count - 1.0)
        axes.append((count, lower.socket, upper.socket, (at - lower).socket))

    def mixed(axis: int, tile: bpy.types.NodeSocket | float) -> Num:
        if axis == len(axes):
            y = scalar(tree, "MULTIPLY_ADD", tile, rows, texel.socket)
            seen = lookup(tree, image, x, scalar(tree, "DIVIDE", y, height))
            seen.node.extension = extension
            return Num(tree, seen)
        count, lower, upper, weight = axes[axis]
        lo, hi = (
            mixed(axis + 1, scalar(tree, "MULTIPLY_ADD", tile, count, t))
            for t in (lower, upper)
        )
        return Num(tree, weight).madd(hi - lo, lo)

    return mixed(0, 0.0).socket
