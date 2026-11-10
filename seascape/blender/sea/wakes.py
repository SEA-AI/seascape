"""A hull's wake: its turbulent band, stern foam and Kelvin arms."""

import math
from typing import NamedTuple

import bpy

from seascape.blender.blend import (
    Num,
    combine,
    lookup,
    node,
    scalar,
    separate,
    vector,
)
from seascape.blender.sea.foam import FOAM_STREAK, FOAM_TILE_M
from seascape.physics.earth import GRAVITY_MS2
from seascape.physics.wakes import (
    BUBBLE_EFOLD_S,
    BUBBLE_GAIN,
    FOAM_EFOLD_S,
    KELVIN_HALF_ANGLE_RAD,
    MAX_STEEPNESS,
    NARROWING_FROUDE,
    SEEN_SLOPE_VARIANCE,
    Wake,
)


class _Wakes(NamedTuple):
    band: bpy.types.NodeSocket  # 1 in the turbulent wake
    foam: bpy.types.NodeSocket  # fresh foam's cover
    bubbles: bpy.types.NodeSocket  # water-body reflectance added, a fraction of its own
    tilt: bpy.types.NodeSocket | None  # the Kelvin arms' -grad(height)


class _Path(NamedTuple):
    """Where a shading point lies against a hull's path."""

    along: bpy.types.NodeSocket  # behind the hull
    across: bpy.types.NodeSocket  # to its starboard
    d_along: bpy.types.NodeSocket  # gradients, (east, north)
    d_across: bpy.types.NodeSocket
    fixed: bpy.types.NodeSocket  # along and across, fixed in the water


def _orbit_path(
    tree: bpy.types.NodeTree,
    wake: Wake,
    time_s: bpy.types.NodeSocket,
    east: bpy.types.NodeSocket,
    north: bpy.types.NodeSocket,
) -> _Path:
    assert wake.orbit_m is not None
    bearing = Num(tree, east).op("ARCTAN2", north)
    hull = Num(tree, time_s).madd(2 * math.pi / wake.lap_s, wake.start_bearing_rad)
    # Clockwise: behind a hull is anticlockwise of it, as far as the hull after it.
    angle = (hull - bearing).op("FLOORED_MODULO", 2 * math.pi / wake.count)
    radius = vector(tree, "LENGTH", combine(tree, east, north))
    cos_b, sin_b = bearing.op("COSINE"), bearing.op("SINE")
    # Round the circle in whole foam tiles, so the bearing's wrap is no seam.
    round_m = FOAM_STREAK * FOAM_TILE_M
    round_m *= max(1, round(2 * math.pi * wake.orbit_m / round_m))
    return _Path(
        along=(angle * wake.orbit_m).socket,
        # Clockwise, starboard is in towards the centre.
        across=(wake.orbit_m - Num(tree, radius)).socket,
        d_along=combine(tree, cos_b * -1.0, sin_b),
        d_across=combine(tree, sin_b * -1.0, cos_b * -1.0),
        fixed=combine(tree, bearing * (-round_m / (2 * math.pi)), radius),
    )


def _line_path(
    tree: bpy.types.NodeTree,
    wake: Wake,
    time_s: bpy.types.NodeSocket,
    east: bpy.types.NodeSocket,
    north: bpy.types.NodeSocket,
) -> _Path:
    sin_h, cos_h = math.sin(wake.heading_rad), math.cos(wake.heading_rad)
    e0, n0 = wake.start_m
    hull_e = Num(tree, time_s).madd(wake.speed_mps * sin_h, e0)
    hull_n = Num(tree, time_s).madd(wake.speed_mps * cos_h, n0)
    east, north = Num(tree, east), Num(tree, north)
    de = east - hull_e
    dn = north - hull_n

    def ahead(e: Num, n: Num) -> Num:
        return e.madd(sin_h, n * cos_h)

    def starboard(e: Num, n: Num) -> Num:
        # Starboard of the heading is (cos, -sin).
        return e.madd(cos_h, n * -sin_h)

    return _Path(
        along=(ahead(de, dn) * -1.0).socket,
        across=starboard(de, dn).socket,
        d_along=combine(tree, -sin_h, -cos_h),
        d_across=combine(tree, cos_h, -sin_h),
        fixed=combine(tree, ahead(east, north), starboard(east, north)),
    )


def _path(tree: bpy.types.NodeTree, wake: Wake, where: bpy.types.NodeSocket) -> _Path:
    east, north, time_s = separate(tree, where)
    east.node.name = "wake_position"
    if wake.orbit_m is not None:
        return _orbit_path(tree, wake, time_s, east, north)
    return _line_path(tree, wake, time_s, east, north)


def _gaussian(x: Num, half: Num | float) -> Num:
    """exp(-(x / half)^2)."""
    ratio = x / half
    return (ratio * ratio * -1.0).op("EXPONENT")


def _wakes(
    tree: bpy.types.NodeTree,
    wakes: tuple[Wake, ...],
    where: bpy.types.NodeSocket,
    footprint_m: bpy.types.NodeSocket,
    noise: bpy.types.Image,
    ppf: bpy.types.Image,
    sea_slope_variance: float,
) -> _Wakes:
    """Every hull's turbulent band, stern foam and Kelvin arms: the strongest band and
    foam, and the arms summed.

    Limitation: the arms are Kelvin's straight-path pattern laid along the path, which a
    tight turn bends; sum the arms from past poses if a turn reads wrong.

    Args:
        tree: The sea material's shader tree.
        wakes: One per hull under way; an orbit's hulls share one.
        where: (east, north, time) the wakes are read at.
        footprint_m: The pixel's longer side, which fades the arms it cannot resolve.
        noise: Von Karman noise, zero mean and unit variance, that patches the foam.
        ppf: The unit normal's 1 - p quantile, tabulated in p: a unit Gaussian tops
            ppf(p) over p of its area.
        sea_slope_variance: The sea's total, both axes summed; a hull's arms are
            drawn only where they stand out from it.
    """
    # (band, foam, bubbles) maxed over wakes; three accumulators would each hold every
    # wake on Cycles' SVM stack until the last closure compiles.
    strongest = None
    tilt = None
    for wake in wakes:
        path = _path(tree, wake, where)
        along = Num(tree, path.along)
        behind = along.op("GREATER_THAN", 0.0)
        # Ahead of the hull nothing has aged: unclamped, the exponentials overflow
        # there, and the power below takes a negative base.
        astern = along.op("MAXIMUM", 0.0)
        side = Num(tree, path.across).op("ABSOLUTE")
        # A judgement: a beam wide at the stern, widening as distance^(1/5).
        width = astern.madd(1 / wake.length_m, 1.0).op("POWER", 0.2) * wake.beam_m
        core = behind * _gaussian(side, width * 0.5)
        age = astern / wake.speed_mps
        fresh = (age * (-1 / FOAM_EFOLD_S)).op("EXPONENT")
        stern = behind * _gaussian(side, wake.beam_m / 2)
        patchy = _patchy(tree, stern * fresh, path.fixed, noise, ppf)
        rising = (age * (-1 / BUBBLE_EFOLD_S)).op("EXPONENT")
        bubbles = core * (rising * BUBBLE_GAIN)
        these = combine(tree, core.socket, patchy.socket, bubbles.socket)
        strongest = (
            these if strongest is None else vector(tree, "MAXIMUM", strongest, these)
        )
        if wake.steepest_arm**2 / 2 < SEEN_SLOPE_VARIANCE * sea_slope_variance:
            continue
        lasting = behind
        if wake.orbit_m is not None:
            # A judgement: faded over the radius, past which the straight-path pattern
            # no longer follows the turn.
            fade = (along * (-1 / wake.orbit_m)).op("EXPONENT")
            lasting = behind * fade
        # Above the narrowing only the divergent wave, round its peak (Darmon et al.).
        roots = (1.0, -1.0) if wake.froude <= NARROWING_FROUDE else (1.0,)
        for root in roots:
            arm = vector(
                tree,
                "SCALE",
                _kelvin(tree, wake, root, path, footprint_m),
                lasting,
            )
            tilt = arm if tilt is None else vector(tree, "ADD", tilt, arm)
    assert strongest is not None
    return _Wakes(*separate(tree, strongest), tilt)


def _kelvin(
    tree: bpy.types.NodeTree,
    wake: Wake,
    root: float,
    path: _Path,
    footprint_m: bpy.types.NodeSocket,
) -> bpy.types.NodeSocket:
    """One of Kelvin's two waves as -grad(height), (east, north).

    Stationary phase: at tan(psi) = side / along, the wave's direction theta solves
    tan(psi) (1 + 2 t^2) = t, t = tan(theta), and its phase is
    g / U^2 (along cos(theta) + side sin(theta)) / cos(theta)^2 (DLMF 36.13).

    Args:
        tree: The sea material's shader tree.
        wake: The hull's speed U, which sets the wavenumber scale g / U^2, and its size.
        root: The stationary-phase root: 1 the divergent wave, -1 the transverse.
        path: The point's place along and across the hull's path.
        footprint_m: The pixel's longer side f; a wave of wavenumber k keeps
            sin(k f / 2) / (k f / 2) of its slope, a box average's response.
    """
    along = Num(tree, path.along)
    side = Num(tree, path.across).op("ABSOLUTE")
    edge = math.tan(KELVIN_HALF_ANGLE_RAD)
    # Numerics: the floors keep the divisions by along and by tan_psi finite.
    tan_psi = side / along.op("MAXIMUM", 1e-3)
    inside = tan_psi.op("LESS_THAN", edge)
    tan_psi = tan_psi.op("MINIMUM", edge).op("MAXIMUM", 1e-4)
    disc = (tan_psi * tan_psi).madd(-8.0, 1.0).op("MAXIMUM", 0.0).op("SQRT")
    t = disc.madd(root, 1.0) / (tan_psi * 4.0)
    cos_sq = 1.0 / t.madd(t, 1.0)
    cos_t = cos_sq.op("SQRT")
    sin_t = t * cos_t
    k = GRAVITY_MS2 / wake.speed_mps**2 / cos_sq
    phase = k * along.madd(cos_t, side * sin_t)
    envelope: Num | float = 1.0
    if wake.froude > NARROWING_FROUDE:
        peak = wake.peak_angle_rad
        # A judgement: the peak half as wide as its angle.
        envelope = _gaussian(tan_psi.op("ARCTANGENT") - peak, peak / 2)
    # Kriebel & Seelig's height, half of it the amplitude.
    amplitude = (side / wake.length_m).op("MAXIMUM", 1e-2).op("POWER", -1 / 3) * (
        wake.height_scale_m / 2
    )
    steepest = (amplitude * k).op("MINIMUM", math.pi * MAX_STEEPNESS)
    # `waves.visibility`, as the sea's own waves fade.
    y = Num(tree, footprint_m) * k * 0.5
    resolved = y.op("MINIMUM", math.pi).op("SINE") / y
    slope = steepest
    for factor in (phase.op("SINE"), envelope, inside, resolved):
        slope = slope * factor
    sign = Num(tree, path.across).op("SIGN")
    direction = vector(
        tree,
        "ADD",
        vector(tree, "SCALE", path.d_along, cos_t),
        vector(tree, "SCALE", path.d_across, sin_t * sign),
    )
    return vector(tree, "SCALE", direction, slope)


def _patchy(
    tree: bpy.types.NodeTree,
    cover: Num,
    fixed: bpy.types.NodeSocket,
    noise: bpy.types.Image,
    ppf: bpy.types.Image,
) -> Num:
    """A cover as white where a unit Gaussian tops its own fraction of that cover:
    patches whose mean is the cover, laid along the path.

    Args:
        tree: The sea material's shader tree.
        cover: The share of the sea the foam covers.
        fixed: The point's place in the path's frame, which the patches are laid on.
        noise: Von Karman noise, zero mean and unit variance.
        ppf: The unit normal's 1 - p quantile, tabulated in p.
    """
    stretch = vector(
        tree,
        "MULTIPLY",
        fixed,
        (1 / (FOAM_STREAK * FOAM_TILE_M), 1 / FOAM_TILE_M, 0.0),
    )
    texture = node(
        tree,
        "ShaderNodeTexImage",
        {"Vector": stretch},
        image=noise,
        extension="REPEAT",
    )
    threshold = lookup(tree, ppf, cover)
    # A judgement: edges half the tile's sigma wide.
    edge = (Num(tree, texture.outputs["Color"]) - threshold).madd(2.0, 0.5)
    edge.socket.node.use_clamp = True
    return edge * cover.op("GREATER_THAN", 1e-3)


def _smoothed(
    tree: bpy.types.NodeTree,
    slick: bpy.types.NodeSocket | None,
    wake: _Wakes | None,
) -> bpy.types.NodeSocket | None:
    """Where the unresolved roughness is a slick's: under a slick, and in a turbulent
    wake, whose short waves stay damped (Milgram et al. 1993).
    """
    if wake is None:
        return slick
    if slick is None:
        return wake.band
    return scalar(tree, "MAXIMUM", slick, wake.band)
