"""A hull's wake: its turbulent band, stern foam and Kelvin arms."""

import math
from statistics import NormalDist
from typing import NamedTuple

import bpy
import numpy as np

from seascape.blend import (
    CURVE_SAMPLES,
    _math,
    _vector,
    curve_image,
    lookup,
)
from seascape.blender.sea.foam import FOAM_STREAK, FOAM_TILE_M
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
from seascape.physics.waves import (
    GRAVITY_MS2,
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


def _pair(
    tree: bpy.types.NodeTree,
    east: bpy.types.NodeSocket | float,
    north: bpy.types.NodeSocket | float,
) -> bpy.types.NodeSocket:
    node = tree.nodes.new("ShaderNodeCombineXYZ")
    for socket, value in zip(("X", "Y"), (east, north), strict=True):
        if isinstance(value, bpy.types.NodeSocket):
            tree.links.new(value, node.inputs[socket])
        else:
            node.inputs[socket].default_value = value
    return node.outputs["Vector"]


def _orbit_path(
    tree: bpy.types.NodeTree,
    wake: Wake,
    time_s: bpy.types.NodeSocket,
    east: bpy.types.NodeSocket,
    north: bpy.types.NodeSocket,
) -> _Path:
    assert wake.orbit_m is not None
    bearing = _math(tree, "ARCTAN2", east, north)
    hull = _math(
        tree, "MULTIPLY_ADD", time_s, 2 * math.pi / wake.lap_s, wake.start_bearing_rad
    )
    # Clockwise: behind a hull is anticlockwise of it, as far as the hull after it.
    angle = _math(
        tree,
        "FLOORED_MODULO",
        _math(tree, "SUBTRACT", hull, bearing),
        2 * math.pi / wake.count,
    )
    radius = _vector(tree, "LENGTH", _pair(tree, east, north))
    cos_b, sin_b = _math(tree, "COSINE", bearing), _math(tree, "SINE", bearing)
    # Round the circle in whole foam tiles, so the bearing's wrap is no seam.
    round_m = FOAM_STREAK * FOAM_TILE_M
    round_m *= max(1, round(2 * math.pi * wake.orbit_m / round_m))
    return _Path(
        along=_math(tree, "MULTIPLY", angle, wake.orbit_m),
        # Clockwise, starboard is in towards the centre.
        across=_math(tree, "SUBTRACT", wake.orbit_m, radius),
        d_along=_pair(tree, _math(tree, "MULTIPLY", cos_b, -1.0), sin_b),
        d_across=_pair(
            tree,
            _math(tree, "MULTIPLY", sin_b, -1.0),
            _math(tree, "MULTIPLY", cos_b, -1.0),
        ),
        fixed=_pair(
            tree, _math(tree, "MULTIPLY", bearing, -round_m / (2 * math.pi)), radius
        ),
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
    hull_e = _math(tree, "MULTIPLY_ADD", time_s, wake.speed_mps * sin_h, e0)
    hull_n = _math(tree, "MULTIPLY_ADD", time_s, wake.speed_mps * cos_h, n0)
    de = _math(tree, "SUBTRACT", east, hull_e)
    dn = _math(tree, "SUBTRACT", north, hull_n)

    def ahead(e: bpy.types.NodeSocket, n: bpy.types.NodeSocket) -> bpy.types.NodeSocket:
        return _math(tree, "MULTIPLY_ADD", e, sin_h, _math(tree, "MULTIPLY", n, cos_h))

    def starboard(
        e: bpy.types.NodeSocket, n: bpy.types.NodeSocket
    ) -> bpy.types.NodeSocket:
        # Starboard of the heading is (cos, -sin).
        return _math(tree, "MULTIPLY_ADD", e, cos_h, _math(tree, "MULTIPLY", n, -sin_h))

    return _Path(
        along=_math(tree, "MULTIPLY", ahead(de, dn), -1.0),
        across=starboard(de, dn),
        d_along=_pair(tree, -sin_h, -cos_h),
        d_across=_pair(tree, cos_h, -sin_h),
        fixed=_pair(tree, ahead(east, north), starboard(east, north)),
    )


def _path(tree: bpy.types.NodeTree, wake: Wake, where: bpy.types.NodeSocket) -> _Path:
    position = tree.nodes.new("ShaderNodeSeparateXYZ")
    position.name = "wake_position"
    tree.links.new(where, position.inputs["Vector"])
    east, north, time_s = (position.outputs[c] for c in ("X", "Y", "Z"))
    if wake.orbit_m is not None:
        return _orbit_path(tree, wake, time_s, east, north)
    return _line_path(tree, wake, time_s, east, north)


def _gaussian(
    tree: bpy.types.NodeTree,
    x: bpy.types.NodeSocket,
    half: bpy.types.NodeSocket | float,
) -> bpy.types.NodeSocket:
    """exp(-(x / half)^2)."""
    ratio = _math(tree, "DIVIDE", x, half)
    return _math(
        tree,
        "EXPONENT",
        _math(tree, "MULTIPLY", _math(tree, "MULTIPLY", ratio, ratio), -1.0),
    )


def _wakes(
    tree: bpy.types.NodeTree,
    wakes: tuple[Wake, ...],
    where: bpy.types.NodeSocket,
    footprint_m: bpy.types.NodeSocket,
    noise: bpy.types.Image,
    sea_slope_variance: float,
) -> _Wakes:
    """Every hull's turbulent band, stern foam and Kelvin arms: the strongest band and
    foam, and the arms summed, read at `where`, (east, north, time).

    ponytail: the arms are Kelvin's straight-path pattern laid along the path, which a
    tight turn bends; sum the arms from past poses if a turn reads wrong.
    """
    levels = (np.arange(CURVE_SAMPLES) + 0.5) / CURVE_SAMPLES
    ppf = curve_image("normal_ppf", np.vectorize(NormalDist().inv_cdf)(1 - levels))
    band: bpy.types.NodeSocket | float = 0.0
    foam: bpy.types.NodeSocket | float = 0.0
    bubbles: bpy.types.NodeSocket | float = 0.0
    tilt = None
    for wake in wakes:
        path = _path(tree, wake, where)
        behind = _math(tree, "GREATER_THAN", path.along, 0.0)
        # Ahead of the hull nothing has aged: unclamped, the exponentials overflow
        # there, and the power below takes a negative base.
        astern = _math(tree, "MAXIMUM", path.along, 0.0)
        side = _math(tree, "ABSOLUTE", path.across)
        # A judgement: a beam wide at the stern, widening as distance^(1/5).
        width = _math(
            tree,
            "MULTIPLY",
            _math(
                tree,
                "POWER",
                _math(tree, "MULTIPLY_ADD", astern, 1 / wake.length_m, 1.0),
                0.2,
            ),
            wake.beam_m,
        )
        core = _math(
            tree,
            "MULTIPLY",
            behind,
            _gaussian(tree, side, _math(tree, "MULTIPLY", width, 0.5)),
        )
        band = _math(tree, "MAXIMUM", band, core)
        age = _math(tree, "DIVIDE", astern, wake.speed_mps)
        fresh = _math(tree, "EXPONENT", _math(tree, "MULTIPLY", age, -1 / FOAM_EFOLD_S))
        stern = _math(tree, "MULTIPLY", behind, _gaussian(tree, side, wake.beam_m / 2))
        cover = _math(tree, "MULTIPLY", stern, fresh)
        foam = _math(
            tree, "MAXIMUM", foam, _patchy(tree, cover, path.fixed, noise, ppf)
        )
        rising = _math(
            tree, "EXPONENT", _math(tree, "MULTIPLY", age, -1 / BUBBLE_EFOLD_S)
        )
        bubbles = _math(
            tree,
            "MAXIMUM",
            bubbles,
            _math(tree, "MULTIPLY", core, _math(tree, "MULTIPLY", rising, BUBBLE_GAIN)),
        )
        if wake.steepest_arm**2 / 2 < SEEN_SLOPE_VARIANCE * sea_slope_variance:
            continue
        lasting: bpy.types.NodeSocket = behind
        if wake.orbit_m is not None:
            # A judgement: faded over the radius, past which the straight-path pattern
            # no longer follows the turn.
            fade = _math(
                tree, "EXPONENT", _math(tree, "MULTIPLY", path.along, -1 / wake.orbit_m)
            )
            lasting = _math(tree, "MULTIPLY", behind, fade)
        # Above the narrowing only the divergent wave, round its peak (Darmon et al.).
        roots = (1.0, -1.0) if wake.froude <= NARROWING_FROUDE else (1.0,)
        for root in roots:
            arm = _vector(
                tree, "SCALE", _kelvin(tree, wake, root, path, footprint_m), lasting
            )
            tilt = arm if tilt is None else _vector(tree, "ADD", tilt, arm)
    return _Wakes(band, foam, bubbles, tilt)


def _kelvin(
    tree: bpy.types.NodeTree,
    wake: Wake,
    root: float,
    path: _Path,
    footprint_m: bpy.types.NodeSocket,
) -> bpy.types.NodeSocket:
    """One of Kelvin's two waves, divergent at `root` 1, transverse at -1, as
    -grad(height), (east, north).

    Stationary phase: at tan(psi) = side / along, the wave's direction theta solves
    tan(psi) (1 + 2 t^2) = t, t = tan(theta), and its phase is
    g / U^2 (along cos(theta) + side sin(theta)) / cos(theta)^2 (DLMF 36.13).
    """
    side = _math(tree, "ABSOLUTE", path.across)
    edge = math.tan(KELVIN_HALF_ANGLE_RAD)
    tan_psi = _math(tree, "DIVIDE", side, _math(tree, "MAXIMUM", path.along, 1e-3))
    inside = _math(tree, "LESS_THAN", tan_psi, edge)
    tan_psi = _math(tree, "MAXIMUM", _math(tree, "MINIMUM", tan_psi, edge), 1e-4)
    disc = _math(
        tree,
        "SQRT",
        _math(
            tree,
            "MAXIMUM",
            _math(
                tree,
                "MULTIPLY_ADD",
                _math(tree, "MULTIPLY", tan_psi, tan_psi),
                -8.0,
                1.0,
            ),
            0.0,
        ),
    )
    t = _math(
        tree,
        "DIVIDE",
        _math(tree, "MULTIPLY_ADD", disc, root, 1.0),
        _math(tree, "MULTIPLY", tan_psi, 4.0),
    )
    cos_sq = _math(tree, "DIVIDE", 1.0, _math(tree, "MULTIPLY_ADD", t, t, 1.0))
    cos_t = _math(tree, "SQRT", cos_sq)
    sin_t = _math(tree, "MULTIPLY", t, cos_t)
    k = _math(tree, "DIVIDE", GRAVITY_MS2 / wake.speed_mps**2, cos_sq)
    phase = _math(
        tree,
        "MULTIPLY",
        k,
        _math(
            tree,
            "MULTIPLY_ADD",
            path.along,
            cos_t,
            _math(tree, "MULTIPLY", side, sin_t),
        ),
    )
    envelope: bpy.types.NodeSocket | float = 1.0
    if wake.froude > NARROWING_FROUDE:
        peak = wake.peak_angle_rad
        # A judgement: the peak half as wide as its angle.
        off = _math(
            tree,
            "SUBTRACT",
            _math(tree, "ARCTANGENT", tan_psi),
            peak,
        )
        envelope = _gaussian(tree, off, peak / 2)
    # Kriebel & Seelig's height, half of it the amplitude.
    amplitude = _math(
        tree,
        "MULTIPLY",
        _math(
            tree,
            "POWER",
            _math(tree, "MAXIMUM", _math(tree, "DIVIDE", side, wake.length_m), 1e-2),
            -1 / 3,
        ),
        wake.height_scale_m / 2,
    )
    steepest = _math(
        tree,
        "MINIMUM",
        _math(tree, "MULTIPLY", amplitude, k),
        math.pi * MAX_STEEPNESS,
    )
    # `waves.visibility`, as the sea's own waves fade.
    y = _math(tree, "MULTIPLY", _math(tree, "MULTIPLY", footprint_m, k), 0.5)
    resolved = _math(
        tree, "DIVIDE", _math(tree, "SINE", _math(tree, "MINIMUM", y, math.pi)), y
    )
    slope = steepest
    for factor in (
        _math(tree, "SINE", phase),
        envelope,
        inside,
        resolved,
    ):
        slope = _math(tree, "MULTIPLY", slope, factor)
    sign = _math(tree, "SIGN", path.across)
    direction = _vector(
        tree,
        "ADD",
        _vector(tree, "SCALE", path.d_along, cos_t),
        _vector(tree, "SCALE", path.d_across, _math(tree, "MULTIPLY", sin_t, sign)),
    )
    return _vector(tree, "SCALE", direction, slope)


def _patchy(
    tree: bpy.types.NodeTree,
    cover: bpy.types.NodeSocket,
    fixed: bpy.types.NodeSocket,
    noise: bpy.types.Image,
    ppf: bpy.types.Image,
) -> bpy.types.NodeSocket:
    """`cover` as white where the Gaussian `noise` tops its own `cover` fraction,
    `ppf` its quantiles: patches whose mean is the cover, laid along the path at
    `fixed`."""
    stretch = tree.nodes.new("ShaderNodeVectorMath")
    stretch.operation = "MULTIPLY"
    tree.links.new(fixed, stretch.inputs[0])
    stretch.inputs[1].default_value = (
        1 / (FOAM_STREAK * FOAM_TILE_M),
        1 / FOAM_TILE_M,
        0.0,
    )
    texture = tree.nodes.new("ShaderNodeTexImage")
    texture.image = noise
    texture.extension = "REPEAT"
    tree.links.new(stretch.outputs["Vector"], texture.inputs["Vector"])
    threshold = lookup(tree, ppf, cover)
    # A judgement: edges half the tile's sigma wide.
    edge = _math(
        tree,
        "MULTIPLY_ADD",
        _math(tree, "SUBTRACT", texture.outputs["Color"], threshold),
        2.0,
        0.5,
    )
    edge.node.use_clamp = True
    return _math(tree, "MULTIPLY", edge, _math(tree, "GREATER_THAN", cover, 1e-3))


def _smoothed(
    tree: bpy.types.NodeTree,
    slick: bpy.types.NodeSocket | None,
    wake: _Wakes | None,
) -> bpy.types.NodeSocket | None:
    """Where the unresolved roughness is a slick's: under a slick, and in a turbulent
    wake, whose short waves stay damped (Milgram et al. 1993)."""
    if wake is None:
        return slick
    if slick is None:
        return wake.band
    return _math(tree, "MAXIMUM", slick, wake.band)
