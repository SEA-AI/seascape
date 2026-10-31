"""A hull's wake: its turbulent band, stern foam and Kelvin arms."""

import math
from typing import NamedTuple

import bpy

from seascape.blender.blend import (
    combine,
    lookup,
    scalar,
    vector,
)
from seascape.blender.sea.foam import FOAM_STREAK, FOAM_TILE_M, normal_ppf
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
    bearing = scalar(tree, "ARCTAN2", east, north)
    hull = scalar(
        tree, "MULTIPLY_ADD", time_s, 2 * math.pi / wake.lap_s, wake.start_bearing_rad
    )
    # Clockwise: behind a hull is anticlockwise of it, as far as the hull after it.
    angle = scalar(
        tree,
        "FLOORED_MODULO",
        scalar(tree, "SUBTRACT", hull, bearing),
        2 * math.pi / wake.count,
    )
    radius = vector(tree, "LENGTH", combine(tree, east, north))
    cos_b, sin_b = scalar(tree, "COSINE", bearing), scalar(tree, "SINE", bearing)
    # Round the circle in whole foam tiles, so the bearing's wrap is no seam.
    round_m = FOAM_STREAK * FOAM_TILE_M
    round_m *= max(1, round(2 * math.pi * wake.orbit_m / round_m))
    return _Path(
        along=scalar(tree, "MULTIPLY", angle, wake.orbit_m),
        # Clockwise, starboard is in towards the centre.
        across=scalar(tree, "SUBTRACT", wake.orbit_m, radius),
        d_along=combine(tree, scalar(tree, "MULTIPLY", cos_b, -1.0), sin_b),
        d_across=combine(
            tree,
            scalar(tree, "MULTIPLY", sin_b, -1.0),
            scalar(tree, "MULTIPLY", cos_b, -1.0),
        ),
        fixed=combine(
            tree, scalar(tree, "MULTIPLY", bearing, -round_m / (2 * math.pi)), radius
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
    hull_e = scalar(tree, "MULTIPLY_ADD", time_s, wake.speed_mps * sin_h, e0)
    hull_n = scalar(tree, "MULTIPLY_ADD", time_s, wake.speed_mps * cos_h, n0)
    de = scalar(tree, "SUBTRACT", east, hull_e)
    dn = scalar(tree, "SUBTRACT", north, hull_n)

    def ahead(e: bpy.types.NodeSocket, n: bpy.types.NodeSocket) -> bpy.types.NodeSocket:
        return scalar(
            tree, "MULTIPLY_ADD", e, sin_h, scalar(tree, "MULTIPLY", n, cos_h)
        )

    def starboard(
        e: bpy.types.NodeSocket, n: bpy.types.NodeSocket
    ) -> bpy.types.NodeSocket:
        # Starboard of the heading is (cos, -sin).
        return scalar(
            tree, "MULTIPLY_ADD", e, cos_h, scalar(tree, "MULTIPLY", n, -sin_h)
        )

    return _Path(
        along=scalar(tree, "MULTIPLY", ahead(de, dn), -1.0),
        across=starboard(de, dn),
        d_along=combine(tree, -sin_h, -cos_h),
        d_across=combine(tree, cos_h, -sin_h),
        fixed=combine(tree, ahead(east, north), starboard(east, north)),
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
    ratio = scalar(tree, "DIVIDE", x, half)
    return scalar(
        tree,
        "EXPONENT",
        scalar(tree, "MULTIPLY", scalar(tree, "MULTIPLY", ratio, ratio), -1.0),
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

    Limitation: the arms are Kelvin's straight-path pattern laid along the path, which a
    tight turn bends; sum the arms from past poses if a turn reads wrong.
    """
    ppf = normal_ppf()
    band: bpy.types.NodeSocket | float = 0.0
    foam: bpy.types.NodeSocket | float = 0.0
    bubbles: bpy.types.NodeSocket | float = 0.0
    tilt = None
    for wake in wakes:
        path = _path(tree, wake, where)
        behind = scalar(tree, "GREATER_THAN", path.along, 0.0)
        # Ahead of the hull nothing has aged: unclamped, the exponentials overflow
        # there, and the power below takes a negative base.
        astern = scalar(tree, "MAXIMUM", path.along, 0.0)
        side = scalar(tree, "ABSOLUTE", path.across)
        # A judgement: a beam wide at the stern, widening as distance^(1/5).
        width = scalar(
            tree,
            "MULTIPLY",
            scalar(
                tree,
                "POWER",
                scalar(tree, "MULTIPLY_ADD", astern, 1 / wake.length_m, 1.0),
                0.2,
            ),
            wake.beam_m,
        )
        core = scalar(
            tree,
            "MULTIPLY",
            behind,
            _gaussian(tree, side, scalar(tree, "MULTIPLY", width, 0.5)),
        )
        band = scalar(tree, "MAXIMUM", band, core)
        age = scalar(tree, "DIVIDE", astern, wake.speed_mps)
        fresh = scalar(
            tree, "EXPONENT", scalar(tree, "MULTIPLY", age, -1 / FOAM_EFOLD_S)
        )
        stern = scalar(tree, "MULTIPLY", behind, _gaussian(tree, side, wake.beam_m / 2))
        cover = scalar(tree, "MULTIPLY", stern, fresh)
        foam = scalar(
            tree, "MAXIMUM", foam, _patchy(tree, cover, path.fixed, noise, ppf)
        )
        rising = scalar(
            tree, "EXPONENT", scalar(tree, "MULTIPLY", age, -1 / BUBBLE_EFOLD_S)
        )
        bubbles = scalar(
            tree,
            "MAXIMUM",
            bubbles,
            scalar(
                tree, "MULTIPLY", core, scalar(tree, "MULTIPLY", rising, BUBBLE_GAIN)
            ),
        )
        if wake.steepest_arm**2 / 2 < SEEN_SLOPE_VARIANCE * sea_slope_variance:
            continue
        lasting: bpy.types.NodeSocket = behind
        if wake.orbit_m is not None:
            # A judgement: faded over the radius, past which the straight-path pattern
            # no longer follows the turn.
            fade = scalar(
                tree,
                "EXPONENT",
                scalar(tree, "MULTIPLY", path.along, -1 / wake.orbit_m),
            )
            lasting = scalar(tree, "MULTIPLY", behind, fade)
        # Above the narrowing only the divergent wave, round its peak (Darmon et al.).
        roots = (1.0, -1.0) if wake.froude <= NARROWING_FROUDE else (1.0,)
        for root in roots:
            arm = vector(
                tree, "SCALE", _kelvin(tree, wake, root, path, footprint_m), lasting
            )
            tilt = arm if tilt is None else vector(tree, "ADD", tilt, arm)
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
    side = scalar(tree, "ABSOLUTE", path.across)
    edge = math.tan(KELVIN_HALF_ANGLE_RAD)
    tan_psi = scalar(tree, "DIVIDE", side, scalar(tree, "MAXIMUM", path.along, 1e-3))
    inside = scalar(tree, "LESS_THAN", tan_psi, edge)
    tan_psi = scalar(tree, "MAXIMUM", scalar(tree, "MINIMUM", tan_psi, edge), 1e-4)
    disc = scalar(
        tree,
        "SQRT",
        scalar(
            tree,
            "MAXIMUM",
            scalar(
                tree,
                "MULTIPLY_ADD",
                scalar(tree, "MULTIPLY", tan_psi, tan_psi),
                -8.0,
                1.0,
            ),
            0.0,
        ),
    )
    t = scalar(
        tree,
        "DIVIDE",
        scalar(tree, "MULTIPLY_ADD", disc, root, 1.0),
        scalar(tree, "MULTIPLY", tan_psi, 4.0),
    )
    cos_sq = scalar(tree, "DIVIDE", 1.0, scalar(tree, "MULTIPLY_ADD", t, t, 1.0))
    cos_t = scalar(tree, "SQRT", cos_sq)
    sin_t = scalar(tree, "MULTIPLY", t, cos_t)
    k = scalar(tree, "DIVIDE", GRAVITY_MS2 / wake.speed_mps**2, cos_sq)
    phase = scalar(
        tree,
        "MULTIPLY",
        k,
        scalar(
            tree,
            "MULTIPLY_ADD",
            path.along,
            cos_t,
            scalar(tree, "MULTIPLY", side, sin_t),
        ),
    )
    envelope: bpy.types.NodeSocket | float = 1.0
    if wake.froude > NARROWING_FROUDE:
        peak = wake.peak_angle_rad
        # A judgement: the peak half as wide as its angle.
        off = scalar(
            tree,
            "SUBTRACT",
            scalar(tree, "ARCTANGENT", tan_psi),
            peak,
        )
        envelope = _gaussian(tree, off, peak / 2)
    # Kriebel & Seelig's height, half of it the amplitude.
    amplitude = scalar(
        tree,
        "MULTIPLY",
        scalar(
            tree,
            "POWER",
            scalar(tree, "MAXIMUM", scalar(tree, "DIVIDE", side, wake.length_m), 1e-2),
            -1 / 3,
        ),
        wake.height_scale_m / 2,
    )
    steepest = scalar(
        tree,
        "MINIMUM",
        scalar(tree, "MULTIPLY", amplitude, k),
        math.pi * MAX_STEEPNESS,
    )
    # `waves.visibility`, as the sea's own waves fade.
    y = scalar(tree, "MULTIPLY", scalar(tree, "MULTIPLY", footprint_m, k), 0.5)
    resolved = scalar(
        tree,
        "DIVIDE",
        scalar(tree, "SINE", scalar(tree, "MINIMUM", y, math.pi)),
        y,
    )
    slope = steepest
    for factor in (
        scalar(tree, "SINE", phase),
        envelope,
        inside,
        resolved,
    ):
        slope = scalar(tree, "MULTIPLY", slope, factor)
    sign = scalar(tree, "SIGN", path.across)
    direction = vector(
        tree,
        "ADD",
        vector(tree, "SCALE", path.d_along, cos_t),
        vector(tree, "SCALE", path.d_across, scalar(tree, "MULTIPLY", sin_t, sign)),
    )
    return vector(tree, "SCALE", direction, slope)


def _patchy(
    tree: bpy.types.NodeTree,
    cover: bpy.types.NodeSocket,
    fixed: bpy.types.NodeSocket,
    noise: bpy.types.Image,
    ppf: bpy.types.Image,
) -> bpy.types.NodeSocket:
    """`cover` as white where the unit Gaussian `noise` tops its own `cover` fraction,
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
    edge = scalar(
        tree,
        "MULTIPLY_ADD",
        scalar(tree, "SUBTRACT", texture.outputs["Color"], threshold),
        2.0,
        0.5,
    )
    edge.node.use_clamp = True
    return scalar(tree, "MULTIPLY", edge, scalar(tree, "GREATER_THAN", cover, 1e-3))


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
    return scalar(tree, "MAXIMUM", slick, wake.band)
