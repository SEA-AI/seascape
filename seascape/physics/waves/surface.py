"""The Gerstner surface the waves make, and a hull riding it.

Hull attitude: Jensen, Mansour & Olsen, "Estimation of ship motions using closed-form
expressions", Ocean Engineering 31(1) 61-85, 2004 (doi:10.1016/S0029-8018(03)00108-2):
their pitch, static, without the dynamic or draft (exp(-kT)) factors, is the
least-squares slope of each wave along a box hull; roll is the same across it.
"""

import math
from collections.abc import Callable

import numpy as np

from seascape.physics.waves.spectrum import Wave


def phases(
    field: tuple[Wave, ...],
    east_m: np.ndarray,
    north_m: np.ndarray,
    t_s: float | np.ndarray,
) -> np.ndarray:
    return np.array(
        [
            w.k_east_rad_m * east_m
            + w.k_north_rad_m * north_m
            - w.omega_rad_s * t_s
            + w.phase_rad
            for w in field
        ]
    )


def height_m(
    field: tuple[Wave, ...], east_m: np.ndarray, north_m: np.ndarray, t_s: float
) -> np.ndarray:
    amplitude = np.array([w.amplitude_m for w in field])
    phase = phases(field, np.asarray(east_m), np.asarray(north_m), t_s)
    return np.tensordot(amplitude, np.cos(phase), axes=1)


def slope(
    field: tuple[Wave, ...], east_m: np.ndarray, north_m: np.ndarray, t_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """d height / d east and d height / d north."""
    sine = np.sin(phases(field, np.asarray(east_m), np.asarray(north_m), t_s))
    a_east = np.array([-w.amplitude_m * w.k_east_rad_m for w in field])
    a_north = np.array([-w.amplitude_m * w.k_north_rad_m for w in field])
    return np.tensordot(a_east, sine, axes=1), np.tensordot(a_north, sine, axes=1)


def gerstner_offset_m(
    field: tuple[Wave, ...], east_m: np.ndarray, north_m: np.ndarray, t_s: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(east, north, up) that `field`'s Gerstner surface moves the water at rest at
    (`east_m`, `north_m`): up a cos(phase), and -k_hat a sin(phase) along the sea
    (Tessendorf, "Simulating ocean water", SIGGRAPH 2001 course, eq. 29, lambda 1)."""
    phase = phases(field, np.asarray(east_m), np.asarray(north_m), t_s)
    sine, cosine = np.sin(phase), np.cos(phase)
    along_east = np.array([-w.amplitude_m * w.k_east_rad_m / w.k_rad_m for w in field])
    along_north = np.array(
        [-w.amplitude_m * w.k_north_rad_m / w.k_rad_m for w in field]
    )
    amplitude = np.array([w.amplitude_m for w in field])
    return (
        np.tensordot(along_east, sine, axes=1),
        np.tensordot(along_north, sine, axes=1),
        np.tensordot(amplitude, cosine, axes=1),
    )


def gerstner_rest_m(
    field: tuple[Wave, ...], east_m: np.ndarray, north_m: np.ndarray, t_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """Where the water now over (`east_m`, `north_m`) was at rest."""
    east, north = np.asarray(east_m, float), np.asarray(north_m, float)
    _, rest_east, rest_north = _gerstner_height_m(
        field, east, north, t_s, (east, north)
    )
    return rest_east, rest_north


def gerstner_normal(
    field: tuple[Wave, ...], east_m: np.ndarray, north_m: np.ndarray, t_s: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The unit normal of `field`'s Gerstner surface at the water at rest at
    (`east_m`, `north_m`): the cross product of the surface's derivatives along east
    and north at rest."""
    phase = phases(field, np.asarray(east_m), np.asarray(north_m), t_s)
    sine, cosine = np.sin(phase), np.cos(phase)

    def summed(of: Callable[[Wave], float], wave: np.ndarray) -> np.ndarray:
        return np.tensordot(np.array([of(w) for w in field]), wave, axes=1)

    # d(offset)/d(rest): -a k_i k_j / k cos along, -a k_j sin up.
    ee = 1 - summed(lambda w: w.amplitude_m * w.k_east_rad_m**2 / w.k_rad_m, cosine)
    nn = 1 - summed(lambda w: w.amplitude_m * w.k_north_rad_m**2 / w.k_rad_m, cosine)
    en = -summed(
        lambda w: w.amplitude_m * w.k_east_rad_m * w.k_north_rad_m / w.k_rad_m, cosine
    )
    ue = -summed(lambda w: w.amplitude_m * w.k_east_rad_m, sine)
    un = -summed(lambda w: w.amplitude_m * w.k_north_rad_m, sine)
    # (ee, en, ue) x (en, nn, un)
    normal = np.array([en * un - ue * nn, ue * en - ee * un, ee * nn - en * en])
    normal /= np.linalg.norm(normal, axis=0)
    return normal[0], normal[1], normal[2]


def _gerstner_height_m(
    field: tuple[Wave, ...],
    east_m: np.ndarray,
    north_m: np.ndarray,
    t_s: float,
    rest: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The Gerstner surface's height over (`east_m`, `north_m`), and the water's rest
    there, by Newton's method from `rest`, to a nanometre."""
    rest_east, rest_north = rest
    k_east = np.array([w.k_east_rad_m for w in field])
    k_north = np.array([w.k_north_rad_m for w in field])
    along = np.array([w.amplitude_m / w.k_rad_m for w in field])
    for _ in range(50):
        phase = phases(field, rest_east, rest_north, t_s)
        sine, cosine = np.sin(phase), np.cos(phase)
        miss_east = rest_east - np.tensordot(along * k_east, sine, 1) - east_m
        miss_north = rest_north - np.tensordot(along * k_north, sine, 1) - north_m
        if (
            max(np.abs(miss_east).max(initial=0), np.abs(miss_north).max(initial=0))
            < 1e-9
        ):
            break
        # d(rest + offset)/d(rest), as `gerstner_normal`'s.
        ee = 1 - np.tensordot(along * k_east**2, cosine, 1)
        nn = 1 - np.tensordot(along * k_north**2, cosine, 1)
        en = -np.tensordot(along * k_east * k_north, cosine, 1)
        det = ee * nn - en * en
        rest_east = rest_east - (nn * miss_east - en * miss_north) / det
        rest_north = rest_north - (ee * miss_north - en * miss_east) / det
    _, _, up = gerstner_offset_m(field, rest_east, rest_north, t_s)
    return up, rest_east, rest_north


def gerstner_hit(
    field: tuple[Wave, ...],
    origin_m: np.ndarray,
    direction: np.ndarray,
    t_s: float,
    step_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Where rays from `origin_m` along unit `direction` (both 3 x n, east, north,
    up) first meet `field`'s Gerstner surface: the water's rest there, and whether
    they meet it at all before rising clear of the crests. Marched along the sea by
    the ray's height over the water, over the steepest the surface can rise, or by
    `step_m` where that is less; then halved to a micrometre. A crest thinner than
    `step_m` is missed."""
    origin, direction = np.asarray(origin_m, float), np.asarray(direction, float)
    crest = sum(w.amplitude_m for w in field)
    # The surface's slope is the offset's over its stretch, 1 - sum(a k) at least. A
    # field steeper than 1 has no such bound, and marches at `step_m`.
    steepness = sum(w.amplitude_m * w.k_rad_m for w in field)
    steepest = steepness / (1 - steepness) if steepness < 1 else math.inf
    run = np.hypot(direction[0], direction[1])
    # From where the ray is first as low as a crest.
    start = np.where(
        direction[2] < 0, np.maximum((origin[2] - crest) / -direction[2], 0.0), 0.0
    )
    near, far = start.copy(), start.copy()
    rest_east = origin[0] + start * direction[0]
    rest_north = origin[1] + start * direction[1]
    met = np.zeros(origin.shape[1], bool)
    gap = np.zeros(origin.shape[1])
    # Rising no faster than `steepest` along the sea, the water cannot close the gap
    # within a step.
    closing = np.maximum(steepest * run - direction[2], 1e-6)
    marching = np.arange(origin.shape[1])
    while marching.size:
        i = marching
        step = np.maximum(step_m / np.maximum(run[i], 1e-6), gap[i] / closing[i])
        far[i] = near[i] + step
        at = origin[:, i] + far[i] * direction[:, i]
        # The water under a point moves with it, near enough to start from.
        up, rest_east[i], rest_north[i] = _gerstner_height_m(
            field,
            at[0],
            at[1],
            t_s,
            (
                rest_east[i] + step * direction[0, i],
                rest_north[i] + step * direction[1, i],
            ),
        )
        gap[i] = at[2] - up
        below = gap[i] < 0
        met[i] = below
        clear = (at[2] > crest) & (direction[2, i] >= 0)
        sunk = at[2] < -crest
        going = ~below & ~clear & ~sunk
        near[i[going]] = far[i[going]]
        marching = i[going]
    # The surface lies between `near` and `far` on every ray that met it.
    i = np.flatnonzero(met)
    widest = float((far[i] - near[i]).max(initial=1e-6))
    for _ in range(int(np.ceil(np.log2(widest / 1e-6)))):
        middle = (near[i] + far[i]) / 2
        at = origin[:, i] + middle * direction[:, i]
        up, rest_east[i], rest_north[i] = _gerstner_height_m(
            field, at[0], at[1], t_s, (rest_east[i], rest_north[i])
        )
        below = at[2] < up
        far[i] = np.where(below, middle, far[i])
        near[i] = np.where(below, near[i], middle)
    return rest_east, rest_north, met


def attitude(
    field: tuple[Wave, ...],
    rest_east_m: float,
    rest_north_m: float,
    heading_rad: float,
    length_m: float,
    beam_m: float,
    t_s: float,
) -> tuple[float, float]:
    """Bow-up pitch and starboard-up roll of the plane fitted to the sea under a hull:
    each wave's least-squares slope at its phase where the water under the centre was
    at rest, (`rest_east_m`, `rest_north_m`). The hull follows it at once: no
    inertia."""
    if not field:
        return 0.0, 0.0
    k_east = np.array([w.k_east_rad_m for w in field])
    k_north = np.array([w.k_north_rad_m for w in field])
    # Starboard of the heading is (cos, -sin).
    k_along = k_east * math.sin(heading_rad) + k_north * math.cos(heading_rad)
    k_across = k_east * math.cos(heading_rad) - k_north * math.sin(heading_rad)
    u, v = k_along * length_m / 2, k_across * beam_m / 2
    at = phases(field, np.array(rest_east_m), np.array(rest_north_m), t_s)
    amplitude = np.array([w.amplitude_m for w in field])
    # 1, x, y are orthogonal over a hull centred at the rest position, so each wave's
    # slope separates.
    rise_along = -amplitude * np.sin(at) * k_along * _lever(u) * np.sinc(v / math.pi)
    rise_across = -amplitude * np.sin(at) * k_across * _lever(v) * np.sinc(u / math.pi)
    return math.atan(float(rise_along.sum())), math.atan(float(rise_across.sum()))


def heave_m(
    field: tuple[Wave, ...],
    rest_east_m: float,
    rest_north_m: float,
    heading_rad: float,
    length_m: float,
    beam_m: float,
    t_s: float,
) -> float:
    """Mean height of `field`'s Gerstner surface over a hull's waterplane, where a
    wall-sided hull floats (Archimedes): each wave's mean over the rectangle, as
    `attitude`'s, at its phase where the water under the centre was at rest,
    (`rest_east_m`, `rest_north_m`). Exact
    for a point; under a hull, it misses the Gerstner surface's second order in
    steepness. The hull follows it at once: no inertia."""
    if not field:
        return 0.0
    k_east = np.array([w.k_east_rad_m for w in field])
    k_north = np.array([w.k_north_rad_m for w in field])
    # Starboard of the heading is (cos, -sin).
    k_along = k_east * math.sin(heading_rad) + k_north * math.cos(heading_rad)
    k_across = k_east * math.cos(heading_rad) - k_north * math.sin(heading_rad)
    u, v = k_along * length_m / 2, k_across * beam_m / 2
    at = phases(field, np.array(rest_east_m), np.array(rest_north_m), t_s)
    amplitude = np.array([w.amplitude_m for w in field])
    under = amplitude * np.cos(at) * np.sinc(u / math.pi) * np.sinc(v / math.pi)
    return float(under.sum())


def _lever(u: np.ndarray) -> np.ndarray:
    """3 (sin u - u cos u) / u^3; its series near 0, where the formula cancels."""
    small = np.abs(u) < 1e-3
    safe = np.where(small, 1.0, u)
    return np.where(
        small, 1 - u**2 / 10, 3 * (np.sin(safe) - safe * np.cos(safe)) / safe**3
    )


def tallest(field: tuple[Wave, ...], share: float) -> tuple[Wave, ...]:
    """The fewest waves carrying `share` of the height variance, in field order."""
    variance = np.array([w.amplitude_m**2 for w in field])
    order = np.argsort(variance)[::-1]
    count = int(np.searchsorted(np.cumsum(variance[order]), share * variance.sum())) + 1
    kept = set(order[:count].tolist())
    return tuple(w for i, w in enumerate(field) if i in kept)
