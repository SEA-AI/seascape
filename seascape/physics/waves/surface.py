"""The Gerstner surface the waves make, and a hull riding it.

Hull attitude: Jensen, Mansour & Olsen, "Estimation of ship motions using closed-form
expressions", Ocean Engineering 31(1) 61-85, 2004 (doi:10.1016/S0029-8018(03)00108-2):
their pitch, static, without the dynamic or draft (exp(-kT)) factors, is the
least-squares slope of each wave along a box hull; roll is the same across it.
"""

import math
from typing import NamedTuple

import numpy as np

from seascape.physics.waves.spectrum import Wave, columns


def phases(
    field: tuple[Wave, ...],
    east_m: np.ndarray,
    north_m: np.ndarray,
    t_s: float | np.ndarray,
) -> np.ndarray:
    """Each wave's phase at each point and time, waves on the first axis."""
    _, _, kx, ky, omega, phase = columns(field)
    ndim = np.broadcast(east_m, north_m, t_s).ndim

    def col(of_wave: np.ndarray) -> np.ndarray:
        return of_wave.reshape(-1, *[1] * ndim)

    return col(kx) * east_m + col(ky) * north_m - col(omega) * t_s + col(phase)


def gerstner_offset_m(
    field: tuple[Wave, ...], east_m: np.ndarray, north_m: np.ndarray, t_s: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(east, north, up) that a Gerstner surface moves the water at rest: up a
    cos(phase), and -k_hat a sin(phase) along the sea (Tessendorf, "Simulating ocean
    water", SIGGRAPH 2001 course, eq. 29, lambda 1).

    Args:
        field: Gerstner waves, whose offsets add.
        east_m: Where the water sits on a calm sea, east; a Gerstner wave carries it
            round a circle about there.
        north_m: The same, north.
        t_s: Time; each wave's phase advances at its angular frequency.
    """
    phase = phases(field, np.asarray(east_m), np.asarray(north_m), t_s)
    sine, cosine = np.sin(phase), np.cos(phase)
    a, k, kx, ky, *_ = columns(field)
    return (
        np.tensordot(-a * kx / k, sine, axes=1),
        np.tensordot(-a * ky / k, sine, axes=1),
        np.tensordot(a, cosine, axes=1),
    )


def gerstner_rest_m(
    field: tuple[Wave, ...],
    east_m: np.ndarray,
    north_m: np.ndarray,
    t_s: float,
    guess: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Where the water now over a point was at rest, by Newton's method to a
    nanometre. It converges while no crest folds.

    Args:
        field: Gerstner waves, whose offsets add.
        east_m: The point on the displaced surface, east.
        north_m: The same, north.
        t_s: Time; each wave's phase advances at its angular frequency.
        guess: The rest position (east, north) to start from; the point itself if
            None.
    """
    east, north = np.asarray(east_m, float), np.asarray(north_m, float)
    rest_east, rest_north = (east, north) if guess is None else guess
    a, k, k_east, k_north, *_ = columns(field)
    along = a / k
    for _ in range(50):
        phase = phases(field, rest_east, rest_north, t_s)
        sine, cosine = np.sin(phase), np.cos(phase)
        miss_east = rest_east - np.tensordot(along * k_east, sine, 1) - east
        miss_north = rest_north - np.tensordot(along * k_north, sine, 1) - north
        if (
            max(np.abs(miss_east).max(initial=0), np.abs(miss_north).max(initial=0))
            < 1e-9
        ):
            break
        # d(rest + offset)/d(rest).
        ee = 1 - np.tensordot(along * k_east**2, cosine, 1)
        nn = 1 - np.tensordot(along * k_north**2, cosine, 1)
        en = -np.tensordot(along * k_east * k_north, cosine, 1)
        det = ee * nn - en * en
        rest_east = rest_east - (nn * miss_east - en * miss_north) / det
        rest_north = rest_north - (ee * miss_north - en * miss_east) / det
    return rest_east, rest_north


class Waterplane(NamedTuple):
    """A hull's waterplane, centred where the water under it was at rest, `length_m`
    along `heading_rad` and `beam_m` across; a point by default.
    """

    rest_east_m: float
    rest_north_m: float
    heading_rad: float = 0.0
    length_m: float = 0.0
    beam_m: float = 0.0


def attitude(
    field: tuple[Wave, ...], hull: Waterplane, t_s: float
) -> tuple[float, float]:
    """Bow-up pitch and starboard-up roll of the plane fitted to the sea under a hull:
    each wave's least-squares slope at its phase at the hull's rest. The hull
    follows it at once: no inertia.

    Args:
        field: Gerstner waves, whose offsets add.
        hull: The rectangle the plane is fitted over; a wave much shorter than it
            averages out.
        t_s: Time; each wave's phase advances at its angular frequency.
    """
    if not field:
        return 0.0, 0.0
    rest_east_m, rest_north_m, heading_rad, length_m, beam_m = hull
    amplitude, _, k_east, k_north, *_ = columns(field)
    # Starboard of the heading is (cos, -sin).
    k_along = k_east * math.sin(heading_rad) + k_north * math.cos(heading_rad)
    k_across = k_east * math.cos(heading_rad) - k_north * math.sin(heading_rad)
    u, v = k_along * length_m / 2, k_across * beam_m / 2
    at = phases(field, np.array(rest_east_m), np.array(rest_north_m), t_s)
    # 1, x, y are orthogonal over a hull centred at the rest position, so each wave's
    # slope separates.
    rise_along = -amplitude * np.sin(at) * k_along * _lever(u) * np.sinc(v / math.pi)
    rise_across = -amplitude * np.sin(at) * k_across * _lever(v) * np.sinc(u / math.pi)
    return math.atan(float(rise_along.sum())), math.atan(float(rise_across.sum()))


def heave_m(field: tuple[Wave, ...], hull: Waterplane, t_s: float) -> float:
    """Mean height of a Gerstner surface over a hull's waterplane, where a wall-sided
    hull floats (Archimedes): each wave's mean over the rectangle, as `attitude`'s,
    at its phase at the hull's rest. Exact for a point; under a hull, it misses the
    Gerstner surface's second order in steepness. The hull follows it at once: no
    inertia.

    Args:
        field: Gerstner waves, whose offsets add.
        hull: The rectangle the height is averaged over; a wave much shorter than
            it averages out.
        t_s: Time; each wave's phase advances at its angular frequency.
    """
    if not field:
        return 0.0
    rest_east_m, rest_north_m, heading_rad, length_m, beam_m = hull
    amplitude, _, k_east, k_north, *_ = columns(field)
    # Starboard of the heading is (cos, -sin).
    k_along = k_east * math.sin(heading_rad) + k_north * math.cos(heading_rad)
    k_across = k_east * math.cos(heading_rad) - k_north * math.sin(heading_rad)
    u, v = k_along * length_m / 2, k_across * beam_m / 2
    at = phases(field, np.array(rest_east_m), np.array(rest_north_m), t_s)
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
    variance = columns(field)[0] ** 2
    order = np.argsort(variance)[::-1]
    count = int(np.searchsorted(np.cumsum(variance[order]), share * variance.sum())) + 1
    kept = set(order[:count].tolist())
    return tuple(w for i, w in enumerate(field) if i in kept)
