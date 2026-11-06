"""The waves a wind and a swell make, spread about their headings.

Spreading: the wind sea's from Elfouhaily et al. 1997, equations 49, 57 and 59, fitted
to Cox & Munk's upwind-crosswind asymmetry; a swell's Mitsuyasu et al., "Observations of
the directional spectrum of ocean waves using a cloverleaf buoy", Journal of Physical
Oceanography 5(4) 750, 1975 (doi:10.1175/1520-0485(1975)005<0750:OOTDSO>2.0.CO;2), in
the form and with the s_max values given by Goda, "Random Seas and Design of Maritime
Structures", 2nd ed., World Scientific 2000, section 2.3.
"""

import math
from collections.abc import Callable
from dataclasses import replace

import numpy as np

from seascape.physics.earth import GRAVITY_MS2
from seascape.physics.waves.slopes import cox_munk_variances, drawn_slope_variances
from seascape.physics.waves.spectrum import (
    CAPILLARY_PEAK_RAD_M,
    CAPILLARY_SPEED_MPS,
    LOWEST_OF_PEAK,
    WIND_WAVES,
    Wave,
    curvature_spectrum,
    friction_velocity_mps,
    inverse_wave_age,
    peak_omega_rad_s,
    phase_speed_mps,
)

# Elfouhaily et al. 1997, eq. 59; a_m is per u* / c_m.
SPREAD_A0 = math.log(2) / 4
SPREAD_AP = 4.0
SPREAD_AM = 0.13
# Goda 2000: swell with a long decay distance.
SWELL_S_MAX = 75.0

# A judgement: runs of this many neighbouring frequencies span the spread, so their
# crests cross and break off instead of running across the frame as one line.
SPREAD_STRATA = 3

# A judgement: enough headings that a swell's crests do not read as one line.
SWELL_HEADINGS = 4


def spreading_delta(
    omega_rad_s: np.ndarray, wind_speed_mps: float, fetch_m: float | None
) -> np.ndarray:
    """Elfouhaily's Delta(k), eq. 57: the cos 2 theta share of the spread, 1 at the
    peak and least between it and the capillaries.
    """
    k = omega_rad_s**2 / GRAVITY_MS2
    c_m = CAPILLARY_SPEED_MPS
    c = phase_speed_mps(k)
    c_p = wind_speed_mps / inverse_wave_age(wind_speed_mps, fetch_m)
    a_m = SPREAD_AM * friction_velocity_mps(wind_speed_mps) / c_m
    return np.tanh(SPREAD_A0 + SPREAD_AP * (c / c_p) ** 2.5 + a_m * (c_m / c) ** 2.5)


# Angles off downwind the spread's CDF is tabled at.
_DOWNWIND_RAD = np.linspace(-math.pi / 2, math.pi / 2, 361)


def _spread_cdf(delta: np.ndarray) -> np.ndarray:
    """1 + Delta cos 2 theta's CDF over the downwind half, a row per Delta.

    Elfouhaily's form is centrosymmetric, a frozen surface's: folded into the downwind
    half, the waves run with the wind and the slopes keep its statistics.
    """
    theta = _DOWNWIND_RAD
    cdf = theta + math.pi / 2 + delta[:, None] / 2 * np.sin(2 * theta)[None, :]
    return cdf / cdf[:, -1:]


def _downwind_rad(cdf: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Angles off downwind by inverse CDF, one per row of `cdf`."""
    return np.array(
        [np.interp(ui, c, _DOWNWIND_RAD) for ui, c in zip(u, cdf, strict=True)]
    )


def spread_share(
    delta: np.ndarray, low_rad: np.ndarray | float, high_rad: np.ndarray | float
) -> np.ndarray:
    """The share of Elfouhaily's spread (eq. 49) between two angles off downwind."""
    a, b = low_rad, high_rad
    return (b - a + delta / 2 * (np.sin(2 * b) - np.sin(2 * a))) / math.pi


def heading_band(
    delta: np.ndarray, low_rad: np.ndarray, high_rad: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """|E e^(2i theta)| and half its argument over Elfouhaily's spread (eq. 49)
    between two angles off downwind.
    """
    a, b = low_rad, high_rad
    weight = spread_share(delta, a, b) * math.pi
    cos2 = (np.sin(2 * b) - np.sin(2 * a)) / 2 + delta * (
        (b - a) / 2 + (np.sin(4 * b) - np.sin(4 * a)) / 8
    )
    sin2 = (
        -(np.cos(2 * b) - np.cos(2 * a)) / 2
        - delta * (np.cos(4 * b) - np.cos(4 * a)) / 8
    )
    return np.hypot(cos2, sin2) / weight, np.arctan2(sin2, cos2) / 2


def _bins(
    wind_speed_mps: float, fetch_m: float | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Frequency bins log-spaced from the longest wave to the shortest, one per wave:
    their edges, their centres, and the height variance each carries.
    """
    if wind_speed_mps <= 0.0:
        return np.empty(0), np.empty(0), np.empty(0)
    peak = peak_omega_rad_s(wind_speed_mps, fetch_m)
    low = LOWEST_OF_PEAK * peak
    # Elfouhaily's k_m, the minimum phase speed: past it omega^2 = g k fails.
    high = math.sqrt(GRAVITY_MS2 * CAPILLARY_PEAK_RAD_M)
    if low >= high:
        return np.empty(0), np.empty(0), np.empty(0)
    edges = np.geomspace(low, high, WIND_WAVES + 1)
    omega = np.sqrt(edges[:-1] * edges[1:])
    k = omega**2 / GRAVITY_MS2
    # S(k) dk with dk = 2 omega / g d omega: deep water, as the drawn waves run.
    density = curvature_spectrum(k, wind_speed_mps, fetch_m) * k**-3 * 2 * omega
    return edges, omega, density / GRAVITY_MS2 * np.diff(edges)


def _snapped(omega: np.ndarray, snap: Callable[[float], float]) -> np.ndarray:
    """`snap` takes a period, so a loop can round it to a whole fraction of itself."""
    return np.array([2 * math.pi / snap(2 * math.pi / w) for w in omega])


def _off_mean_rad(s: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Angles off the mean direction, cos^2s(theta / 2) by inverse CDF, one per s."""
    theta = np.linspace(-math.pi, math.pi, 721)
    cdf = np.cumsum(np.cos(theta / 2)[None, :] ** (2 * s[:, None]), axis=1)
    cdf /= cdf[:, -1:]
    return np.array([np.interp(ui, c, theta) for ui, c in zip(u, cdf, strict=True)])


def wind_sea(
    wind_speed_mps: float,
    wind_from_deg: float,
    fetch_m: float | None,
    snap: Callable[[float], float],
    rng: np.random.Generator,
) -> tuple[Wave, ...]:
    """Elfouhaily's spectrum and spread, held to Cox & Munk's slope on each axis.

    Where it carries more slope than Cox & Munk measured, as below the 3-21 m/s it was
    fitted over, the whole sea is scaled to their variance on the axis it overshoots
    most, as Elfouhaily tied their short waves to it.
    """
    edges, _, variance = _bins(wind_speed_mps, fetch_m)
    if not len(variance):
        return ()
    # Anywhere in its bin: on the bins' centres, the few waves that carry the slope
    # keep a fixed spacing and beat into bands.
    omega = np.exp(rng.uniform(np.log(edges[:-1]), np.log(edges[1:])))
    # One share of the spread per wave in a run, shuffled so heading does not track
    # frequency.
    runs = np.tile(np.arange(SPREAD_STRATA), (len(variance) // SPREAD_STRATA, 1))
    share = rng.permuted(runs, axis=1).ravel()
    delta = spreading_delta(omega, wind_speed_mps, fetch_m)
    cdf = _spread_cdf(delta)
    off = _downwind_rad(cdf, (share + rng.uniform(size=len(omega))) / SPREAD_STRATA)
    concentration, axis = heading_band(
        delta,
        _downwind_rad(cdf, share / SPREAD_STRATA),
        _downwind_rad(cdf, (share + 1) / SPREAD_STRATA),
    )
    # Wind is named for where it blows from; waves run the other way.
    toward = math.radians(wind_from_deg + 180.0) + off
    phase = rng.uniform(0.0, 2 * math.pi, len(omega))
    # Limitation: a loop rounds each period to a whole fraction of its span, so
    # neighbouring waves can land on the same frequency again, more so in a short loop.
    field = tuple(
        Wave(math.sqrt(2 * v), w, t, p, c, b)
        for v, w, t, p, c, b in zip(
            variance.tolist(),
            _snapped(omega, snap).tolist(),
            toward.tolist(),
            phase.tolist(),
            concentration.tolist(),
            (axis - off).tolist(),
            strict=True,
        )
    )
    drawn = drawn_slope_variances(field, 0.0, math.radians(wind_from_deg))
    measured = cox_munk_variances(wind_speed_mps)
    held = min(1.0, *(m / d for m, d in zip(measured, drawn, strict=True) if d > 0))
    return tuple(replace(w, amplitude_m=w.amplitude_m * math.sqrt(held)) for w in field)


def swell(
    height_m: float,
    period_s: float,
    from_deg: float,
    snap: Callable[[float], float],
    rng: np.random.Generator,
) -> tuple[Wave, ...]:
    """One period, its height split over directions: Hs = 4 sqrt(sum a^2 / 2)."""
    omega = 2 * math.pi / snap(period_s)
    spread = np.full(SWELL_HEADINGS, SWELL_S_MAX)
    toward = math.radians(from_deg + 180.0) + _off_mean_rad(
        spread, rng.uniform(size=SWELL_HEADINGS)
    )
    phase = rng.uniform(0.0, 2 * math.pi, SWELL_HEADINGS)
    amplitude = height_m / 4 * math.sqrt(2 / SWELL_HEADINGS)
    return tuple(
        Wave(amplitude, omega, t, p)
        for t, p in zip(toward.tolist(), phase.tolist(), strict=True)
    )


def snap_error(
    wind_speed_mps: float, fetch_m: float | None, snap: Callable[[float], float]
) -> float:
    """Relative frequency error a loop's snapping costs, weighted by height variance,
    at the bins' centres: the drawn frequencies lie within them.
    """
    _, omega, variance = _bins(wind_speed_mps, fetch_m)
    if not len(variance):
        return 0.0
    error = np.abs(_snapped(omega, snap) - omega) / omega
    return float(np.sum(error * variance) / np.sum(variance))
