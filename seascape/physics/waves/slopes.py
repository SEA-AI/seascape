"""Slope variance: Cox & Munk's totals, and what a pixel leaves undrawn.

Slope variance: Cox & Munk, "Measurement of the roughness of the sea surface from
photographs of the sun's glitter", JOSA 44(11) 838, 1954 (doi:10.1364/JOSA.44.000838),
clean-sea fit, equation 13, and the slick-surface fit beside it, over the oil they laid;
their upwind and crosswind fits split each between the axes.

Filtering: Bruneton, Neyret & Holzschuch, "Real-time realistic ocean lighting using
seamless transitions from geometry to BRDF", Computer Graphics Forum 29(2) 487, 2010
(doi:10.1111/j.1467-8659.2009.01618.x): a pixel draws the waves longer than its
footprint and takes the slope variance of the rest as roughness.

Glitter: Longuet-Higgins, "Reflection and refraction at a random moving surface. II.
Number of specular points in a Gaussian surface", JOSA 50(9) 845, 1960
(doi:10.1364/JOSA.50.000845): specular points per area and slope are p(slope)
E|det H|, H the surface's Hessian, independent of the slope at a point.

Groups: Isserlis, "On a formula for the product-moment coefficient of any order of a
normal frequency distribution in any number of variables", Biometrika 12(1/2) 134, 1918
(doi:10.2307/2331932): a Gaussian sea's slopes have E[s_a s_b s_c s_d] = C_ab C_cd +
C_ac C_bd + C_ad C_bc, so the mean square slope over a footprint varies with
2 sum_ab C_ab^2 averaged over pairs of its points.

Slicks: Alpers & Huhnerfuss, "The damping of ocean waves by surface films: a new look
at an old problem", JGR 94(C5) 6251, 1989 (doi:10.1029/JC094iC05p06251): on the long
side of its Marangoni resonance, at centimetres, a film damps more the shorter the wave,
so a slick takes the shortest waves first.
"""

import math
from collections.abc import Callable
from dataclasses import replace
from functools import cache

import numpy as np

from seascape.physics.earth import GRAVITY_MS2
from seascape.physics.waves.spectrum import (
    CAPILLARY_PEAK_RAD_M,
    LOWEST_OF_PEAK,
    Wave,
    curvature_spectrum,
    peak_omega_rad_s,
    wind_at_m,
)

SLOPE_VARIANCE_INTERCEPT = 0.003
SLOPE_VARIANCE_PER_MPS = 0.00512
SLICK_VARIANCE_INTERCEPT = 0.008
SLICK_VARIANCE_PER_MPS = 0.00156
# Cox & Munk 1954, per axis: (intercept, per m/s) upwind, then crosswind.
AXES_CLEAN = ((0.0, 0.00316), (0.003, 0.00192))
AXES_SLICK = ((0.005, 0.00078), (0.003, 0.00084))

# Where Cox & Munk measured their wind.
COX_MUNK_WIND_HEIGHT_M = 12.5

# A judgement: along over across of the footprint the slope's spread is taken over, a
# grazing pixel's, where the roughness moves the radiance most.
SPREAD_ASPECT = 10.0
# A judgement: the short waves' slope laid on wavenumbers and headings, and the draws
# from it that pair with each other.
SPREAD_WAVENUMBERS, SPREAD_HEADINGS, SPREAD_DRAWS = 128, 64, 600
# A judgement: the footprints the spread is taken at, four a decade, between which it is
# interpolated.
SPREAD_FOOTPRINTS_M = np.geomspace(1e-3, 1e4, 29)


def cox_munk_slope(wind_speed_mps: float) -> float:
    """Total RMS surface slope. Dimensionless, a tangent."""
    wind = wind_at_m(wind_speed_mps, COX_MUNK_WIND_HEIGHT_M)
    return math.sqrt(SLOPE_VARIANCE_INTERCEPT + SLOPE_VARIANCE_PER_MPS * wind)


def cox_munk_slick_slope(wind_speed_mps: float) -> float:
    """Total RMS slope under a slick: Cox & Munk's slick fit, capped at the clean one,
    as a slick never roughens the sea."""
    wind = wind_at_m(wind_speed_mps, COX_MUNK_WIND_HEIGHT_M)
    slick = SLICK_VARIANCE_INTERCEPT + SLICK_VARIANCE_PER_MPS * wind
    return min(math.sqrt(slick), cox_munk_slope(wind_speed_mps))


def _axis_shares(
    wind_speed_mps: float, axes: tuple[tuple[float, float], ...]
) -> tuple[float, float]:
    wind = wind_at_m(wind_speed_mps, COX_MUNK_WIND_HEIGHT_M)
    up, cross = (a + b * wind for a, b in axes)
    return up / (up + cross), cross / (up + cross)


def cox_munk_variances(wind_speed_mps: float) -> tuple[float, float]:
    """Upwind and crosswind slope variance: the total fit, split as the axes' fits."""
    total = cox_munk_slope(wind_speed_mps) ** 2
    up, cross = _axis_shares(wind_speed_mps, AXES_CLEAN)
    # Below the axes' fits' crossing the upwind fit runs to 0: a near-calm sea, with no
    # wind to orient it, is isotropic.
    if up < cross:
        up = cross = 0.5
    return total * up, total * cross


def cox_munk_slick_variances(wind_speed_mps: float) -> tuple[float, float]:
    """Upwind and crosswind slope variance under a slick: the slick total split as its
    axes' fits, upwind capped at the clean sea's, as a film only damps; the clean
    split turns isotropic in light wind, the slick's does not."""
    total = cox_munk_slick_slope(wind_speed_mps) ** 2
    up_share, _ = _axis_shares(wind_speed_mps, AXES_SLICK)
    up = min(total * up_share, cox_munk_variances(wind_speed_mps)[0])
    return up, total - up


@cache
def lattice(count: int) -> np.ndarray:
    """(x, y, t) in [0, 1)^3 by Roberts' additive recurrence on the root of
    x^4 = x + 1 (Roberts 2018, "The unreasonable effectiveness of quasirandom
    sequences"): evenly spread, without a generator."""
    root = 1.0
    for _ in range(64):
        root = (1 + root) ** (1 / 4)
    step = root ** -np.arange(1, 4)
    return np.mod(0.5 + np.outer(np.arange(1, count + 1), step), 1.0)


def slope_variance(field: tuple[Wave, ...]) -> float:
    """Also the downward acceleration's variance over g^2, as omega^2 = g k."""
    return sum((w.amplitude_m * w.k_rad_m) ** 2 / 2 for w in field)


def curvature_variance(field: tuple[Wave, ...]) -> float:
    """m4, the fourth moment of the spectrum: sum a^2 k^4 / 2."""
    return sum((w.amplitude_m * w.k_rad_m**2) ** 2 / 2 for w in field)


def specular_cell_m2(wind: tuple[Wave, ...]) -> float:
    """The sea per specular point, per unit area of slope: 1 / E|det H|.

    Isotropic, as the short waves that carry the curvature are: det H = A^2 - B, A
    normal of variance m4 / 4 and B exponential of mean m4 / 4, so E|det H| =
    m4 / (2 sqrt 3).
    """
    m4 = curvature_variance(wind)
    return 2 * math.sqrt(3) / m4 if m4 > 0.0 else math.inf


def twinkle_hz(wind: tuple[Wave, ...]) -> float:
    """A judgement: specular points live as long as the curvature that makes them, so
    they are re-drawn at its mean frequency, weighted by each wave's share of m4."""
    m4 = curvature_variance(wind)
    if m4 <= 0.0:
        return 0.0
    moment = sum(
        w.omega_rad_s**2 * (w.amplitude_m * w.k_rad_m**2) ** 2 / 2 for w in wind
    )
    return math.sqrt(moment / m4) / (2 * math.pi)


def visibility(wavelength_m: np.ndarray, footprint_m: np.ndarray | float) -> np.ndarray:
    """How much of a wave a pixel of `footprint_m` draws, 1 whole and 0 left to the
    roughness: a box's transfer function, sin(pi x) / (pi x) at x footprints per
    wavelength (Goodman, Introduction to Fourier Optics, 2005, table 2.1), so 2 / pi at
    half a wavelength. Held at 0 past its first zero, as the shader's."""
    x = np.asarray(footprint_m) / np.asarray(wavelength_m)
    return np.where(x < 1.0, np.sinc(x), 0.0)


def mss_spread(
    wind_speed_mps: float, fetch_m: float | None, footprint_m: float
) -> float:
    """How much the mean square of the slope a pixel leaves unresolved varies from one
    of its footprints to the next, relative to its mean: the short waves come in
    groups. `footprint_m` is the footprint's geometric mean; it is `SPREAD_ASPECT`
    times as long as it is wide.

    The short waves are taken as isotropic, as for the glitter."""
    spreads = _mss_spreads(wind_speed_mps, fetch_m)
    at = math.log(footprint_m)
    return float(np.interp(at, np.log(SPREAD_FOOTPRINTS_M), spreads))


@cache
def _mss_spreads(wind_speed_mps: float, fetch_m: float | None) -> np.ndarray:
    return np.array(
        [_mss_spread_at(wind_speed_mps, fetch_m, float(f)) for f in SPREAD_FOOTPRINTS_M]
    )


def _mss_spread_at(
    wind_speed_mps: float, fetch_m: float | None, footprint_m: float
) -> float:
    """Isserlis's spread over a footprint, as `mss_spread`: the relative variance of
    its mean |s|^2 is 2 E[cos^2(theta_1 - theta_2) |H(k_1 - k_2)|^2], k_1 and k_2 drawn
    independently from the slope spectrum the pixel leaves, H the footprint's box
    transfer function."""
    if wind_speed_mps <= 0.0:
        return 0.0
    along = footprint_m * math.sqrt(SPREAD_ASPECT)
    across = footprint_m / math.sqrt(SPREAD_ASPECT)
    # From the sea's longest wave; Elfouhaily's capillaries are gone by ten times their
    # peak.
    longest = LOWEST_OF_PEAK * peak_omega_rad_s(wind_speed_mps, fetch_m)
    k = np.geomspace(
        longest**2 / GRAVITY_MS2, 10 * CAPILLARY_PEAK_RAD_M, SPREAD_WAVENUMBERS
    )
    heading = (np.arange(SPREAD_HEADINGS) + 0.5) * (2 * math.pi / SPREAD_HEADINGS)
    # The footprint across a wave running at `heading` off the view, as the shader's.
    seen_m = np.hypot(across, along * np.cos(heading))
    shown = visibility(2 * math.pi / k[:, None], seen_m[None, :])
    # Slope variance per log wavenumber and radian is B(k) / 2 pi.
    left = curvature_spectrum(k, wind_speed_mps, fetch_m)[:, None] * (1 - shown**2)
    if left.sum() <= 0.0:
        return 0.0
    cdf = np.cumsum(left.ravel()) / left.sum()
    picks = np.searchsorted(cdf, lattice(SPREAD_DRAWS)[:, :2], side="right")
    picks = np.minimum(picks, cdf.size - 1)
    wavenumber = k[picks // SPREAD_HEADINGS]
    angle = heading[picks % SPREAD_HEADINGS]
    k_along, k_across = wavenumber * np.cos(angle), wavenumber * np.sin(angle)
    # np.sinc is sin(pi x) / (pi x); a box of L passes sin(q L / 2) / (q L / 2).
    passed = (
        np.sinc(np.subtract.outer(k_along[:, 0], k_along[:, 1]) * along / (2 * math.pi))
        * np.sinc(
            np.subtract.outer(k_across[:, 0], k_across[:, 1]) * across / (2 * math.pi)
        )
    ) ** 2
    turned = np.cos(np.subtract.outer(angle[:, 0], angle[:, 1])) ** 2
    return math.sqrt(2 * float((turned * passed).mean()))


def filtered(field: tuple[Wave, ...], footprint_m: float) -> tuple[Wave, ...]:
    """The waves a pixel of `footprint_m` draws, each scaled by its visibility."""
    shown = visibility(np.array([2 * math.pi / w.k_rad_m for w in field]), footprint_m)
    return tuple(
        replace(w, amplitude_m=w.amplitude_m * v)
        for w, v in zip(field, shown.tolist(), strict=True)
    )


def resolvable(field: tuple[Wave, ...], footprint_m: float) -> tuple[Wave, ...]:
    """The waves a pixel no finer than `footprint_m` can draw: the rest are gone, a
    footprint a wavelength or wider."""
    return tuple(w for w in field if 2 * math.pi / w.k_rad_m > footprint_m)


def drawn_slope_variances(
    field: tuple[Wave, ...], footprint_m: float, downwind_rad: float
) -> tuple[float, float]:
    """The slope variance a pixel `footprint_m` long draws of `field`, along and
    across `downwind_rad`: `slope_variance(filtered(field, footprint_m))` without
    building the waves."""
    if not field:
        return 0.0, 0.0
    k = np.array([w.k_rad_m for w in field])
    ak = np.array([w.amplitude_m for w in field]) * k
    along = np.cos(np.array([w.toward_rad for w in field]) - downwind_rad) ** 2
    shown = (ak * visibility(2 * math.pi / k, footprint_m)) ** 2 / 2
    return float(np.sum(shown * along)), float(np.sum(shown * (1 - along)))


def slick_survivors(
    wind_speed_mps: float, wind: tuple[Wave, ...], axis_rad: float
) -> tuple[Wave, ...]:
    """The longest waves that fit in Cox & Munk's slick variance on both axes, the
    wind's along `axis_rad`, a budget the clean fit caps: what a slick leaves of
    `wind`, damping the shortest first."""
    budget = np.array(cox_munk_slick_variances(wind_speed_mps))
    left: list[Wave] = []
    # A wavelength goes at every heading at once.
    for k in sorted({w.k_rad_m for w in wind}):
        same = tuple(w for w in wind if w.k_rad_m == k)
        budget -= drawn_slope_variances(same, 0.0, axis_rad)
        if np.any(budget < 0.0):
            break
        left.extend(same)
    return tuple(left)


def unresolved_slope_variances(
    wind_speed_mps: float,
    downwind_rad: float,
    wind: tuple[Wave, ...],
    swell: tuple[Wave, ...],
    footprint_m: float,
    fit: Callable[[float], tuple[float, float]] = cox_munk_variances,
) -> tuple[float, float]:
    """The slope variance a pixel `footprint_m` long leaves undrawn, upwind and
    crosswind: Cox & Munk's per axis less what the wind sea draws, and a swell's own
    undrawn part. A sea short of fetch keeps Cox & Munk's open-ocean total, so the lobe
    takes the slope of its missing long waves."""
    drawn = drawn_slope_variances(wind, footprint_m, downwind_rad)
    swell_all = drawn_slope_variances(swell, 0.0, downwind_rad)
    swell_drawn = drawn_slope_variances(swell, footprint_m, downwind_rad)
    up, cross = (
        max(total - d, 0.0) + a - s
        for total, d, a, s in zip(
            fit(wind_speed_mps), drawn, swell_all, swell_drawn, strict=True
        )
    )
    return up, cross
