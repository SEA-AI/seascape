"""Gusts, and the windrows and slicks the wind lines up.

Gusts: DNV-RP-C205 2010, section 2.3.2.10, for the turbulence intensity 1 / ln(z / z0),
and section 2.3.4.8, for IEC 61400-1's Kaimal integral scale; the Kaimal spectrum as
IEC 61400-1 (2005) gives it, f S(f) / sigma^2 = 4 x / (1 + 6 x)^(5/3) at x = f L / U.
von Karman, "Progress in the statistical theory of turbulence", PNAS 34(11) 530, 1948
(doi:10.1073/pnas.34.11.530), for a von Karman-type isotropic spectrum,
(k0^2 + k^2)^(-p), whose transects fall as Kaimal's -5/3 at p = 4/3 in 2D. It is a
Matern field of nu = 1/3, whose transect integral scale is
sqrt(pi) Gamma(5/6) / (Gamma(1/3) k0).
Taylor, "The spectrum of turbulence", Proc. R. Soc. A 164(919) 476, 1938
(doi:10.1098/rspa.1938.0032): the eddies are carried past at the mean wind, frozen.
Plant, "A relationship between wind stress and wave slope", JGR 87(C3) 1961-1967, 1982
(doi:10.1029/JC087iC03p01961): a wave grows at 0.04 (u* / c)^2 omega, so only the short
waves keep up with a gust.

Slicks: Leibovich, "The form and dynamics of Langmuir circulations", Annu. Rev. Fluid
Mech. 15 391, 1983 (doi:10.1146/annurev.fl.15.010183.002135): films gather in windrows
along the wind, within 20 degrees, spaced 4.8 s x U (Faller & Woodcock 1964), and none
below 3 m/s. Thorpe, "Langmuir circulation", Annu. Rev. Fluid Mech. 36 55, 2004
(doi:10.1146/annurev.fluid.36.052203.071431): three to ten times as long as their
spacing. ASCE Task Committee on Modeling of Oil Spills, "State-of-the-art review of
modeling transport and fate of oil spills", J. Hydraul. Eng. 122(11) 594, 1996
(doi:10.1061/(ASCE)0733-9429(1996)122:11(594)): a slick drifts at about 3 % of the wind.
"""

import math
from functools import cache
from statistics import NormalDist

import numpy as np

from seascape.physics.waves.slopes import cox_munk_slope
from seascape.physics.waves.spectrum import roughness_m

WINDROW_SPACING_S = 4.8
# Assaf, Gerard & Gordon 1971: finer rows nest between windrows, about an eighth of
# their spacing; Zedel & Farmer 1991: bubble bands about 10 m apart over 10 m/s, which
# Farmer & Li 1995 find barely widen with the wind.
FINE_WINDROW_SPACING_M = 10.0
# WMO Beaufort scale at 10 m: force 7, "foam begins to be blown in streaks along the
# direction of the wind", from 13.9 m/s; force 8, "well-marked streaks", from 17.2 m/s.
BEAUFORT_7_MPS = 13.9
BEAUFORT_8_MPS = 17.2
WINDROW_MIN_WIND_MPS = 3.0
# A judgement inside Thorpe's three to ten.
WINDROW_ASPECT = 5.0
SLICK_DRIFT = 0.03

# IEC 61400-1 via DNV-RP-C205 2.3.4.8: L = 8.1 Lambda, Lambda = 0.7 z below 60 m, at the
# 10 m the wind is given at.
GUST_LENGTH_M = 8.1 * 0.7 * 10.0


def turbulence_intensity(wind_speed_mps: float) -> float:
    """The wind's standard deviation over its mean, at 10 m."""
    if wind_speed_mps <= 0.0:
        return 0.0
    return 1.0 / math.log(10.0 / roughness_m(wind_speed_mps))


def gust_slope_variance(wind_speed_mps: float) -> float:
    """What a gust adds to Cox & Munk's slope variance, per unit u / U. Cox & Munk is
    linear in the wind but for z0's drift, so this is the gust's whole effect."""
    step = 1e-4
    up, down = (cox_munk_slope(wind_speed_mps * (1 + s)) ** 2 for s in (step, -step))
    return (up - down) / (2 * step)


def gaussian_ranks(tile: np.ndarray) -> np.ndarray:
    """`tile` with its values replaced, rank for rank, by a unit Gaussian's quantiles:
    the same pattern, exactly symmetric about 0."""
    ranked = np.empty(tile.size, tile.dtype)
    ranked[np.argsort(tile, axis=None)] = _gaussian_quantiles(tile.size)
    return ranked.reshape(tile.shape)


@cache
def _gaussian_quantiles(count: int) -> np.ndarray:
    """A unit Gaussian's quantiles at the centres of `count` equal shares, ascending.
    Frozen: the cache hands every caller the same array."""
    quantiles = np.vectorize(NormalDist().inv_cdf)((np.arange(count) + 0.5) / count)
    quantiles.setflags(write=False)
    return quantiles


def von_karman_field(
    rng: np.random.Generator, cells: int, spacing_m: float, length_m: float
) -> np.ndarray:
    """A periodic tile, (cells, cells), of von Karman's spectrum at integral scale
    `length_m`, zero mean and unit variance."""
    k0_rad_m = math.sqrt(math.pi) * math.gamma(5 / 6) / (math.gamma(1 / 3) * length_m)
    k = 2 * math.pi * np.fft.fftfreq(cells, spacing_m)
    k_sq = k[:, None] ** 2 + k[None, :] ** 2
    white = np.fft.fft2(rng.standard_normal((cells, cells)))
    field = np.fft.ifft2(white * (k0_rad_m**2 + k_sq) ** (-2 / 3)).real
    field -= field.mean()
    return field / field.std()
