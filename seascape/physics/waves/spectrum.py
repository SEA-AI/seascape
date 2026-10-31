"""The waves' spectrum: how much of each wavelength a wind raises.

Spectrum: Elfouhaily, Chapron, Katsaros & Vandemark, "A unified directional spectrum
for long and short wind-driven waves", Journal of Geophysical Research 102(C7) 15781,
1997 (doi:10.1029/97JC00467), omnidirectional, equations 30-44.

Dispersion: Lamb, "Hydrodynamics", 6th ed., Cambridge University Press 1932, chapter
IX; deep water, omega^2 = g k.

Wind height: DNV, "DNV-RP-C205: Environmental conditions and environmental loads",
Det Norske Veritas 2010, section 2.3.2.4: the neutral log profile with Charnock's
roughness.
"""

import math
from dataclasses import dataclass

import numpy as np

from seascape.physics.earth import GRAVITY_MS2

# Elfouhaily et al. 1997: eq. 37 (0.84, X0), eq. 3's range (5), section 5.2.2 (k_m),
# eq. 43 (c_m).
FULLY_DEVELOPED_INVERSE_AGE = 0.84
FETCH_X0 = 2.2e4
YOUNGEST_INVERSE_AGE = 5.0
CAPILLARY_PEAK_RAD_M = 370.0
CAPILLARY_SPEED_MPS = 0.23

# DNV-RP-C205 2.3.2.4: 0.011-0.014 over open sea; the von Karman constant.
CHARNOCK = 0.011
VON_KARMAN = 0.4

# A judgement: frequencies, enough that no single wave shows.
FREQUENCIES = 32

# A judgement, as a fraction of the peak frequency: the spectrum's
# exp(-5/4 (w_p / w)^4) leaves little height variance below it.
LOWEST_OF_PEAK = 0.7

# Minimum phase speed, 2 pi sqrt(gamma / rho g) at gamma = 0.074 N/m, rho = 1000
# kg/m^3: below it omega^2 = g k fails.
CAPILLARY_WAVELENGTH_M = 0.0173


@dataclass(frozen=True)
class Wave:
    """Height a cos(k . x - omega t + phase). k follows from omega, so a loop snapping
    omega cannot break dispersion."""

    amplitude_m: float
    omega_rad_s: float
    toward_rad: float  # clockwise from the ownship's bow
    phase_rad: float

    @property
    def k_rad_m(self) -> float:
        return self.omega_rad_s**2 / GRAVITY_MS2

    @property
    def k_east_rad_m(self) -> float:
        return self.k_rad_m * math.sin(self.toward_rad)

    @property
    def k_north_rad_m(self) -> float:
        return self.k_rad_m * math.cos(self.toward_rad)


def roughness_m(wind_speed_mps: float) -> float:
    """Charnock's z0 = A u*^2 / g, with u* = kappa U10 / ln(10 / z0), to a fixed
    point."""
    z0_m = 1e-4
    for _ in range(50):
        friction_mps = VON_KARMAN * wind_speed_mps / math.log(10.0 / z0_m)
        z0_m = CHARNOCK * friction_mps**2 / GRAVITY_MS2
    return z0_m


def wind_at_m(wind_speed_mps: float, height_m: float) -> float:
    """The wind at `height_m` for `wind_speed_mps` at 10 m, over a neutral sea."""
    if wind_speed_mps <= 0.0:
        return 0.0
    z0_m = roughness_m(wind_speed_mps)
    return wind_speed_mps * math.log(height_m / z0_m) / math.log(10.0 / z0_m)


def friction_velocity_mps(wind_speed_mps: float) -> float:
    """u* of DNV's log profile, not Elfouhaily's own."""
    return VON_KARMAN * wind_speed_mps / math.log(10.0 / roughness_m(wind_speed_mps))


def inverse_wave_age(wind_speed_mps: float, fetch_m: float | None) -> float:
    """U10 / c at the spectrum's peak; a short fetch raises it."""
    if fetch_m is None:
        return FULLY_DEVELOPED_INVERSE_AGE
    x = GRAVITY_MS2 * fetch_m / wind_speed_mps**2
    # Eq. 37, held where eq. 3 is fitted.
    return min(
        FULLY_DEVELOPED_INVERSE_AGE * math.tanh((x / FETCH_X0) ** 0.4) ** -0.75,
        YOUNGEST_INVERSE_AGE,
    )


def peak_omega_rad_s(wind_speed_mps: float, fetch_m: float | None) -> float:
    return GRAVITY_MS2 * inverse_wave_age(wind_speed_mps, fetch_m) / wind_speed_mps


def phase_speed_mps(k_rad_m: np.ndarray) -> np.ndarray:
    """Gravity-capillary phase speed in deep water, the capillary peak at Elfouhaily's
    k_m."""
    return np.sqrt(GRAVITY_MS2 / k_rad_m * (1 + (k_rad_m / CAPILLARY_PEAK_RAD_M) ** 2))


def curvature_spectrum(
    k_rad_m: np.ndarray, wind_speed_mps: float, fetch_m: float | None
) -> np.ndarray:
    """B(k) = k^3 S(k), the omnidirectional curvature spectrum, eq. 30."""
    age = inverse_wave_age(wind_speed_mps, fetch_m)
    k_p = GRAVITY_MS2 * age**2 / wind_speed_mps**2
    k_m, c_m = CAPILLARY_PEAK_RAD_M, CAPILLARY_SPEED_MPS
    c = phase_speed_mps(k_rad_m)
    # Eq. 3.
    gamma = 1.7 if age < 1.0 else 1.7 + 6 * math.log10(age)
    sigma = 0.08 * (1 + 4 * age**-3)
    root = np.sqrt(k_rad_m / k_p)
    # Eq. 2 and 3: Pierson-Moskowitz's shape, JONSWAP's peak enhancement.
    peaked = np.exp(-1.25 * (k_p / k_rad_m) ** 2) * gamma ** np.exp(
        -((root - 1) ** 2) / (2 * sigma**2)
    )
    # Eq. 31, 32 and 34, with c_p = U10 / Omega.
    long = (
        0.5
        * 0.006
        * math.sqrt(age)
        * (wind_speed_mps / age)
        / c
        * peaked
        * np.exp(-age / math.sqrt(10) * (root - 1))
    )
    friction_mps = friction_velocity_mps(wind_speed_mps)
    gain = 1 if friction_mps <= c_m else 3
    # Eq. 44, negative below u* = c_m / e, where the paper's fit runs out.
    alpha_m = max(0.01 * (1 + gain * math.log(friction_mps / c_m)), 0.0)
    # Eq. 40 and 41.
    short = 0.5 * alpha_m * c_m / c * peaked * np.exp(-0.25 * (k_rad_m / k_m - 1) ** 2)
    return long + short
