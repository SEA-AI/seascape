"""The sea surface as numbers: the earth's curve under it and the waves on it.

Sources
-------
Spectrum: Pierson & Moskowitz, "A proposed spectral form for fully developed wind seas
based on the similarity theory of S. A. Kitaigorodskii", Journal of Geophysical Research
69(24) 5181, 1964 (doi:10.1029/JZ069i024p05181).

Spreading: Mitsuyasu et al., "Observations of the directional spectrum of ocean waves
using a cloverleaf buoy", Journal of Physical Oceanography 5(4) 750, 1975
(doi:10.1175/1520-0485(1975)005<0750:OOTDSO>2.0.CO;2), in the form and with the s_max
values given by Goda, "Random Seas and Design of Maritime Structures", 2nd ed.,
World Scientific 2000, section 2.3.

Whitecaps: Monahan & O'Muircheartaigh, "Optimal power-law description of oceanic
whitecap coverage dependence on wind speed", Journal of Physical Oceanography 10(12)
2094, 1980, for how much; Snyder & Kennedy, "On the formation of whitecaps by a
threshold mechanism. Part I: Basic formalism", Journal of Physical Oceanography 13(8)
1482, 1983, for where: the downward acceleration past a threshold. A pixel whitecaps
over the fraction of it whose unresolved acceleration carries it past.

Slope variance: Cox & Munk, "Measurement of the roughness of the sea surface from
photographs of the sun's glitter", JOSA 44(11) 838, 1954 (doi:10.1364/JOSA.44.000838),
clean-sea fit, equation 13.

Filtering: Bruneton, Neyret & Holzschuch, "Real-time realistic ocean lighting using
seamless transitions from geometry to BRDF", Computer Graphics Forum 29(2) 487, 2010
(doi:10.1111/j.1467-8659.2009.01618.x): a pixel draws the waves longer than its
footprint and takes the slope variance of the rest as roughness.

Microfacet lobe: Walter, Marschner, Li & Torrance, "Microfacet models for refraction
through rough surfaces", EGSR 2007 (doi:10.2312/EGWR/EGSR07/195-206) for GGX; Burley,
"Physically-based shading at Disney", SIGGRAPH 2012 course notes, for the alpha =
roughness^2 convention Cycles follows.

Dispersion: Lamb, "Hydrodynamics", 6th ed., Cambridge University Press 1932, chapter
IX; deep water, omega^2 = g k.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

GRAVITY_MS2 = 9.81

# Waves, end to end. Each step is a published relation or follows from one:
#
#   spectrum        alpha g^2 w^-5 exp(-5/4 (wp/w)^4)  Pierson-Moskowitz 1964
#   peak            wp = 0.877 g / U                    Pierson-Moskowitz 1964
#   direction       cos^2s(theta / 2)                   Mitsuyasu 1975, Goda 2000
#   wavenumber      k = w^2 / g                         deep-water dispersion, Lamb
#   total slope     sqrt(0.003 + 0.00512 U)             Cox & Munk 1954, eq. 13
#   drawn variance  sum of v^2 a^2 k^2 / 2              v = visibility
#   unresolved      total^2 - drawn                     Bruneton 2010, per pixel
#   emissivity      Fresnel over unresolved slopes      Masuda 1988
#   lobe width      alpha = sqrt(2) sigma per axis      Beckmann; GGX alpha = r^2
PM_ALPHA = 8.1e-3
PM_PEAK = 0.877
SPREAD_S_MAX = 10.0
# Goda 2000: swell with a long decay distance.
SWELL_S_MAX = 75.0
# Monahan & O'Muircheartaigh 1980, robust biweight fit, U at 10 m.
WHITECAP_COEFFICIENT = 3.84e-6
WHITECAP_EXPONENT = 3.41
SLOPE_VARIANCE_INTERCEPT = 0.003
SLOPE_VARIANCE_PER_MPS = 0.00512

# A judgement: enough that no single wave shows.
COMPONENTS = 48

# A judgement: enough directions that a swell's crests do not read as one line.
SWELL_COMPONENTS = 4

# A judgement, as a fraction of the peak frequency: PM puts exp(-5/4 x^-4) of the
# height variance below x.
LOWEST_OF_PEAK = 0.7

# Minimum phase speed, 2 pi sqrt(gamma / rho g) at gamma = 0.074 N/m, rho = 1000
# kg/m^3: below it omega^2 = g k fails.
CAPILLARY_WAVELENGTH_M = 0.0173

# Footprints per wavelength: gone at Nyquist, whole at a judgement.
FADE_FOOTPRINTS = (2.0, 4.0)

# Mean radius, IUGG.
EARTH_RADIUS_M = 6_371_000.0


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


def peak_omega_rad_s(wind_speed_mps: float) -> float:
    return PM_PEAK * GRAVITY_MS2 / wind_speed_mps


def wave_slope(wind_speed_mps: float) -> float:
    """Total RMS surface slope. Dimensionless, a tangent."""
    return math.sqrt(SLOPE_VARIANCE_INTERCEPT + SLOPE_VARIANCE_PER_MPS * wind_speed_mps)


def _bins(wind_speed_mps: float) -> tuple[np.ndarray, np.ndarray]:
    """Frequencies log-spaced from the longest wave to the shortest, and the height
    variance each carries."""
    if wind_speed_mps <= 0.0:
        return np.empty(0), np.empty(0)
    peak = peak_omega_rad_s(wind_speed_mps)
    low = LOWEST_OF_PEAK * peak
    high = math.sqrt(2 * math.pi * GRAVITY_MS2 / CAPILLARY_WAVELENGTH_M)
    if low >= high:
        return np.empty(0), np.empty(0)
    edges = np.geomspace(low, high, COMPONENTS + 1)
    omega = np.sqrt(edges[:-1] * edges[1:])
    density = (
        PM_ALPHA * GRAVITY_MS2**2 * omega**-5 * np.exp(-1.25 * (peak / omega) ** 4)
    )
    return omega, density * np.diff(edges)


def _snapped(omega: np.ndarray, snap: Callable[[float], float]) -> np.ndarray:
    """`snap` takes a period, so a loop can round it to a whole fraction of itself."""
    return np.array([2 * math.pi / snap(2 * math.pi / w) for w in omega])


def _off_mean_rad(s: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Angles off the mean direction, cos^2s(theta / 2) by inverse CDF, one per s."""
    theta = np.linspace(-math.pi, math.pi, 721)
    cdf = np.cumsum(np.cos(theta / 2)[None, :] ** (2 * s[:, None]), axis=1)
    cdf /= cdf[:, -1:]
    return np.array([np.interp(ui, c, theta) for ui, c in zip(u, cdf, strict=True)])


def components(
    wind_speed_mps: float,
    wind_from_deg: float,
    snap: Callable[[float], float],
    rng: np.random.Generator,
) -> tuple[Wave, ...]:
    omega, variance = _bins(wind_speed_mps)
    if not len(omega):
        return ()
    ratio = omega / peak_omega_rad_s(wind_speed_mps)
    s = SPREAD_S_MAX * np.where(ratio < 1.0, ratio**5, ratio**-2.5)
    # Wind is named for where it blows from; waves run the other way.
    toward = math.radians(wind_from_deg + 180.0) + _off_mean_rad(
        s, rng.uniform(size=len(omega))
    )
    phase = rng.uniform(0.0, 2 * math.pi, len(omega))
    return tuple(
        Wave(math.sqrt(2 * v), w, t, p)
        for v, w, t, p in zip(
            variance.tolist(),
            _snapped(omega, snap).tolist(),
            toward.tolist(),
            phase.tolist(),
            strict=True,
        )
    )


def swell(
    height_m: float,
    period_s: float,
    from_deg: float,
    snap: Callable[[float], float],
    rng: np.random.Generator,
) -> tuple[Wave, ...]:
    """One period, its height split over directions: Hs = 4 sqrt(sum a^2 / 2)."""
    omega = 2 * math.pi / snap(period_s)
    spread = np.full(SWELL_COMPONENTS, SWELL_S_MAX)
    toward = math.radians(from_deg + 180.0) + _off_mean_rad(
        spread, rng.uniform(size=SWELL_COMPONENTS)
    )
    phase = rng.uniform(0.0, 2 * math.pi, SWELL_COMPONENTS)
    amplitude = height_m / 4 * math.sqrt(2 / SWELL_COMPONENTS)
    return tuple(
        Wave(amplitude, omega, t, p)
        for t, p in zip(toward.tolist(), phase.tolist(), strict=True)
    )


def whitecap_fraction(wind_speed_mps: float) -> float:
    return min(WHITECAP_COEFFICIENT * wind_speed_mps**WHITECAP_EXPONENT, 1.0)


def downward_acceleration_g(
    field: tuple[Wave, ...], east_m: np.ndarray, north_m: np.ndarray, t_s: float
) -> np.ndarray:
    """-d^2 height / dt^2 over g: a k cos(phase) per wave, as omega^2 = g k."""
    ak = np.array([w.amplitude_m * w.k_rad_m for w in field])
    phase = _phase(field, np.asarray(east_m), np.asarray(north_m), t_s)
    return np.tensordot(ak, np.cos(phase), axes=1)


def breaking_threshold_g(wind: tuple[Wave, ...], fraction: float) -> float:
    """The downward acceleration a Gaussian sea of these waves exceeds over `fraction`
    of its area, where it whitecaps."""
    if not wind or fraction <= 0.0:
        return math.inf
    if fraction >= 1.0:
        return -math.inf
    sigma = math.sqrt(resolved_slope_variance(wind))
    return sigma * NormalDist().inv_cdf(1.0 - fraction)


def snap_error(wind_speed_mps: float, snap: Callable[[float], float]) -> float:
    """Relative frequency error a loop's snapping costs, weighted by height variance."""
    omega, variance = _bins(wind_speed_mps)
    if not len(omega):
        return 0.0
    error = np.abs(_snapped(omega, snap) - omega) / omega
    return float(np.sum(error * variance) / np.sum(variance))


def _phase(
    field: tuple[Wave, ...], east_m: np.ndarray, north_m: np.ndarray, t_s: float
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
    phase = _phase(field, np.asarray(east_m), np.asarray(north_m), t_s)
    return np.tensordot(amplitude, np.cos(phase), axes=1)


def slope(
    field: tuple[Wave, ...], east_m: np.ndarray, north_m: np.ndarray, t_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """d height / d east and d height / d north."""
    sine = np.sin(_phase(field, np.asarray(east_m), np.asarray(north_m), t_s))
    a_east = np.array([-w.amplitude_m * w.k_east_rad_m for w in field])
    a_north = np.array([-w.amplitude_m * w.k_north_rad_m for w in field])
    return np.tensordot(a_east, sine, axes=1), np.tensordot(a_north, sine, axes=1)


def resolved_slope_variance(field: tuple[Wave, ...]) -> float:
    return sum((w.amplitude_m * w.k_rad_m) ** 2 / 2 for w in field)


def visibility(wavelength_m: np.ndarray, footprint_m: float) -> np.ndarray:
    """How much of a wave a pixel of `footprint_m` draws: 1 whole, 0 left to the
    roughness."""
    # Smoothstep in footprint squared, as the shader's Map Range fades it.
    low, high = FADE_FOOTPRINTS
    gone, whole = (
        (np.asarray(wavelength_m) / low) ** 2,
        (np.asarray(wavelength_m) / high) ** 2,
    )
    x = np.clip((footprint_m**2 - gone) / (whole - gone), 0, 1)
    return x * x * (3 - 2 * x)


def _shown(
    field: tuple[Wave, ...], footprint_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """Each wave's visibility at `footprint_m`, and its slope variance a^2 k^2 / 2,
    which is also its downward acceleration's over g^2."""
    wavelength = np.array([2 * math.pi / w.k_rad_m for w in field])
    variance = np.array([(w.amplitude_m * w.k_rad_m) ** 2 / 2 for w in field])
    return visibility(wavelength, footprint_m), variance


def pixel_slope_variance(
    wind_speed_mps: float, field: tuple[Wave, ...], footprint_m: float
) -> float:
    """The slope variance a pixel of `footprint_m` does not draw."""
    shown, variance = _shown(field, footprint_m)
    return max(
        wave_slope(wind_speed_mps) ** 2 - float(np.sum(shown**2 * variance)), 0.0
    )


def pixel_acceleration_variance(wind: tuple[Wave, ...], footprint_m: float) -> float:
    """The downward acceleration variance, over g^2, a pixel of `footprint_m` does not
    draw."""
    shown, variance = _shown(wind, footprint_m)
    return float(np.sum((1 - shown**2) * variance))


def whitecap_cover(
    wind: tuple[Wave, ...],
    east_m: np.ndarray,
    north_m: np.ndarray,
    t_s: float,
    footprint_m: float,
    fraction: float,
) -> np.ndarray:
    """How much of a pixel of `footprint_m` whitecaps, for a sea that does over
    `fraction`: its drawn waves' downward acceleration against the threshold, the rest
    Gaussian."""
    shown, _ = _shown(wind, footprint_m)
    ak = np.array([w.amplitude_m * w.k_rad_m for w in wind]) * shown
    drawn = np.tensordot(ak, np.cos(_phase(wind, east_m, north_m, t_s)), axes=1)
    sigma = max(math.sqrt(pixel_acceleration_variance(wind, footprint_m)), 1e-6)
    z = (drawn - breaking_threshold_g(wind, fraction)) / sigma
    return np.vectorize(NormalDist().cdf)(z)


def earth_radius_m(refraction_k: float) -> float:
    """Effective radius, R / (1 - k).

    Surveying's standard refraction treatment: a bent ray over R is straight over R'.
    """
    return EARTH_RADIUS_M / (1.0 - refraction_k)


def sea_z_m(east_m: float, north_m: float, radius_m: float) -> float:
    """Height of the sea at a point, relative to the tangent plane at the origin.

    The parabola that osculates the sphere; one definition, so hull and mesh share it.
    """
    return -(east_m * east_m + north_m * north_m) / (2.0 * radius_m)


def horizon_m(height_m: float, refraction_k: float) -> float:
    """Distance to the horizon from `height_m`, tangent to the effective sphere."""
    return math.sqrt(2.0 * earth_radius_m(refraction_k) * height_m)
