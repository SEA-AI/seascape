"""The sea surface as numbers: the earth's curve under it and the waves on it.

Sources
-------
Spectrum: Elfouhaily, Chapron, Katsaros & Vandemark, "A unified directional spectrum
for long and short wind-driven waves", Journal of Geophysical Research 102(C7) 15781,
1997 (doi:10.1029/97JC00467), omnidirectional, equations 30-44.

Spreading: Mitsuyasu et al., "Observations of the directional spectrum of ocean waves
using a cloverleaf buoy", Journal of Physical Oceanography 5(4) 750, 1975
(doi:10.1175/1520-0485(1975)005<0750:OOTDSO>2.0.CO;2), in the form and with the s_max
values given by Goda, "Random Seas and Design of Maritime Structures", 2nd ed.,
World Scientific 2000, section 2.3.

Whitecaps: Monahan & O'Muircheartaigh, "Optimal power-law description of oceanic
whitecap coverage dependence on wind speed", Journal of Physical Oceanography 10(12)
2094, 1980, for how much; Snyder & Kennedy, "On the formation of whitecaps by a
threshold mechanism. Part I: Basic formalism", Journal of Physical Oceanography 13(8)
1482, 1983, for where: the downward acceleration past a threshold, set from the
coverage (Tse, McGill & Kelly, SPIE Ocean Optics X 1302, 505, 1990; Tessendorf, Reinhard
& Gao, "Whitecap phenomenology for ocean surface simulation", Clemson University report,
2023). Per
pixel: Dupuy & Bruneton, "Real-time animation and rendering of ocean whitecaps",
SIGGRAPH Asia 2012 Technical Briefs 15 (doi:10.1145/2407746.2407761), whose Jacobian
criterion is this acceleration to first order in the choppiness; the drawn waves set the
mean, the rest a Gaussian spread.

Slope variance: Cox & Munk, "Measurement of the roughness of the sea surface from
photographs of the sun's glitter", JOSA 44(11) 838, 1954 (doi:10.1364/JOSA.44.000838),
clean-sea fit, equation 13, and the slick-surface fit beside it, over the oil they laid.

Slicks: Leibovich, "The form and dynamics of Langmuir circulations", Annu. Rev. Fluid
Mech. 15 391, 1983 (doi:10.1146/annurev.fl.15.010183.002135): films gather in windrows
along the wind, within 20 degrees, spaced 4.8 s x U (Faller & Woodcock 1964), and none
below 3 m/s. Thorpe, "Langmuir circulation", Annu. Rev. Fluid Mech. 36 55, 2004
(doi:10.1146/annurev.fluid.36.052203.071431): three to ten times as long as their
spacing. ASCE Task Committee on Modeling of Oil Spills, "State-of-the-art review of
modeling transport and fate of oil spills", J. Hydraul. Eng. 122(11) 594, 1996
(doi:10.1061/(ASCE)0733-9429(1996)122:11(594)): a slick drifts at about 3 % of the wind.
Alpers & Huhnerfuss, "The damping of ocean waves by surface films: a new look at an old
problem", JGR 94(C5) 6251, 1989 (doi:10.1029/JC094iC05p06251): on the long side of its
Marangoni resonance, at centimetres, a film damps more the shorter the wave, so a slick
takes the shortest waves first.

Filtering: Bruneton, Neyret & Holzschuch, "Real-time realistic ocean lighting using
seamless transitions from geometry to BRDF", Computer Graphics Forum 29(2) 487, 2010
(doi:10.1111/j.1467-8659.2009.01618.x): a pixel draws the waves longer than its
footprint and takes the slope variance of the rest as roughness.

Glitter: Longuet-Higgins, "Reflection and refraction at a random moving surface. II.
Number of specular points in a Gaussian surface", JOSA 50(9) 845, 1960
(doi:10.1364/JOSA.50.000845): specular points per area and slope are p(slope)
E|det H|, H the surface's Hessian, independent of the slope at a point.

Wind height: DNV, "DNV-RP-C205: Environmental conditions and environmental loads",
Det Norske Veritas 2010, section 2.3.2.4: the neutral log profile with Charnock's
roughness.

Hull attitude: Jensen, Mansour & Olsen, "Estimation of ship motions using closed-form
expressions", Ocean Engineering 31(1) 61-85, 2004 (doi:10.1016/S0029-8018(03)00108-2):
their pitch, static, without the dynamic or draft (exp(-kT)) factors, is the
least-squares slope of each wave along a box hull; roll is the same across it.

Dispersion: Lamb, "Hydrodynamics", 6th ed., Cambridge University Press 1932, chapter
IX; deep water, omega^2 = g k.

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
"""

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import cache
from statistics import NormalDist

import numpy as np

GRAVITY_MS2 = 9.81

# Waves, end to end. Each step is a published relation or follows from one:
#
#   wind            U10 ln(z / z0) / ln(10 / z0)        DNV-RP-C205 2010
#   spectrum        k^-3 (B_long + B_short)             Elfouhaily et al. 1997
#   peak            kp = g Omega^2 / U10^2              deep-water dispersion
#   wave age        Omega 0.84, more at a short fetch   Elfouhaily et al. 1997, eq. 37
#   direction       cos^2s(theta / 2)                   Mitsuyasu 1975, Goda 2000
#   wavenumber      k = w^2 / g                         deep-water dispersion, Lamb
#   total slope     sqrt(0.003 + 0.00512 U(12.5 m))     Cox & Munk 1954, eq. 13
#   drawn variance  sum of v^2 a^2 k^2 / 2              v = visibility
#   unresolved      total^2 - drawn, + undrawn swell    Bruneton 2010, per pixel
#   whitecaps       P(sum v a k cos(phase) > threshold) Snyder & Kennedy 1983
#   glitter         one specular point per 1 / E|det H| Longuet-Higgins 1960
#   gust            u / U, sigma 1 / ln(10 / z0)        DNV-RP-C205
#   gust field      (k0^2 + k^2)^(-4/3), L of Kaimal's  von Karman 1948, IEC 61400-1
#   gusty slope     + u / U dCM^2 / d ln U, undrawn     Cox & Munk, Plant 1982
#   slick slope     sqrt(0.008 + 0.00156 U(12.5 m))     Cox & Munk 1954
#   windrows        4.8 s x U apart, 3-10x as long      Leibovich 1983, Thorpe 2004
#   windrow cutoff  none below 3 m/s                    Leibovich 1983
# Elfouhaily et al. 1997: eq. 37 (0.84, X0), eq. 3's range (5), section 5.2.2 (k_m),
# eq. 43 (c_m).
FULLY_DEVELOPED_INVERSE_AGE = 0.84
FETCH_X0 = 2.2e4
YOUNGEST_INVERSE_AGE = 5.0
CAPILLARY_PEAK_RAD_M = 370.0
CAPILLARY_SPEED_MPS = 0.23
SPREAD_S_MAX = 10.0
# Goda 2000: swell with a long decay distance.
SWELL_S_MAX = 75.0
# Monahan & O'Muircheartaigh 1980, robust biweight fit, U at 10 m.
WHITECAP_COEFFICIENT = 3.84e-6
WHITECAP_EXPONENT = 3.41
SLOPE_VARIANCE_INTERCEPT = 0.003
SLOPE_VARIANCE_PER_MPS = 0.00512
SLICK_VARIANCE_INTERCEPT = 0.008
SLICK_VARIANCE_PER_MPS = 0.00156
WINDROW_SPACING_S = 4.8
WINDROW_MIN_WIND_MPS = 3.0
# A judgement inside Thorpe's three to ten.
WINDROW_ASPECT = 5.0
SLICK_DRIFT = 0.03

# Where Cox & Munk measured their wind.
COX_MUNK_WIND_HEIGHT_M = 12.5

# DNV-RP-C205 2.3.2.4: 0.011-0.014 over open sea; the von Karman constant.
CHARNOCK = 0.011
VON_KARMAN = 0.4

# A judgement: frequencies, enough that no single wave shows.
FREQUENCIES = 32

# A judgement: each frequency at several headings, so its crests cross and break off
# instead of running across the frame as one line.
WIND_HEADINGS = 3

# A judgement: enough headings that a swell's crests do not read as one line.
SWELL_HEADINGS = 4

# A judgement, as a fraction of the peak frequency: the spectrum's
# exp(-5/4 (w_p / w)^4) leaves little height variance below it.
LOWEST_OF_PEAK = 0.7

# Minimum phase speed, 2 pi sqrt(gamma / rho g) at gamma = 0.074 N/m, rho = 1000
# kg/m^3: below it omega^2 = g k fails.
CAPILLARY_WAVELENGTH_M = 0.0173

# Footprints per wavelength: gone at Nyquist, whole at a judgement.
FADE_FOOTPRINTS = (2.0, 4.0)

# IEC 61400-1 via DNV-RP-C205 2.3.4.8: L = 8.1 Lambda, Lambda = 0.7 z below 60 m, at the
# 10 m the wind is given at.
GUST_LENGTH_M = 8.1 * 0.7 * 10.0

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


def _roughness_m(wind_speed_mps: float) -> float:
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
    z0_m = _roughness_m(wind_speed_mps)
    return wind_speed_mps * math.log(height_m / z0_m) / math.log(10.0 / z0_m)


def turbulence_intensity(wind_speed_mps: float) -> float:
    """The wind's standard deviation over its mean, at 10 m."""
    if wind_speed_mps <= 0.0:
        return 0.0
    return 1.0 / math.log(10.0 / _roughness_m(wind_speed_mps))


def gust_slope_variance(wind_speed_mps: float) -> float:
    """What a gust adds to Cox & Munk's slope variance, per unit u / U. Cox & Munk is
    linear in the wind but for z0's drift, so this is the gust's whole effect."""
    step = 1e-4
    up, down = (cox_munk_slope(wind_speed_mps * (1 + s)) ** 2 for s in (step, -step))
    return (up - down) / (2 * step)


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


def friction_velocity_mps(wind_speed_mps: float) -> float:
    """u* of DNV's log profile, not Elfouhaily's own."""
    return VON_KARMAN * wind_speed_mps / math.log(10.0 / _roughness_m(wind_speed_mps))


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


def curvature_spectrum(
    k_rad_m: np.ndarray, wind_speed_mps: float, fetch_m: float | None
) -> np.ndarray:
    """B(k) = k^3 S(k), the omnidirectional curvature spectrum, eq. 30."""
    age = inverse_wave_age(wind_speed_mps, fetch_m)
    k_p = GRAVITY_MS2 * age**2 / wind_speed_mps**2
    k_m, c_m = CAPILLARY_PEAK_RAD_M, CAPILLARY_SPEED_MPS
    c = np.sqrt(GRAVITY_MS2 / k_rad_m * (1 + (k_rad_m / k_m) ** 2))
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


def _bins(
    wind_speed_mps: float, fetch_m: float | None
) -> tuple[np.ndarray, np.ndarray]:
    """Frequencies log-spaced from the longest wave to the shortest, and the height
    variance each carries."""
    if wind_speed_mps <= 0.0:
        return np.empty(0), np.empty(0)
    peak = peak_omega_rad_s(wind_speed_mps, fetch_m)
    low = LOWEST_OF_PEAK * peak
    high = math.sqrt(2 * math.pi * GRAVITY_MS2 / CAPILLARY_WAVELENGTH_M)
    if low >= high:
        return np.empty(0), np.empty(0)
    edges = np.geomspace(low, high, FREQUENCIES + 1)
    omega = np.sqrt(edges[:-1] * edges[1:])
    k = omega**2 / GRAVITY_MS2
    # S(k) dk with dk = 2 omega / g d omega: deep water, as the drawn waves run.
    density = curvature_spectrum(k, wind_speed_mps, fetch_m) * k**-3 * 2 * omega
    return omega, density / GRAVITY_MS2 * np.diff(edges)


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
    omega, variance = _bins(wind_speed_mps, fetch_m)
    if not len(omega):
        return ()
    # Stratified, so a frequency's headings fall in separate shares of the spread.
    share = np.tile(np.arange(WIND_HEADINGS), len(omega))
    omega = np.repeat(omega, WIND_HEADINGS)
    variance = np.repeat(variance / WIND_HEADINGS, WIND_HEADINGS)
    ratio = omega / peak_omega_rad_s(wind_speed_mps, fetch_m)
    s = SPREAD_S_MAX * np.where(ratio < 1.0, ratio**5, ratio**-2.5)
    # Wind is named for where it blows from; waves run the other way.
    toward = math.radians(wind_from_deg + 180.0) + _off_mean_rad(
        s, (share + rng.uniform(size=len(omega))) / WIND_HEADINGS
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


def whitecap_fraction(wind_speed_mps: float) -> float:
    return min(WHITECAP_COEFFICIENT * wind_speed_mps**WHITECAP_EXPONENT, 1.0)


@cache
def _gust_mean_cover(wind_speed_mps: float) -> float:
    sigma = turbulence_intensity(wind_speed_mps)
    # Gauss-Hermite over the normal gust.
    x, weight = np.polynomial.hermite_e.hermegauss(32)
    local = [whitecap_fraction(wind_speed_mps * max(1 + sigma * g, 0.0)) for g in x]
    return float(np.dot(weight, local) / weight.sum())


def gusty_whitecap_fraction(wind_speed_mps: float, gust: float) -> float:
    """The cover where the wind is `1 + gust` times its mean. Monahan measured the
    mean wind over a gusty sea, so the cover is scaled to average his over the
    gusts."""
    mean = _gust_mean_cover(wind_speed_mps)
    if mean <= 0.0:
        return 0.0
    cover = whitecap_fraction(wind_speed_mps * max(1 + gust, 0.0))
    return min(cover * whitecap_fraction(wind_speed_mps) / mean, 1.0)


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
    sigma = math.sqrt(slope_variance(wind))
    return sigma * NormalDist().inv_cdf(1.0 - fraction)


def snap_error(
    wind_speed_mps: float, fetch_m: float | None, snap: Callable[[float], float]
) -> float:
    """Relative frequency error a loop's snapping costs, weighted by height variance."""
    omega, variance = _bins(wind_speed_mps, fetch_m)
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


def attitude(
    field: tuple[Wave, ...],
    east_m: float,
    north_m: float,
    heading_rad: float,
    length_m: float,
    beam_m: float,
    t_s: float,
) -> tuple[float, float]:
    """Bow-up pitch and starboard-up roll of the plane fitted to the sea under a hull.
    The hull follows it at once: no inertia."""
    if not field:
        return 0.0, 0.0
    k_east = np.array([w.k_east_rad_m for w in field])
    k_north = np.array([w.k_north_rad_m for w in field])
    # Starboard of the heading is (cos, -sin).
    k_along = k_east * math.sin(heading_rad) + k_north * math.cos(heading_rad)
    k_across = k_east * math.cos(heading_rad) - k_north * math.sin(heading_rad)
    u, v = k_along * length_m / 2, k_across * beam_m / 2
    at = _phase(field, np.array(east_m), np.array(north_m), t_s)
    amplitude = np.array([w.amplitude_m for w in field])
    # 1, x, y are orthogonal over a hull centred here, so each wave's slope separates.
    rise_along = -amplitude * np.sin(at) * k_along * _lever(u) * np.sinc(v / math.pi)
    rise_across = -amplitude * np.sin(at) * k_across * _lever(v) * np.sinc(u / math.pi)
    return math.atan(float(rise_along.sum())), math.atan(float(rise_across.sum()))


def _lever(u: np.ndarray) -> np.ndarray:
    """3 (sin u - u cos u) / u^3; its series near 0, where the formula cancels."""
    small = np.abs(u) < 1e-3
    safe = np.where(small, 1.0, u)
    return np.where(
        small, 1 - u**2 / 10, 3 * (np.sin(safe) - safe * np.cos(safe)) / safe**3
    )


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


def fade_footprints_m(
    wavelength_m: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray]:
    """The footprints at which a wave is gone and whole."""
    low, high = FADE_FOOTPRINTS
    return np.asarray(wavelength_m) / low, np.asarray(wavelength_m) / high


def visibility(wavelength_m: np.ndarray, footprint_m: float) -> np.ndarray:
    """How much of a wave a pixel of `footprint_m` draws: 1 whole, 0 left to the
    roughness. Smoothstep in footprint squared, as the shader's Map Range."""
    gone, whole = fade_footprints_m(wavelength_m)
    x = np.clip((footprint_m**2 - gone**2) / (whole**2 - gone**2), 0, 1)
    return x * x * (3 - 2 * x)


def filtered(field: tuple[Wave, ...], footprint_m: float) -> tuple[Wave, ...]:
    """The waves a pixel of `footprint_m` draws, each scaled by its visibility."""
    shown = visibility(np.array([2 * math.pi / w.k_rad_m for w in field]), footprint_m)
    return tuple(
        replace(w, amplitude_m=w.amplitude_m * v)
        for w, v in zip(field, shown.tolist(), strict=True)
    )


def slick_survivors(wind_speed_mps: float, wind: tuple[Wave, ...]) -> tuple[Wave, ...]:
    """The longest waves that fit in Cox & Munk's slick variance: what a slick leaves
    of `wind`, damping the shortest first."""
    budget = cox_munk_slick_slope(wind_speed_mps) ** 2
    left: list[Wave] = []
    # A wavelength goes at every heading at once.
    for k in sorted({w.k_rad_m for w in wind}):
        same = tuple(w for w in wind if w.k_rad_m == k)
        budget -= slope_variance(same)
        if budget < 0.0:
            break
        left.extend(same)
    return tuple(left)


def unresolved_slope_variance(
    wind_speed_mps: float,
    wind: tuple[Wave, ...],
    swell: tuple[Wave, ...],
    footprint_m: float,
    total_slope: Callable[[float], float] = cox_munk_slope,
) -> float:
    """Cox & Munk's is the wind sea's variance; a swell adds its own undrawn part.

    A sea short of fetch keeps Cox & Munk's open-ocean total, so the lobe takes the
    slope of its missing long waves.
    """
    wind_left = total_slope(wind_speed_mps) ** 2 - slope_variance(
        filtered(wind, footprint_m)
    )
    swell_left = slope_variance(swell) - slope_variance(filtered(swell, footprint_m))
    return max(wind_left, 0.0) + swell_left


def unresolved_acceleration_variance(
    wind: tuple[Wave, ...], footprint_m: float
) -> float:
    """Over g^2."""
    return slope_variance(wind) - slope_variance(filtered(wind, footprint_m))


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
    drawn = downward_acceleration_g(filtered(wind, footprint_m), east_m, north_m, t_s)
    sigma = max(math.sqrt(unresolved_acceleration_variance(wind, footprint_m)), 1e-6)
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
