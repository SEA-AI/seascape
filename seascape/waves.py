"""The sea surface as numbers: the earth's curve under it and the waves on it.

Sources
-------
Dominant wavelength: Pierson & Moskowitz, "A proposed spectral form for fully developed
wind seas based on the similarity theory of S. A. Kitaigorodskii", Journal of
Geophysical Research 69(24) 5181, 1964 (doi:10.1029/JZ069i024p05181).

Slope variance: Cox & Munk, "Measurement of the roughness of the sea surface from
photographs of the sun's glitter", JOSA 44(11) 838, 1954 (doi:10.1364/JOSA.44.000838),
clean-sea fit, equation 13.

Slope spectrum: Phillips, "The equilibrium range in the spectrum of wind-generated
waves", Journal of Fluid Mechanics 4(4) 426, 1958 (doi:10.1017/S0022112058000550).

Microfacet lobe: Walter, Marschner, Li & Torrance, "Microfacet models for refraction
through rough surfaces", EGSR 2007 (doi:10.2312/EGWR/EGSR07/195-206) for GGX; Burley,
"Physically-based shading at Disney", SIGGRAPH 2012 course notes, for the alpha =
roughness^2 convention Cycles follows.

Dispersion: Lamb, "Hydrodynamics", 6th ed., Cambridge University Press 1932, chapter
IX; deep water, omega^2 = g k.
"""

import math

GRAVITY_MS2 = 9.81

# Waves, end to end. Each step is a published relation or follows from one:
#
#   wavelength      2 pi U^2 / (0.877^2 g)          Pierson-Moskowitz 1964
#   period          sqrt(2 pi lam / g)              deep-water dispersion, Lamb
#   total slope     sqrt(0.003 + 0.00512 U)         Cox & Munk 1954, eq. 13
#   resolved share  sqrt(octaves / log2(lam/1.7cm)) Phillips 1958 equilibrium range
#   bump relief     resolved share x slope x lam    over the noise transfer below
#   unresolved      sqrt(total^2 - resolved^2)      variances subtract
#   emissivity      Fresnel over unresolved slopes  Masuda 1988, in lwir.py
#   lobe roughness  sqrt(sqrt(2) x unresolved)      GGX alpha = roughness^2

SLOPE_VARIANCE_INTERCEPT = 0.003
SLOPE_VARIANCE_PER_MPS = 0.00512
PM_PEAK = 0.877
MIN_WAVELENGTH_M = 1.0

# 2 pi sqrt(gamma / rho g) = 1.73 cm at gamma = 0.074 N/m: the wavelength of minimum
# phase speed, where surface tension takes over from gravity: the bottom of the slope
# spectrum.
CAPILLARY_WAVELENGTH_M = 0.0173

# Blender's Detail input, which is octaves *beyond* the first.
NOISE_DETAIL = 4.0

# Amplitude ratio between octaves. Slope goes as amplitude x wavenumber and wavenumber
# doubles each octave, so 0.5 is the ratio that puts equal slope variance in each --
# which is what the Phillips equilibrium range says a wind sea does.
NOISE_ROUGHNESS = 0.5

# RMS gradient of the noise's Fac per noise unit, at 2 cm sampling: finer sampling
# finds more.
NOISE_SLOPE_PER_UNIT = 0.55
NOISE_SLOPE_PER_UNIT_4D = 0.48

# Mean radius, IUGG.
EARTH_RADIUS_M = 6_371_000.0


def wave_length_m(wind_speed_mps: float) -> float:
    """Dominant wavelength of a fully developed sea, Pierson-Moskowitz."""
    length = 2 * math.pi * wind_speed_mps**2 / (PM_PEAK**2 * GRAVITY_MS2)
    return max(length, MIN_WAVELENGTH_M)


def wave_period_s(wind_speed_mps: float) -> float:
    return math.sqrt(2 * math.pi * wave_length_m(wind_speed_mps) / GRAVITY_MS2)


def wave_slope(wind_speed_mps: float) -> float:
    """Total RMS surface slope, Cox & Munk. Dimensionless, a tangent."""
    return math.sqrt(SLOPE_VARIANCE_INTERCEPT + SLOPE_VARIANCE_PER_MPS * wind_speed_mps)


def resolved_slope_fraction(wind_speed_mps: float) -> float:
    """Fraction of the RMS slope the noise field can carry, over its octaves.

    Cox & Munk measured the whole spectrum down to capillaries; the noise stops a few
    octaves below the dominant wave. In the Phillips equilibrium range the slope
    spectrum goes as 1/k, so mean-square slope accumulates equally per octave and the
    captured share is a ratio of logs rather than an integral, square-rooted because
    this is slope and that was variance.
    """
    octaves = math.log(wave_length_m(wind_speed_mps) / CAPILLARY_WAVELENGTH_M, 2.0)
    return math.sqrt(min((NOISE_DETAIL + 1.0) / octaves, 1.0))


def bump_slope(wind_speed_mps: float) -> float:
    """The part of that slope the noise field can carry."""
    return resolved_slope_fraction(wind_speed_mps) * wave_slope(wind_speed_mps)


def unresolved_slope(wind_speed_mps: float) -> float:
    """RMS slope the bump cannot carry, left for the shading to account for.

    Variances add, so this is a difference of squares rather than of slopes.
    """
    u = wind_speed_mps
    return math.sqrt(max(wave_slope(u) ** 2 - bump_slope(u) ** 2, 0.0))


def specular_roughness(wind_speed_mps: float) -> float:
    """Blender roughness for a reflection lobe matching the unresolved slope.

    Cycles' GGX takes alpha = roughness^2, and a Gaussian slope of sigma maps to
    alpha = sqrt(2) sigma.
    """
    return math.sqrt(min(math.sqrt(2.0) * unresolved_slope(wind_speed_mps), 1.0))


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
