"""The waves on the sea, as numbers."""

# Waves, end to end. Each step is a published relation or follows from one:
#
#   wind            U10 ln(z / z0) / ln(10 / z0)        DNV-RP-C205 2010
#   spectrum        k^-3 (B_long + B_short)             Elfouhaily et al. 1997
#   peak            kp = g Omega^2 / U10^2              deep-water dispersion
#   wave age        Omega 0.84, more at a short fetch   Elfouhaily et al. 1997, eq. 37
#   direction       1 + Delta(k) cos 2 theta, downwind  Elfouhaily et al. 1997
#   wavenumber      k = w^2 / g                         deep-water dispersion, Lamb
#   total slope     sqrt(0.003 + 0.00512 U(12.5 m))     Cox & Munk 1954, eq. 13
#   per axis        up 3.16e-3 U, across 3e-3+1.92e-3 U Cox & Munk 1954
#   held            scaled to it where it overshoots    Cox & Munk 1954
#   drawn variance  sum of v^2 a^2 k^2 / 2              v = visibility
#   unresolved      total^2 - drawn, + undrawn swell    Bruneton 2010, per pixel
#   whitecaps       tall crests' a k cos(phase) > thresh Snyder & Kennedy 1983
#   foam            youngest break, area exp(-age/3.85 s) Monahan 1971
#   foam fade       e-fold where the life's mean is 0.22 Koepke 1984
#   glitter         one specular point per 1 / E|det H| Longuet-Higgins 1960
#   groups          footprint's unresolved slope^2 varies Isserlis 1918
#   gust            u / U, sigma 1 / ln(10 / z0)        DNV-RP-C205
#   gust field      (k0^2 + k^2)^(-4/3), L of Kaimal's  von Karman 1948, IEC 61400-1
#   gusty slope     + u / U dCM^2 / d ln U, undrawn     Cox & Munk, Plant 1982
#   slick slope     sqrt(0.008 + 0.00156 U(12.5 m))     Cox & Munk 1954
#   windrows        4.8 s x U apart, 3-10x as long      Leibovich 1983, Thorpe 2004
#   windrow cutoff  none below 3 m/s                    Leibovich 1983

from seascape.physics.waves.foam import (
    FOAM_CELLS_PER_WAVE,
    FOAM_VISIBLE_COVER,
    THIN_FOAM_SHARE,
    WHITECAP_AREA_EFOLD_S,
    WHITECAP_EFFECTIVE_REFLECTANCE,
    FoamGrid,
    foam_drift_mps,
    foam_lattice_field,
    foam_memory,
    foam_memory_thresholds_g,
    whitecap_fraction,
)
from seascape.physics.waves.sea import (
    WIND_HEADINGS,
    snap_error,
    spreading_delta,
    swell,
    wind_sea,
)
from seascape.physics.waves.slopes import (
    COX_MUNK_WIND_HEIGHT_M,
    SLICK_VARIANCE_INTERCEPT,
    SLICK_VARIANCE_PER_MPS,
    SPREAD_ASPECT,
    cox_munk_slick_slope,
    cox_munk_slick_variances,
    cox_munk_slope,
    cox_munk_variances,
    drawn_slope_variances,
    filtered,
    mss_spread,
    resolvable,
    slick_survivors,
    slope_variance,
    specular_cell_m2,
    twinkle_hz,
    unresolved_slope_variances,
    visibility,
)
from seascape.physics.waves.spectrum import (
    CAPILLARY_PEAK_RAD_M,
    CAPILLARY_WAVELENGTH_M,
    FREQUENCIES,
    Wave,
    curvature_spectrum,
    friction_velocity_mps,
    peak_omega_rad_s,
    wind_at_m,
)
from seascape.physics.waves.surface import (
    attitude,
    gerstner_hit,
    gerstner_normal,
    gerstner_offset_m,
    gerstner_rest_m,
    heave_m,
    height_m,
    slope,
    tallest,
)
from seascape.physics.waves.wind import (
    BEAUFORT_7_MPS,
    BEAUFORT_8_MPS,
    FINE_WINDROW_SPACING_M,
    GUST_LENGTH_M,
    SLICK_DRIFT,
    WINDROW_ASPECT,
    WINDROW_MIN_WIND_MPS,
    WINDROW_SPACING_S,
    gaussian_ranks,
    gust_slope_variance,
    turbulence_intensity,
    von_karman_field,
)

__all__ = [
    "BEAUFORT_7_MPS",
    "BEAUFORT_8_MPS",
    "CAPILLARY_PEAK_RAD_M",
    "CAPILLARY_WAVELENGTH_M",
    "COX_MUNK_WIND_HEIGHT_M",
    "FINE_WINDROW_SPACING_M",
    "FOAM_CELLS_PER_WAVE",
    "FOAM_VISIBLE_COVER",
    "FREQUENCIES",
    "GUST_LENGTH_M",
    "SLICK_DRIFT",
    "SLICK_VARIANCE_INTERCEPT",
    "SLICK_VARIANCE_PER_MPS",
    "SPREAD_ASPECT",
    "THIN_FOAM_SHARE",
    "WHITECAP_AREA_EFOLD_S",
    "WHITECAP_EFFECTIVE_REFLECTANCE",
    "WINDROW_ASPECT",
    "WINDROW_MIN_WIND_MPS",
    "WINDROW_SPACING_S",
    "WIND_HEADINGS",
    "FoamGrid",
    "Wave",
    "attitude",
    "cox_munk_slick_slope",
    "cox_munk_slick_variances",
    "cox_munk_slope",
    "cox_munk_variances",
    "curvature_spectrum",
    "drawn_slope_variances",
    "filtered",
    "foam_drift_mps",
    "foam_lattice_field",
    "foam_memory",
    "foam_memory_thresholds_g",
    "friction_velocity_mps",
    "gaussian_ranks",
    "gerstner_hit",
    "gerstner_normal",
    "gerstner_offset_m",
    "gerstner_rest_m",
    "gust_slope_variance",
    "heave_m",
    "height_m",
    "mss_spread",
    "peak_omega_rad_s",
    "resolvable",
    "slick_survivors",
    "slope",
    "slope_variance",
    "snap_error",
    "specular_cell_m2",
    "spreading_delta",
    "swell",
    "tallest",
    "turbulence_intensity",
    "twinkle_hz",
    "unresolved_slope_variances",
    "visibility",
    "von_karman_field",
    "whitecap_fraction",
    "wind_at_m",
    "wind_sea",
]
