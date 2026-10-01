"""A cloud layer as numbers: where it lies, how thick it is, what it lets through.

Sources
-------
Field: Davis, Marshak, Wiscombe & Cahalan, "Scale invariance of liquid water
distributions in marine stratocumulus. Part I", JAS 53(11) 1538, 1996
(doi:10.1175/1520-0469(1996)053<1538:SIOLWD>2.0.CO;2): liquid water falls as k^-1.56
from 20 m to 20 km, near Kolmogorov's -5/3, hence `waves.von_karman_field`. Wood &
Hartmann, "Spatial variability of liquid water path in marine low cloud: the importance
of mesoscale cellular convection", J. Climate 19(9) 1748, 2006 (doi:10.1175/JCLI3702.1),
section 4b: the spectrum peaks at 35 boundary-layer depths. Minnis, "Viewing zenith
angle dependence of cloudiness determined from coincident GOES East and GOES West
data", JGR 94(D2) 2303, 1989 (doi:10.1029/JD094iD02p02303), and Zhao & Di Girolamo, "A
cloud fraction versus view angle technique for automatic in-scene evaluation of the
MISR cloud mask", JAM 43(6) 860, 2004
(doi:10.1175/1520-0450(2004)043<0860:ACFVVA>2.0.CO;2): seen aslant, cover grows, as
cloud sides hide the gaps between.

Thickness to optical depth: Brenguier et al., "Radiative properties of boundary layer
clouds: droplet effective radius versus number concentration", JAS 57(6) 803, 2000
(doi:10.1175/1520-0469(2000)057<0803:RPOBLC>2.0.CO;2): an adiabatic cloud's tau goes as
its thickness^(5/3). Stephens, "Radiation profiles in extended water clouds. II", JAS
35(11) 2123, 1978 (doi:10.1175/1520-0469(1978)035<2123:RPIEWC>2.0.CO;2): tau = 3 LWP /
(2 rho r_e). Han, Rossow & Lacis, "Near-global survey of effective droplet radii in
liquid water clouds using ISCCP data", J. Climate 7(4) 465, 1994
(doi:10.1175/1520-0442(1994)007<0465:NGSOED>2.0.CO;2), for the mean marine LWP, as
Kokhanovsky (below) quotes it. Ackerman et al., "Cloud detection with MODIS. Part II:
validation", JTECH 25(7) 1073, 2008 (doi:10.1175/2007JTECHA1053.1): cloud is seen from
tau 0.4.

Light: Davis & Marshak, "Solar radiation transport in the cloudy atmosphere: a 3D
perspective on observations and climate impacts", Rep. Prog. Phys. 73(2) 026801, 2010
(doi:10.1088/0034-4885/73/2/026801), eqs 61-62: diffusion's transmission 1 / (1 + (1 -
g) tau / 2 chi), chi = 2/3. Kokhanovsky, "Optical properties of terrestrial clouds",
Earth-Sci. Rev. 64(3-4) 189, 2004 (doi:10.1016/S0012-8252(03)00042-4), section 3.1.4:
g = 0.85 for droplets. Minnis, Garber, Young, Arduini & Takano, "Parameterizations of
reflectance and effective emittance for satellite remote sensing of cloud properties",
JAS 55(22) 3313, 1998 (doi:10.1175/1520-0469(1998)055<3313:PORAEE>2.0.CO;2), tables 7-8:
at 10.8 um a droplet absorbs 0.38 of its visible extinction at r_e 8 um and 0.46 at 12
um.

Height and motion: Eastman, Warren & Hahn, "Variations in cloud cover and cloud types
over the ocean from surface observations, 1954-2008", J. Climate 24(22) 5914, 2011
(doi:10.1175/2011JCLI3972.1), table 1: stratocumulus over the ocean bases at 600 m.
Wallace & Hobbs, "Atmospheric Science", 2nd ed., Academic Press 2006, eq 3.53: below the
base, a mixed layer cools at the dry adiabat. Hasler, Shenk & Skillman, NASA
X-911-75-302, 1975 (NTRS 19760010653): clouds move with the wind at their base. Hasse &
Wagner, "On the relationship between geostrophic and surface wind at sea", MWR 99(4)
255, 1971 (doi:10.1175/1520-0493(1971)099<0255:OTRBGA>2.3.CO;2): over the German Bight
the surface wind is 0.56 of the geostrophic."""

import math
from statistics import NormalDist

import numpy as np

from seascape import waves

CELLS = 512
# Each tile's spacing; the finer takes the spectrum above the coarser one's Nyquist.
# Where the finer stops is a judgement: a gap that narrow hides behind the layer's sides
# unless seen overhead.
BAND_SPACINGS_M = (400.0, 25.0)
PEAK_PER_DEPTH = 35.0
ASYMMETRY = 0.85
DIFFUSION_CHI = 2 / 3
MEAN_LWP_KG_M2 = 86e-3
# The droplet Kokhanovsky's g is for.
DROPLET_RADIUS_M = 10e-6
WATER_DENSITY_KG_M3 = 1000.0
MEAN_OPTICAL_DEPTH = 3 * MEAN_LWP_KG_M2 / (2 * WATER_DENSITY_KG_M3 * DROPLET_RADIUS_M)
THICKNESS_POWER = 5 / 3
SEEN_OPTICAL_DEPTH = 0.4
# Minnis et al. 1998, linear between their 8 and 12 um droplets at DROPLET_RADIUS_M.
LWIR_ABSORPTION = 0.42
DRY_LAPSE_K_M = 9.8e-3
SURFACE_OVER_GEOSTROPHIC = 0.56
STRATOCUMULUS_BASE_M = 600.0


def integral_scale_m(base_m: float) -> float:
    """The von Karman scale whose spectrum, radially summed, 2 pi k (k0^2 + k^2)^(-4/3),
    peaks where Wood & Hartmann's does: at k = k0 sqrt(3/5).

    ponytail: the base stands in for the boundary layer's depth, which is the base plus
    the cloud. Add the cloud's thickness if the cells read small.
    """
    k0_rad_m = 2 * math.pi / (PEAK_PER_DEPTH * base_m * math.sqrt(3 / 5))
    return math.sqrt(math.pi) * math.gamma(5 / 6) / (math.gamma(1 / 3) * k0_rad_m)


def tiles(rng: np.random.Generator, base_m: float) -> list[np.ndarray]:
    """One spectrum split over `BAND_SPACINGS_M`, each tile scaled to its share, so
    their sum has unit variance."""
    length_m = integral_scale_m(base_m)
    longest_m = [math.inf, *(2 * s for s in BAND_SPACINGS_M[:-1])]
    variances = [
        waves.von_karman_variance(CELLS, s, length_m, longest)
        for s, longest in zip(BAND_SPACINGS_M, longest_m, strict=True)
    ]
    total = sum(variances)
    return [
        math.sqrt(v / total) * waves.von_karman_field(rng, CELLS, s, length_m, longest)
        for s, longest, v in zip(BAND_SPACINGS_M, longest_m, variances, strict=True)
    ]


def optical_depth(cover: float) -> tuple[float, float]:
    """`threshold` and `scale` of tau = scale max(f - threshold, 0)^THICKNESS_POWER on
    a unit Gaussian f: `cover` of it above `SEEN_OPTICAL_DEPTH`, with mean
    `MEAN_OPTICAL_DEPTH` there.

    f - threshold stands for the cloud's thickness, 0 at its edge.
    """
    seen = NormalDist().inv_cdf(1 - cover)
    z = np.linspace(seen, seen + 12.0, 20001)
    density = np.exp(-(z**2) / 2)
    threshold, scale = seen, 0.0
    # A fixed point: the edge sets the mean, the mean sets where tau reaches
    # SEEN_OPTICAL_DEPTH.
    for _ in range(50):
        excess = (z - threshold) ** THICKNESS_POWER
        scale = MEAN_OPTICAL_DEPTH * np.trapezoid(density, z)
        scale /= np.trapezoid(excess * density, z)
        threshold = seen - (SEEN_OPTICAL_DEPTH / scale) ** (1 / THICKNESS_POWER)
    return float(threshold), float(scale)


def base_k(t_air_k: float, base_m: float) -> float:
    return t_air_k - DRY_LAPSE_K_M * base_m


def drift_mps(wind_speed_mps: float) -> float:
    """The wind at the base, taken as geostrophic.

    ponytail: Hasse & Wagner's offset for stability and the wind's veer with height are
    left out, so clouds run straight downwind; add them if clouds must cross the
    surface wind.
    """
    return wind_speed_mps / SURFACE_OVER_GEOSTROPHIC


def reach_m(base_m: float, radius_m: float) -> float:
    """Where a ray along the horizon meets the layer, on an earth of `radius_m`."""
    return math.sqrt(2 * radius_m * base_m + base_m**2)
