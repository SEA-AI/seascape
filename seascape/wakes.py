"""A hull's wake as numbers: where its Kelvin arms lie, how tall, how long it foams.

Sources
-------
Kelvin: the wedge's half-angle arcsin(1/3), and the stationary-phase waves inside it
whose cusp runs at cos^2 = 2/3 and falls as range^(-1/3), NIST DLMF section 36.13
(https://dlmf.nist.gov/36.13). Rabaud & Moisy, "Ship wakes: Kelvin or Mach angle?", PRL
110(21) 214503, 2013 (doi:10.1103/PhysRevLett.110.214503), eq. 3b: past Froude
sqrt(3 / 4 pi) the wake narrows to its amplitude peak. Darmon, Benzaquen & Raphael,
"Kelvin wake pattern at large Froude numbers", JFM 738 R3, 2014
(doi:10.1017/jfm.2013.607): the peak narrows, the wedge's edge stays.

Height: Kriebel & Seelig 2005, as Scully & McCartney give it in ERDC/CHL CHETN-IX-46,
2017: gH / U^2 = beta (F - 0.1)^2 (y / L)^(-1/3) in deep water, with
beta = 1 + 8 tanh^3(0.45 (L / Le - 2)). Ausenco, "Fraser River vessel wake assessment",
2015, table 4: the entrance Le is a third of the length on every hull it lists.
Macfarlane, PhD thesis, University of Tasmania, 2012, p. 56: wake height peaks near
F = 0.55 and falls past it. Michell, "The highest waves in water", Phil. Mag. 36(222)
430, 1893 (doi:10.1080/14786449308620499): no wave is steeper than height / length 1/7.

Foam: Callaghan et al., JGR 117(C9), 2012 (doi:10.1029/2012JC008147): whitecap foam
decays with an area-weighted e-fold of 1.4-4.8 s. Whitlock, Bartlett & Gurganus, "Sea
foam reflectance and influence on optimum wavelength for remote sensing of ocean
aerosols", GRL 9(6) 719, 1982 (doi:10.1029/GL009i006p00719): fresh foam sends back
0.55, as Koepke, "Effective reflectance of oceanic whitecaps", Appl. Opt. 23(11) 1816,
1984 (doi:10.1364/AO.23.001816), quotes it before averaging over the foam's ageing.

Turbulent wake: Milgram et al., JGR 98(C4) 7103, 1993 (doi:10.1029/92JC02612): short
waves stay damped in a wake an hour old. Trevorrow, Vagle & Farmer, JASA 95(4) 1922,
1994 (doi:10.1121/1.408706): its bubbles last 7.5 minutes. Gatebe et al., GRL 38(17),
2011 (doi:10.1029/2011GL048819): a wake reflects twice the sunlight of the sea beside
it.
"""

import math
from dataclasses import dataclass

from seascape.waves import GRAVITY_MS2

# DLMF 36.13.
KELVIN_HALF_ANGLE_RAD = math.asin(1 / 3)
# Rabaud & Moisy eq. 3b.
NARROWING_FROUDE = math.sqrt(3 / (4 * math.pi))
# Kriebel & Seelig: none below this Froude, by length.
WAVELESS_FROUDE = 0.1
# Ausenco 2015, table 4.
ENTRANCE_PER_LENGTH = 1 / 3
# Macfarlane 2012.
HUMP_FROUDE = 0.55
# Michell 1893.
MAX_STEEPNESS = 1 / 7
# Whitlock et al. 1982.
FRESH_FOAM_REFLECTANCE = 0.55
# A judgement: the slowest of Callaghan et al.'s e-folds, as a hull churns its foam
# deeper than a whitecap does.
FOAM_EFOLD_S = 4.8
# A judgement: Trevorrow et al.'s 7.5 minutes taken as an e-fold.
BUBBLE_EFOLD_S = 7.5 * 60
# Gatebe et al. 2011: twice the sea's reflectance.
BUBBLE_GAIN = 1.0
# A judgement: arms adding under 1 % to the sea's slope variance go unseen.
SEEN_SLOPE_VARIANCE = 0.01


@dataclass(frozen=True)
class Wake:
    """A hull under way: along `heading_rad` from `start_m` (east, north), or round the
    origin at `orbit_m`, clockwise from `start_bearing_rad`, `count` hulls a lap."""

    speed_mps: float
    length_m: float
    beam_m: float
    start_m: tuple[float, float] = (0.0, 0.0)
    heading_rad: float = 0.0
    orbit_m: float | None = None
    start_bearing_rad: float = 0.0
    lap_s: float = 0.0
    count: int = 1

    @property
    def froude(self) -> float:
        return self.speed_mps / math.sqrt(GRAVITY_MS2 * self.length_m)

    @property
    def height_scale_m(self) -> float:
        """Kriebel & Seelig's H at y = L, held at its hump-Froude value past it: a
        judgement, as Macfarlane's heights fall there."""
        froude = min(self.froude, HUMP_FROUDE)
        speed_mps = froude * math.sqrt(GRAVITY_MS2 * self.length_m)
        beta = 1 + 8 * math.tanh(0.45 * (1 / ENTRANCE_PER_LENGTH - 2)) ** 3
        excess = max(froude - WAVELESS_FROUDE, 0.0)
        return beta * speed_mps**2 / GRAVITY_MS2 * excess**2

    @property
    def steepest_arm(self) -> float:
        """The divergent wave's slope where it is tallest: on the cusp, half a beam
        out, a judgement."""
        k_rad_m = GRAVITY_MS2 / self.speed_mps**2 / (2 / 3)
        side = self.beam_m / 2 / self.length_m
        amplitude_m = self.height_scale_m / 2 * side ** (-1 / 3)
        return min(amplitude_m * k_rad_m, math.pi * MAX_STEEPNESS)

    @property
    def peak_angle_rad(self) -> float:
        """Rabaud & Moisy eq. 3b; Kelvin's below their critical Froude."""
        if self.froude <= NARROWING_FROUDE:
            return KELVIN_HALF_ANGLE_RAD
        fr_sq = self.froude**2
        return math.atan(math.sqrt(2 * math.pi * fr_sq - 1) / (4 * math.pi * fr_sq - 1))
