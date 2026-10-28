"""The sea's radiating skin, which is not the water a thermometer reads.

An infrared camera sees the top 10 um or so. Fairall et al.'s equation 1 relates it to
the bulk, the mixed layer under it: T_skin = T_bulk + dT_warm - dT_cool. Kelvin.

Sources
-------
Fairall, Bradley, Godfrey, Wick, Edson & Young, "Cool-skin and warm-layer effects on sea
surface temperature", JGR 101(C1) 1295, 1996 (doi:10.1029/95JC03190): the cool skin,
their model's mean over the COARE data; the warm layer, equations 21, 25 and 26, with
Soloviev's three-band absorption and the critical Richardson number they fitted; their
Table 3, a clear COARE day, as its default forcing.

Fairall et al.'s own NOAA-PSL COARE 3.6 code (github.com/NOAA-PSL/COARE-algorithm,
`coare36vnWarm_et.py`) for seawater's density, heat capacity and thermal expansion.
"""

import math

import numpy as np

from seascape.physics.waves import GRAVITY_MS2

# Fairall et al. 1996, abstract: the mean cool skin at local noon and at night. It
# depends on the net heat flux and the wind (their Figure 2), which a scenario does not
# carry, so the means hold at every wind, and the noon one all day.
COOL_SKIN_DAY_K = 0.18
COOL_SKIN_NIGHT_K = 0.30

# Fairall et al. 1996, Table 3: a nearly cloud-free COARE day, averaged from just after
# sunrise to just past noon.
CLEAR_DAY_SOLAR_W_M2 = 710.0
CLEAR_DAY_HOURS = 7.0
# Same table: the net cooling at the interface that day, by 10 m wind.
_CLEAR_DAY_WIND_MPS = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
_CLEAR_DAY_COOLING_W_M2 = (100.0, 110.0, 125.0, 165.0, 190.0, 230.0, 260.0)

# Fairall et al. 1996's estimate of the mean u*a^2 as 0.0013 u^2, a drag coefficient.
DRAG_COEFFICIENT = 0.0013
# Fairall et al. 1996, equation 21, after Price, Weller & Pinkel 1986.
CRITICAL_RICHARDSON = 0.65
# Fairall et al. 1996, equation 26: Soloviev 1982's bands for moderately clean water.
_SOLOVIEV_FRACTIONS = (0.45, 0.27, 0.28)
_SOLOVIEV_LENGTHS_M = (12.8, 0.357, 0.014)

# COARE 3.6, coare36vnWarm_et.py.
WATER_DENSITY_KG_M3 = 1022.0
WATER_HEAT_J_KG_K = 4000.0
# COARE 3.6 accumulates no less stress than this, so a calm warm layer stays finite.
MIN_STRESS_N_M2 = 0.002
# COARE 3.6's `max_pwp`: the deepest it lets a warm layer grow.
MAX_DEPTH_M = 19.0
# The ICAO standard atmosphere's sea-level pressure with dry air's gas constant, for
# the air's density.
SEA_LEVEL_PA = 101325.0
DRY_AIR_J_KG_K = 287.05


def thermal_expansion_per_k(t_k: float) -> float:
    """Seawater's, COARE 3.6's fit in Celsius."""
    return 2.1e-5 * (t_k - 273.15 + 3.2) ** 0.79


def solar_trapped(depth_m: float) -> float:
    """Fairall et al.'s equation 26: the fraction of the net solar a layer `depth_m`
    deep keeps."""
    passed = sum(
        f * g * (1.0 - math.exp(-depth_m / g))
        for f, g in zip(_SOLOVIEV_FRACTIONS, _SOLOVIEV_LENGTHS_M, strict=True)
    )
    return 1.0 - passed / depth_m


def warm_layer(
    solar_w_m2: float,
    cooling_w_m2: float,
    stress_m2_s2: float,
    hours: float,
    t_sea_k: float,
) -> tuple[float, float]:
    """Peak warming of the surface over the mixed layer, K, and the warm layer's depth,
    m, from Fairall et al.'s equations 21 and 25: the warm layer keeps the heat and
    momentum put in since it formed, to the depth where its bulk Richardson number is
    critical.

    `solar_w_m2` is the net solar at the surface, `cooling_w_m2` the net cooling at the
    interface and `stress_m2_s2` the air's u*^2, each averaged over the `hours`. No
    warming where the trapped sun cannot outweigh the cooling at any depth.

    The air's density is taken at the sea's temperature: 10 K off is 3.5 % of the
    stress.
    """
    alpha_g = thermal_expansion_per_k(t_sea_k) * GRAVITY_MS2
    heat = WATER_DENSITY_KG_M3 * WATER_HEAT_J_KG_K
    # The same stress either side of the interface.
    stress_water = stress_m2_s2 * air_density_kg_m3(t_sea_k) / WATER_DENSITY_KG_M3
    seconds = hours * 3600.0
    # Equation 21 with its integrals as means times t. A deeper layer traps more sun
    # and so asks for less depth: one depth balances, bisected on a log scale between
    # a millimetre and a kilometre.
    low, high = 1e-3, 1e3
    if solar_trapped(high) * solar_w_m2 <= cooling_w_m2:
        return 0.0, math.inf

    wanted = math.sqrt(2 * CRITICAL_RICHARDSON * seconds) * stress_water

    def excess(depth_m: float) -> float:
        net = solar_trapped(depth_m) * solar_w_m2 - cooling_w_m2
        if net <= 0.0:
            return -math.inf
        return depth_m - wanted / math.sqrt(alpha_g * net / heat)

    for _ in range(100):
        mid = math.sqrt(low * high)
        if excess(mid) > 0.0:
            high = mid
        else:
            low = mid
    # The deep end, where the trapped sun is sure to outweigh the cooling.
    depth_m = min(high, MAX_DEPTH_M)
    net = solar_trapped(depth_m) * solar_w_m2 - cooling_w_m2
    # Equation 25.
    rise = (
        math.sqrt(2 * alpha_g * seconds / CRITICAL_RICHARDSON)
        * (net / heat) ** 1.5
        / stress_water
    )
    return rise, depth_m


def clear_day_cooling_w_m2(wind_speed_mps: float) -> float:
    """Fairall et al.'s Table 3, held past its ends."""
    return float(
        np.interp(wind_speed_mps, _CLEAR_DAY_WIND_MPS, _CLEAR_DAY_COOLING_W_M2)
    )


def air_density_kg_m3(t_k: float) -> float:
    return SEA_LEVEL_PA / (DRY_AIR_J_KG_K * t_k)


def stress_m2_s2(wind_speed_mps: float, t_k: float) -> float:
    """The air's u*^2 from the 10 m wind, as Fairall et al. estimated it, no less than
    COARE 3.6's floor."""
    floor = MIN_STRESS_N_M2 / air_density_kg_m3(t_k)
    return max(DRAG_COEFFICIENT * wind_speed_mps**2, floor)


def skin_k(t_sea_k: float, warm_layer_k: float, sun_up: bool) -> float:
    """The radiating skin over a bulk at `t_sea_k`: its warm layer's rise, less the cool
    skin, which the sun thins."""
    cool = COOL_SKIN_DAY_K if sun_up else COOL_SKIN_NIGHT_K
    return t_sea_k + warm_layer_k - cool
