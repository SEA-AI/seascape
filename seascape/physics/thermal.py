"""Surface temperature of a vessel's thermal zones, from a heat balance at the instant.

Each zone is a material with its own LWIR emissivity, solar absorptance and coupling to
the space behind it. A facet of a zone settles where what it absorbs, sun and sky and
sea, leaves as its own emission, as convection to the air and as conduction to the
interior.

The scenario has no history, only the sun where it stands, so the balance is steady.
Limitation: no thermal lag; a hull keeps the morning's cold past sunrise and the
afternoon's heat past sunset, which a transient balance over the day would carry.

Broadband throughout: the balance is energy, so the sky enters as its total longwave
flux, never the 8-14 um band radiance `lwir` gives the sensor.

Sources
-------
Ship zones and their temperatures: Thompson, Vaitekunas & Birk, "IR Signature
Suppression of Modern Naval Ships", ASNE 21st Century Combatant Technology Symposium,
1998 (davis-eng.com/docs/papers/irss_paper.pdf).

Emissivities and solar absorptances, where not cited otherwise: the compilation of total
normal emissivities in Neutrium, "Total Normal Emissivities of Selected Materials",
2018, after Hottel's tables as carried by the heat-transfer handbooks it lists. A row's
name is given beside each number taken from it.

Direct beam: Meinel & Meinel, Applied Solar Energy, Addison-Wesley, 1976, through air
mass after Kasten & Young, "Revised optical air mass tables and approximation formula",
Applied Optics 28(22) 4735, 1989 (doi:10.1364/AO.28.004735).

Clear-sky longwave: Swinbank, "Long-wave radiation from clear skies", Q. J. R.
Meteorol. Soc. 89(381) 339, 1963 (doi:10.1002/qj.49708938105).

Wind convection: McAdams, Heat Transmission, 3rd ed., McGraw-Hill, 1954, as Duffie &
Beckman, Solar Engineering of Thermal Processes, carry it.

Surface resistances: ISO 6946, Building components and building elements, thermal
resistance and thermal transmittance, Annex A.
"""

import math
from dataclasses import dataclass, field, replace
from typing import Literal, get_args

import numpy as np
import numpy.typing as npt

from seascape.physics.atmosphere import ATMOSPHERE, Atmosphere
from seascape.physics.lwir import cloud_window_deficit_w_m2

type FloatArray = npt.NDArray[np.float64]

# What each zone covers; its numbers are in ZONES.
type ZoneName = Literal[
    "paint_hull",  # a painted ship's shell plating, topsides and bottom
    "paint_superstructure",  # painted house, bridge, masts, cranes, buoy bodies
    "deck",  # walking surfaces: steel deck, teak, non-slip
    "glass",  # windows, portholes, windscreens, lamp lenses
    "bare_metal",  # unpainted steel, stainless, aluminium, chrome
    "rubber",  # inflatable tubes, fenders, tyres, hoses
    "fabric",  # canvas, biminis, cushions, sails, rope
    "funnel",  # a ship's stack casing
    "exhaust",  # the hot outlet in or atop a funnel, an exhaust pipe
    "engine_cowl",  # an outboard engine's cover
    "wood_composite",  # timber, and a small craft's moulded GRP or carbon hull
]
ZONE_NAMES: tuple[ZoneName, ...] = get_args(ZoneName.__value__)
type Interior = Literal["none", "cabin", "engine_room", "exhaust"]

# CODATA 2018, exact from the defining constants.
STEFAN_BOLTZMANN_W_M2K4 = 5.670374419e-8

# Meinel & Meinel 1976: I = 1.353 kW m^-2 * 0.7 ** (AM ** 0.678), on a plane facing the
# sun. 1.353 is their solar constant, kept with the fit it belongs to.
_MEINEL_SOLAR_CONSTANT_W_M2 = 1353.0
# Clear-sky diffuse is about a tenth of the direct beam (PVEducation, "Air Mass",
# after Meinel & Meinel), spread over the sky as Liu & Jordan's isotropic sky.
_DIFFUSE_FRACTION = 0.1

# Swinbank 1963: R = 5.31e-14 T^6 mW cm^-2, so 5.31e-13 W m^-2 K^-6, clear sky.
_SWINBANK_W_M2K6 = 5.31e-13

# Water, typical, 0-100 C: 0.95-0.963 (Neutrium compilation). The sea seen below a
# facet's horizon.
SEA_EMISSIVITY = 0.96

# ISO 6946 Annex A: interior surface resistance for horizontal heat flow, within 30
# degrees of it. A wall's inner film; its outer one is this module's convection and
# radiation.
_R_SI_M2K_W = 0.13
# Judgement: a glass fibre blanket, ~0.04 W m^-1 K^-1, Incropera & DeWitt Table A.3
# (0.035-0.046 by density) as recalled, unchecked.
_GLASS_WOOL_W_MK = 0.04
# Thompson et al. 1998: 25 mm (1") of glass wool on a heated compartment's bulkheads.
_INSULATION_M = 0.025
# Judgement: soda-lime glass, 1.0 W m^-1 K^-1, EN 673's value as recalled, unchecked.
_GLASS_W_MK = 1.0
# Judgement: a 6 mm wheelhouse pane.
_PANE_M = 0.006

# Uninsulated steel plate: steel's resistance, ~1e-4 m^2 K W^-1 for 6 mm, is lost
# beside the inner film.
_U_BARE_PLATE_W_M2K = 1.0 / _R_SI_M2K_W
_U_INSULATED_W_M2K = 1.0 / (_R_SI_M2K_W + _INSULATION_M / _GLASS_WOOL_W_MK)
_U_PANE_W_M2K = 1.0 / (_R_SI_M2K_W + _PANE_M / _GLASS_W_MK)

# Incropera & DeWitt Table A.1, plain carbon steel at 300 K: rho 7854 kg m^-3, c_p 434
# J kg^-1 K^-1.
_STEEL_HEAT_CAPACITY_J_M3K = 7854.0 * 434.0
# Judgement: an uptake's or a funnel's wall. Biot h d / k stays below 0.01 for steel,
# so the wall cools as one temperature.
_MACHINERY_WALL_M = 0.01

# Paint over steel. Oil paints, 100 C: 0.92-0.96; black paint 0.97, white paint 0.93
# (Neutrium compilation). Colour hardly moves it: what absorbs the sun is pigment, what
# emits at 10 um is the binder. Judgement: 0.94, mid-range of the oil paints. Navy
# grey's own figure is unsourced: Vaitekunas (SPIE 7662, 2010, doi:10.1117/12.852131)
# gives none in its abstract.
_PAINT_EMISSIVITY = 0.94
# White paint, 20 C: alpha 0.14; black paint 0.97 (Neutrium compilation).
PAINT_ABSORPTANCE = (0.14, 0.97)
# Judgement: no measured navy grey absorptance is sourced. Between the
# compilation's white and black, so the default stays within what is verified.
GREY_PAINT_ABSORPTANCE = 0.65


@dataclass(frozen=True)
class Zone:
    """A material on a vessel and what lies behind it.

    `absorptance` is the one the balance uses; `absorptance_range` is what the material
    spans, white to dark where colour matters. `replace` it to paint a hull.
    """

    emissivity: float  # broadband, LWIR
    absorptance_range: tuple[float, float]  # solar
    absorptance: float
    interior: Interior
    # Interior air to outer skin. inf pins the skin to the interior: exhaust metal.
    u_w_m2k: float

    def __post_init__(self) -> None:
        low, high = self.absorptance_range
        if not 0.0 <= low <= self.absorptance <= high <= 1.0:
            raise ValueError(
                f"absorptance {self.absorptance} outside {self.absorptance_range}"
            )
        if not 0.0 <= self.emissivity <= 1.0:
            raise ValueError(f"emissivity {self.emissivity} outside [0, 1]")
        if self.interior == "none" and self.u_w_m2k != 0.0:
            raise ValueError("a zone with no interior conducts to nothing")


ZONES: dict[ZoneName, Zone] = {
    # Judgement: a hull side over tanks, holds and voids. Hull over a machinery space is
    # Thompson et al.'s warm band; give it the engine room.
    "paint_hull": Zone(
        _PAINT_EMISSIVITY, PAINT_ABSORPTANCE, GREY_PAINT_ABSORPTANCE, "none", 0.0
    ),
    "paint_superstructure": Zone(
        _PAINT_EMISSIVITY,
        PAINT_ABSORPTANCE,
        GREY_PAINT_ABSORPTANCE,
        "cabin",
        _U_INSULATED_W_M2K,
    ),
    # Deck paint runs dark: grey to black. Judgement: a weather deck over voids, as the
    # hull; a deck over a cabin is the superstructure's case.
    "deck": Zone(_PAINT_EMISSIVITY, (GREY_PAINT_ABSORPTANCE, 0.97), 0.8, "none", 0.0),
    # Rubin, "Optical constants and bulk optical properties of soda lime silica glasses
    # for windows", Sol. Energy Mater. 12(4) 275, 1985: hemispherical total 0.837 at
    # 20 C. Judgement: absorptance, clear float to tinted; no source.
    "glass": Zone(0.837, (0.1, 0.5), 0.15, "cabin", _U_PANE_W_M2K),
    # Stainless steel, typical dull, 20 C: eps 0.21, alpha 0.50; polished alpha 0.37
    # (Neutrium compilation). Rails, stanchions, fittings.
    "bare_metal": Zone(0.21, (0.37, 0.50), 0.50, "none", 0.0),
    # Rubber, soft, grey, rough, 25 C: 0.859; hard glossy 0.945 (Neutrium
    # compilation). Fenders and inflatable tubes. Judgement: absorptance grey to black,
    # taken as paint's; no source.
    "rubber": Zone(0.859, (GREY_PAINT_ABSORPTANCE, 0.97), 0.9, "none", 0.0),
    # Judgement: cloth 0.75-0.90 at 300 K, Incropera & DeWitt Table A.11 as recalled,
    # unchecked.
    # Absorptance by colour, taken as paint's. Limitation: a canopy has air on both
    # sides and convects from each; the balance gives it one.
    "fabric": Zone(0.85, PAINT_ABSORPTANCE, GREY_PAINT_ABSORPTANCE, "none", 0.0),
    # Thompson et al. 1998: funnel walls without insulation, heated by the engine
    # room's ventilation and the uptakes inside.
    "funnel": Zone(
        _PAINT_EMISSIVITY,
        PAINT_ABSORPTANCE,
        GREY_PAINT_ABSORPTANCE,
        "engine_room",
        _U_BARE_PLATE_W_M2K,
    ),
    # Iron, steel oxidised at 872 K, 199-599 C: 0.79 (Neutrium compilation), the range
    # uptake metal runs at. Judgement: absorptance, dark oxide, no source; at
    # uptake temperature the sun is lost in it.
    "exhaust": Zone(0.79, (0.8, 0.95), 0.9, "exhaust", math.inf),
    # Judgement: gelcoat or painted composite emits as paint. Limitation: the inner
    # film alone; a 5 mm laminate adds a few percent to its resistance.
    "engine_cowl": Zone(
        _PAINT_EMISSIVITY,
        PAINT_ABSORPTANCE,
        GREY_PAINT_ABSORPTANCE,
        "engine_room",
        _U_BARE_PLATE_W_M2K,
    ),
    # Wood, oak, planed, 21 C: 0.895 (Neutrium compilation). Judgement: absorptance,
    # teak and varnish, light to dark; no source.
    "wood_composite": Zone(0.895, (0.4, 0.8), 0.6, "none", 0.0),
}


# Plate, completely rusted, 20 C: 0.685 (Neutrium compilation, as a search quoted it;
# unchecked against the table).
RUST_EMISSIVITY = 0.685
# Lampblack, thin layer on iron plate, 21 C: 0.927 (Neutrium compilation, as a search
# quoted it; unchecked against the table).
SOOT_EMISSIVITY = 0.927
# Judgement: red-brown oxide absorbs most of the sun; soot as black paint; dirt over
# paint as the compilation's grey to black. No source for any.
RUST_ABSORPTANCE = 0.75
SOOT_ABSORPTANCE = PAINT_ABSORPTANCE[1]
GRIME_ABSORPTANCE = 0.8

type Wear = Literal["grime", "rust", "soot"]


def worn(zone: Zone, layer: Wear) -> Zone:
    """`zone` under a layer of grime, which darkens it in the sun, or of rust or soot,
    which replace its surface. What lies behind it stays."""
    if layer == "grime":
        emissivity, absorptance = (
            zone.emissivity,
            max(zone.absorptance, GRIME_ABSORPTANCE),
        )
    elif layer == "rust":
        emissivity, absorptance = RUST_EMISSIVITY, RUST_ABSORPTANCE
    else:
        emissivity, absorptance = SOOT_EMISSIVITY, SOOT_ABSORPTANCE
    return replace(
        zone,
        emissivity=emissivity,
        absorptance=absorptance,
        absorptance_range=(absorptance, absorptance),
    )


# Judgement: one paint scheme covers a hull's sides, its superstructure and its funnel.
PAINTED: frozenset[ZoneName] = frozenset(
    {"paint_hull", "paint_superstructure", "funnel"}
)


@dataclass(frozen=True)
class Environment:
    """The air, sea and sun a facet balances against. Wind is the air's speed past
    the facet."""

    t_air_k: float
    t_sea_k: float
    wind_m_s: float
    sun_elevation_rad: float | None  # None, or at or below the horizon: no sun
    cloud_fraction: float = 0.0
    cloud_base_m: float = 0.0
    atmosphere: Atmosphere = ATMOSPHERE
    # Fog under the clouds: its emissivity overhead, and what it leaves of the beam.
    fog_cover: float = 0.0
    beam_transmittance: float = 1.0
    fog_sunlight_transmittance: float = 1.0
    # Judgement: 22 C, ISO 7547's winter design for accommodation as recalled,
    # unchecked.
    t_cabin_k: float = 295.15
    # Thompson et al. 1998: ventilation should hold compartments below 50 C; the
    # ceiling, so a hard-run engine room.
    t_engine_room_k: float = 323.15
    # A four-stroke diesel's gas behind the turbocharger, ~230 C from half to full
    # load (CIMAC Congress 2012, paper 7-1-016). Uptake metal, uninsulated, sits near
    # its gas. Limitation: a wet exhaust, as on most small craft, runs near the sea.
    t_exhaust_k: float = 503.15
    _interiors: dict[Interior, float] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in (
            "cloud_fraction",
            "fog_cover",
            "beam_transmittance",
            "fog_sunlight_transmittance",
        ):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} {getattr(self, name)} outside [0, 1]")
        interiors: dict[Interior, float] = {
            "none": self.t_air_k,
            "cabin": self.t_cabin_k,
            "engine_room": self.t_engine_room_k,
            "exhaust": self.t_exhaust_k,
        }
        object.__setattr__(self, "_interiors", interiors)

    def interior_k(self, interior: Interior) -> float:
        return self._interiors[interior]

    def stopped(self, off_s: float) -> "Environment":
        """The engine room and exhaust `off_s` after the engines stop, each cooling
        toward the air as a steel wall: lumped capacitance (Incropera & DeWitt 5.2),
        tau = rho c_p d / h, h the wind's convection and the wall's radiation.

        Limitation: radiation linearised at the air, so a hot uptake's first minutes
        cool a little slowly; and the engine's own mass keeps a closed engine room warm
        longer than its walls.
        """
        if off_s < 0.0:
            raise ValueError(f"off_s {off_s} is before the engines stopped")
        eps = ZONES["exhaust"].emissivity
        radiation = 4.0 * eps * STEFAN_BOLTZMANN_W_M2K4 * self.t_air_k**3
        h = float(wind_convection_w_m2k(self.wind_m_s)) + radiation
        tau_s = _STEEL_HEAT_CAPACITY_J_M3K * _MACHINERY_WALL_M / h
        left = math.exp(-off_s / tau_s)

        def cooled(running_k: float) -> float:
            return self.t_air_k + (running_k - self.t_air_k) * left

        return replace(
            self,
            t_engine_room_k=cooled(self.t_engine_room_k),
            t_exhaust_k=cooled(self.t_exhaust_k),
        )


def air_mass(elevation_rad: npt.ArrayLike) -> FloatArray:
    """Kasten & Young 1989: relative optical air mass at a sun elevation, finite at the
    horizon where 1 / sin is not."""
    elevation = np.asarray(elevation_rad, dtype=np.float64)
    elevation_deg = np.degrees(elevation)
    return 1.0 / (np.sin(elevation) + 0.50572 * (elevation_deg + 6.07995) ** -1.6364)


def direct_beam_w_m2(elevation_rad: float | None) -> float:
    """Meinel & Meinel 1976: clear-sky direct irradiance on a plane facing the sun."""
    if elevation_rad is None or elevation_rad <= 0.0:
        return 0.0
    am = float(air_mass(elevation_rad))
    return _MEINEL_SOLAR_CONSTANT_W_M2 * 0.7 ** (am**0.678)


def sky_longwave_w_m2(
    t_air_k: float,
    cloud_fraction: float = 0.0,
    cloud_base_m: float = 0.0,
    fog_cover: float = 0.0,
    atmosphere: Atmosphere = ATMOSPHERE,
) -> float:
    """Downwelling broadband longwave on a horizontal surface.

    Clear sky by Swinbank 1963. Cloud covers its fraction as a blackbody at the air,
    less what the 8-14 um window loses to a colder base: only the window sees past the
    lowest air (Kimball, Idso & Aase, Water Resour. Res. 18(4) 931, 1982). Judgement:
    mixed linearly with the clear sky, as in Crawford & Duchon, J. Appl. Meteor. 38(4)
    474, 1999, unchecked. Fog covers `fog_cover` as a blackbody at the air, hiding the
    cloud. Past about 312 K, extrapolated, Swinbank's clear sky outshines an overcast at
    2400 m.
    """
    air = STEFAN_BOLTZMANN_W_M2K4 * t_air_k**4
    clear = _SWINBANK_W_M2K6 * t_air_k**6
    cloud = air - cloud_window_deficit_w_m2(cloud_base_m, t_air_k, atmosphere)
    sky = (1.0 - cloud_fraction) * clear + cloud_fraction * cloud
    return (1.0 - fog_cover) * sky + fog_cover * air


def wind_convection_w_m2k(wind_m_s: npt.ArrayLike) -> FloatArray:
    """McAdams 1954: h = 5.7 + 3.8 V, V in m/s, a flat plate in wind.

    Limitation: the fit was to plates whose loss included some radiation and free
    convection, which the balance also counts, so it runs high and sunlit facets a
    little cool.
    """
    return 5.7 + 3.8 * np.asarray(wind_m_s, dtype=np.float64)


def balance_temperature_k(
    absorbed_w_m2: npt.ArrayLike,
    emissivity: npt.ArrayLike,
    conductance_w_m2k: npt.ArrayLike,
    t_ref_k: npt.ArrayLike,
) -> FloatArray:
    """Solve eps sigma T^4 + g (T - T_ref) = Q for T, elementwise.

    g lumps convection and conduction, and T_ref is their weighted temperature. The
    left side rises monotonically from -g T_ref at T = 0, so bisection between 0 and
    a bound where it exceeds Q always closes.
    """
    q, eps, g, t_ref = np.broadcast_arrays(
        *(
            np.asarray(a, dtype=np.float64)
            for a in (absorbed_w_m2, emissivity, conductance_w_m2k, t_ref_k)
        )
    )
    if np.any((eps <= 0.0) & (g <= 0.0)):
        raise ValueError("a facet that neither radiates nor conducts has no balance")
    with np.errstate(divide="ignore", invalid="ignore"):
        radiative = np.where(
            eps > 0.0, (q / (eps * STEFAN_BOLTZMANN_W_M2K4)) ** 0.25, np.inf
        )
        linear = np.where(g > 0.0, t_ref + q / g, np.inf)
    # Past both, each term alone outweighs Q.
    high = np.minimum(np.maximum(radiative, t_ref), linear)
    low = np.zeros_like(high)
    # Halving from under 2000 K: 64 steps leave less than 1e-15 K.
    for _ in range(64):
        mid = 0.5 * (low + high)
        excess = eps * STEFAN_BOLTZMANN_W_M2K4 * mid**4 + g * (mid - t_ref) - q
        low = np.where(excess < 0.0, mid, low)
        high = np.where(excess < 0.0, high, mid)
    return 0.5 * (low + high)


def facet_temperature_k(
    zone: Zone,
    env: Environment,
    normal_elev_rad: npt.ArrayLike,
    sun_azimuth_rad: npt.ArrayLike,
    sunlit: npt.ArrayLike = True,
) -> FloatArray:
    """Temperature of facets of `zone` whose outward normals stand at `normal_elev_rad`
    above the horizon and `sun_azimuth_rad` round from the sun's bearing.

    Absorbed: alpha times the direct beam by max(0, n . s) where `sunlit`, and the
    isotropic diffuse share over the sky the facet sees; eps times the longwave from
    sky and sea, split by view factors (1 +- sin elev) / 2. Lost: eps sigma T^4,
    convection to the air and conduction to the interior.

    Cloud takes the beam a `cloud_fraction` share of the time, so the beam is that
    much weaker on average. Fog passes `fog_sunlight_transmittance` of the sunlight,
    `beam_transmittance` of the beam and the rest as diffuse light.

    Limitation: no sun off the sea, no light from the vessel's own structure, and the
    diffuse share stays clear-sky's under cloud, where it rises.
    """
    elev = np.asarray(normal_elev_rad, dtype=np.float64)
    azimuth = np.asarray(sun_azimuth_rad, dtype=np.float64)
    sun_elev = env.sun_elevation_rad or 0.0
    cos_incidence = np.cos(elev) * math.cos(sun_elev) * np.cos(azimuth) + np.sin(
        elev
    ) * math.sin(sun_elev)
    lit = np.asarray(sunlit, dtype=bool)
    return lit_temperature_k(
        zone, env, np.maximum(cos_incidence, 0.0) * lit, np.sin(elev)
    )


def lit_temperature_k(
    zone: Zone, env: Environment, sun_cos: npt.ArrayLike, normal_up: npt.ArrayLike
) -> FloatArray:
    """`facet_temperature_k` by the two numbers it depends on: `sun_cos`, max(0, n . s)
    where sunlit and 0 in shadow, and `normal_up`, the normal's vertical component."""
    sun_cos = np.asarray(sun_cos, dtype=np.float64)
    up = np.asarray(normal_up, dtype=np.float64)
    t_interior = env.interior_k(zone.interior)
    shape = np.broadcast_shapes(sun_cos.shape, up.shape)
    if math.isinf(zone.u_w_m2k):
        return np.full(shape, t_interior)

    sky_view = 0.5 * (1.0 + up)
    beam = direct_beam_w_m2(env.sun_elevation_rad)
    direct = (1.0 - env.cloud_fraction) * env.beam_transmittance
    # The beam the fog takes and does not absorb arrives as diffuse light, on a
    # horizontal surface sin e of the beam's.
    passed = env.fog_sunlight_transmittance
    up_sun = math.sin(env.sun_elevation_rad or 0.0)
    scattered = up_sun * max(passed - env.beam_transmittance, 0.0)
    diffuse = _DIFFUSE_FRACTION * passed + (1.0 - env.cloud_fraction) * scattered
    solar = beam * (direct * sun_cos + diffuse * sky_view)

    sky = sky_longwave_w_m2(
        env.t_air_k,
        env.cloud_fraction,
        env.cloud_base_m,
        env.fog_cover,
        env.atmosphere,
    )
    sea = (
        SEA_EMISSIVITY * STEFAN_BOLTZMANN_W_M2K4 * env.t_sea_k**4
        + (1.0 - SEA_EMISSIVITY) * sky
    )
    longwave = sky_view * sky + (1.0 - sky_view) * sea

    absorbed = zone.absorptance * solar + zone.emissivity * longwave
    h = float(wind_convection_w_m2k(env.wind_m_s))
    g = h + zone.u_w_m2k
    t_ref = (h * env.t_air_k + zone.u_w_m2k * t_interior) / g
    return np.broadcast_to(
        balance_temperature_k(absorbed, zone.emissivity, g, t_ref), shape
    ).copy()


def temperature_table_k(
    zone: Zone, env: Environment, n_sun: int = 33, n_up: int = 33
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """`lit_temperature_k` over its two inputs, for a shader to look up.

    Returns (sun_cos, normal_up, table): sun_cos spans 0 to 1, normal_up -1 to 1, and
    the table is shaped (n_up, n_sun). The temperature is smooth in both, so a linear
    lookup needs no fine grid.
    """
    sun_cos = np.linspace(0.0, 1.0, n_sun)
    up = np.linspace(-1.0, 1.0, n_up)
    return sun_cos, up, lit_temperature_k(zone, env, sun_cos[None, :], up[:, None])
