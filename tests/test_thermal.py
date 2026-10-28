"""Heat balance of a vessel's thermal zones. No Blender."""

import dataclasses
import itertools
import math

import numpy as np
import pytest

from seascape.physics import thermal

T_AIR_K = 288.15
# Thompson et al. 1998's frigate cruises at 20 kn.
TWENTY_KNOTS_M_S = 20 * 1852 / 3600
# Thompson et al. 1998: the sun abeam, past 10 degrees up.
SUN_ELEVATION_RAD = math.radians(30.0)


def environment(
    wind_m_s: float, sun_elevation_rad: float | None
) -> thermal.Environment:
    return thermal.Environment(T_AIR_K, T_AIR_K, wind_m_s, sun_elevation_rad)


def painted(absorptance: float) -> thermal.Zone:
    return dataclasses.replace(thermal.ZONES["paint_hull"], absorptance=absorptance)


def facing_sun_k(zone: thermal.Zone, wind_m_s: float) -> float:
    """A facet turned square to the sun."""
    env = environment(wind_m_s, SUN_ELEVATION_RAD)
    return float(thermal.facet_temperature_k(zone, env, SUN_ELEVATION_RAD, 0.0))


def test_radiative_equilibrium() -> None:
    """With nothing to conduct to, eps sigma T^4 = Q whatever eps is."""
    q = 300.0
    t = thermal.balance_temperature_k(q, [0.3, 0.94], 0.0, T_AIR_K)
    expected = (q / (np.array([0.3, 0.94]) * thermal.STEFAN_BOLTZMANN_W_M2K4)) ** 0.25
    np.testing.assert_allclose(t, expected, rtol=1e-12)


def test_a_facet_that_does_not_radiate_stays_at_air_temperature() -> None:
    zone = thermal.Zone(0.0, (0.0, 0.0), 0.0, "none", 0.0)
    t = thermal.facet_temperature_k(zone, environment(5.0, None), 0.3, 0.0)
    assert float(t) == pytest.approx(T_AIR_K)


def test_swinbank_sky_runs_tens_of_watts_below_the_air() -> None:
    """A clear sky's effective emissivity at 288 K is about 0.78."""
    ratio = thermal.sky_longwave_w_m2(T_AIR_K) / (
        thermal.STEFAN_BOLTZMANN_W_M2K4 * T_AIR_K**4
    )
    assert 0.7 < ratio < 0.85


def test_overcast_sky_radiates_as_the_air() -> None:
    sky = thermal.sky_longwave_w_m2(T_AIR_K, cloud_fraction=1.0)
    assert sky == pytest.approx(thermal.STEFAN_BOLTZMANN_W_M2K4 * T_AIR_K**4)


def test_grey_paint_in_sun_at_twenty_knots_is_about_ten_kelvin_over_air() -> None:
    """Thompson et al. 1998: past 10 degrees of sun, a frigate's surfaces at 20 kn run
    in excess of +10 C."""
    rise = facing_sun_k(painted(thermal.GREY_PAINT_ABSORPTANCE), TWENTY_KNOTS_M_S)
    assert 5.0 < rise - T_AIR_K < 15.0


def test_clear_night_deck_runs_below_the_air() -> None:
    """Swinbank's sky is ~85 W m^-2 short of the air's blackbody at 288 K; against
    convection of 6-20 W m^-2 K^-1 that is 3-8 K of cooling."""
    deck = thermal.ZONES["deck"]
    for wind_m_s in (0.0, 3.0):
        t = thermal.facet_temperature_k(
            deck, environment(wind_m_s, None), 0.5 * math.pi, 0.0
        )
        assert -9.0 < float(t) - T_AIR_K < -2.0


def test_no_heated_interior_puts_a_hull_side_between_air_and_sea_at_night() -> None:
    """Thompson et al. 1998: at night a well-insulated hull sits between air and sea.
    A vertical side sees half sky, half sea."""
    env = thermal.Environment(T_AIR_K, T_AIR_K + 4.0, 5.0, None)
    t = float(thermal.facet_temperature_k(thermal.ZONES["paint_hull"], env, 0.0, 0.0))
    assert T_AIR_K - 2.0 < t < T_AIR_K + 4.0


def test_temperature_rises_with_absorptance() -> None:
    rises = [facing_sun_k(painted(a), 3.0) for a in (0.14, 0.65, 0.97)]
    assert rises[0] < rises[1] < rises[2]


def test_wind_pulls_a_sunlit_facet_towards_the_air() -> None:
    zone = painted(thermal.GREY_PAINT_ABSORPTANCE)
    rises = [facing_sun_k(zone, v) - T_AIR_K for v in (1.0, 3.0, 10.0, 20.0)]
    assert all(a > b > 0.0 for a, b in itertools.pairwise(rises))


def test_shadow_and_a_turned_back_take_only_diffuse_sun() -> None:
    zone = thermal.ZONES["paint_hull"]
    env = environment(5.0, SUN_ELEVATION_RAD)
    shadowed = thermal.facet_temperature_k(zone, env, 0.0, 0.0, sunlit=False)
    turned = thermal.facet_temperature_k(zone, env, 0.0, math.pi)
    assert float(shadowed) == pytest.approx(float(turned))


def test_overcast_takes_the_beam_and_leaves_the_diffuse() -> None:
    zone = thermal.ZONES["paint_hull"]
    env = environment(5.0, SUN_ELEVATION_RAD)
    overcast = dataclasses.replace(env, cloud_fraction=1.0)
    facing = thermal.facet_temperature_k(zone, overcast, 0.0, 0.0)
    turned = thermal.facet_temperature_k(zone, overcast, 0.0, math.pi)
    assert float(facing) == pytest.approx(float(turned))
    clear = thermal.facet_temperature_k(zone, env, 0.0, 0.0)
    assert float(facing) < float(clear)


def test_opaque_fog_is_overcast_to_a_facet_at_night() -> None:
    zone = thermal.ZONES["paint_hull"]
    env = environment(5.0, None)
    fog = dataclasses.replace(env, fog_cover=1.0, beam_transmittance=0.0)
    overcast = dataclasses.replace(env, cloud_fraction=1.0)
    assert float(thermal.facet_temperature_k(zone, fog, 0.0, 0.0)) == pytest.approx(
        float(thermal.facet_temperature_k(zone, overcast, 0.0, 0.0))
    )


def test_fog_that_scatters_the_whole_beam_keeps_a_deck_as_warm_as_clear_sky() -> None:
    """Light conserved: a horizontal facet takes the beam as diffuse instead."""
    zone = thermal.ZONES["paint_hull"]
    env = environment(5.0, SUN_ELEVATION_RAD)
    scattered = dataclasses.replace(env, beam_transmittance=0.0)
    deck = math.sin(SUN_ELEVATION_RAD)
    clear = thermal.lit_temperature_k(zone, env, deck, 1.0)
    assert thermal.lit_temperature_k(zone, scattered, 0.0, 1.0) == pytest.approx(clear)
    absorbed = dataclasses.replace(scattered, fog_diffuse_transmittance=0.0)
    assert float(thermal.lit_temperature_k(zone, absorbed, 0.0, 1.0)) < float(clear)


def test_exhaust_metal_holds_its_uptake_temperature() -> None:
    env = environment(10.0, SUN_ELEVATION_RAD)
    t = thermal.facet_temperature_k(thermal.ZONES["exhaust"], env, 0.0, 0.0)
    assert float(t) == env.t_exhaust_k


def test_stopped_engines_cool_toward_the_air() -> None:
    running = environment(5.0, SUN_ELEVATION_RAD)
    assert running.stopped(0.0).t_exhaust_k == pytest.approx(running.t_exhaust_k)
    soon, later = running.stopped(600.0), running.stopped(3600.0)
    for t in ("t_engine_room_k", "t_exhaust_k"):
        assert getattr(running, t) > getattr(soon, t) > getattr(later, t) > T_AIR_K
    assert running.stopped(1e6).t_exhaust_k == pytest.approx(T_AIR_K)
    assert soon.t_cabin_k == running.t_cabin_k


def test_wind_cools_a_stopped_engine_faster() -> None:
    calm = environment(1.0, SUN_ELEVATION_RAD).stopped(600.0)
    windy = environment(12.0, SUN_ELEVATION_RAD).stopped(600.0)
    assert windy.t_exhaust_k < calm.t_exhaust_k


def test_engines_cannot_stop_after_the_frame() -> None:
    with pytest.raises(ValueError, match="before the engines stopped"):
        environment(5.0, SUN_ELEVATION_RAD).stopped(-1.0)


def test_air_mass_is_one_overhead_and_finite_at_the_horizon() -> None:
    """Kasten & Young 1989 tabulate 38.09 at the horizon."""
    assert float(thermal.air_mass(0.5 * math.pi)) == pytest.approx(1.0, abs=1e-3)
    assert float(thermal.air_mass(0.0)) == pytest.approx(38.09, rel=0.01)


def test_table_matches_the_facets() -> None:
    zone = thermal.ZONES["paint_superstructure"]
    env = environment(5.0, SUN_ELEVATION_RAD)
    sun_cos, up, table = thermal.temperature_table_k(zone, env, 9, 7)
    assert table.shape == (7, 9)
    assert np.all(np.diff(table, axis=1) > 0)
    # A facet facing the sun's bearing, tipped up by elevation e: n . s = cos(e - sun).
    elev = math.asin(up[4])
    probe = thermal.facet_temperature_k(zone, env, elev, 0.0)
    expected = np.interp(math.cos(elev - SUN_ELEVATION_RAD), sun_cos, table[4])
    assert float(probe) == pytest.approx(expected, abs=0.05)


def test_zone_refuses_an_absorptance_outside_its_range() -> None:
    with pytest.raises(ValueError, match="absorptance"):
        painted(0.99)
