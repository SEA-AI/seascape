import math

import numpy as np
import pytest

from seascape.physics import aerosol, lwir

# Koschmieder's extinction at 0.5 km: 2% contrast over the visibility.
BETA_500M = math.log(50) / 500.0

# lowtran_haze.csv, north_sea at 90 degrees: what 2 km maritime haze passes overhead.
HAZE_2KM_ZENITH = 0.782024

ELEVATIONS = np.radians([0.0, 10.0, 30.0, 90.0])


def _sky_k(visibility_km: float | None) -> np.ndarray:
    clear = lwir.sky_radiance(ELEVATIONS, 283.0, "north_sea")
    seen = aerosol.sky_radiance(clear, ELEVATIONS, visibility_km, 283.0, "north_sea")
    return lwir.brightness_temperature(seen)


def test_fog_glows_at_the_air_along_the_horizon_and_thins_overhead() -> None:
    elevation = np.radians([0.0, 90.0])
    clear = lwir.sky_radiance(elevation, 283.0)
    horizon, zenith = aerosol.sky_radiance(clear, elevation, 0.5, 283.0)
    air = lwir.band_radiance(283.0)
    assert horizon == pytest.approx(air)
    assert clear[1] < zenith < air


def test_haze_overhead_is_lowtran_s() -> None:
    depth = aerosol.depth(math.pi / 2, 2.0, "north_sea")
    assert depth == pytest.approx(-math.log(HAZE_2KM_ZENITH), rel=1e-6)
    assert aerosol.depth(math.pi / 2, 1.5, "north_sea") > depth
    assert aerosol.depth(ELEVATIONS, None) == pytest.approx(0.0)


def test_fog_is_its_layer_of_drops_alone() -> None:
    slant = aerosol.DEPTH_M / np.sin(ELEVATIONS[1:])
    for visibility_km in (0.999, 0.5, 0.1):
        drops = lwir.path_optical_depth(slant, visibility_km) - lwir.path_optical_depth(
            slant, None
        )
        assert aerosol.depth(ELEVATIONS[1:], visibility_km) == pytest.approx(drops)


@pytest.mark.parametrize("edge_km", [aerosol.FOG_KM, aerosol.HAZE_KM])
def test_the_sky_and_the_sun_run_on_through_mist(edge_km: float) -> None:
    over, under = edge_km * 1.001, edge_km * 0.999
    # A tenth of a kelvin over 0.1% of visibility.
    assert _sky_k(over) == pytest.approx(_sky_k(under), abs=0.1)
    assert aerosol.cover(over) == pytest.approx(aerosol.cover(under), abs=0.01)
    assert aerosol.fog_cover(over) == pytest.approx(aerosol.fog_cover(under), abs=0.01)
    sun = math.radians(30.0)
    assert aerosol.sun_transmittance(sun, over) == pytest.approx(
        aerosol.sun_transmittance(sun, under), abs=0.01
    )
    assert aerosol.diffuse_transmittance(over) == pytest.approx(
        aerosol.diffuse_transmittance(under), abs=0.01
    )


def test_the_aerosol_covers_the_sky_as_a_horizontal_surface_sees_it() -> None:
    e = np.linspace(0.0, np.pi / 2, 20_001)
    weight = np.sin(e) * np.cos(e)
    for visibility_km in (10.0, 1.5, 0.5):
        opaque = 1.0 - aerosol.transmittance(e, visibility_km)
        expected = np.trapezoid(opaque * weight, e) / np.trapezoid(weight, e)
        assert aerosol.cover(visibility_km) == pytest.approx(expected, rel=1e-3)


def test_thicker_air_covers_more_of_the_sky() -> None:
    covers = [aerosol.cover(v) for v in (None, 42.0, 5.0, 1.5, 0.99, 0.5, 0.2)]
    assert covers == sorted(covers)
    assert covers[0] == 0.0
    assert covers[-1] > 0.99


def test_the_heat_balance_takes_the_fog_alone() -> None:
    assert aerosol.fog_cover(5.0) == 0.0
    assert aerosol.fog_cover(0.5) == aerosol.cover(0.5)
    assert aerosol.sun_transmittance(math.pi / 2, 5.0) == 1.0
    assert aerosol.diffuse_transmittance(None) == 1.0


def test_a_high_sun_dims_through_the_fog_and_a_low_one_vanishes() -> None:
    overhead = aerosol.sun_transmittance(math.pi / 2, 0.5)
    assert overhead == pytest.approx(math.exp(-BETA_500M * aerosol.DEPTH_M))
    assert aerosol.sun_transmittance(math.radians(5.0), 0.5) < 1e-6
    assert aerosol.sun_transmittance(-0.1, 0.5) == 0.0


def test_fog_passes_more_sunlight_than_its_beam() -> None:
    passed = aerosol.diffuse_transmittance(0.5)
    # 2 / (2 + 0.15 x 1.565), by hand.
    assert passed == pytest.approx(0.8950, abs=1e-4)
    assert passed > aerosol.sun_transmittance(math.pi / 2, 0.5)
