import math

import numpy as np
import pytest

from seascape.physics import fog, lwir

# Koschmieder's extinction at 0.5 km: 2% contrast over the visibility.
BETA_500M = math.log(50) / 500.0


def test_fog_glows_at_the_air_along_the_horizon_and_thins_overhead() -> None:
    elevation = np.radians([0.0, 90.0])
    clear = lwir.sky_radiance(elevation, 283.0)
    horizon, zenith = fog.sky_radiance(clear, elevation, 0.5, 283.0)
    air = lwir.band_radiance(283.0)
    assert horizon == pytest.approx(air)
    assert clear[1] < zenith < air


def test_fog_covers_the_sky_as_a_horizontal_surface_sees_it() -> None:
    e = np.linspace(0.0, np.pi / 2, 20_001)
    weight = np.sin(e) * np.cos(e)
    for visibility_km in (0.99, 0.5):
        opaque = 1.0 - fog.transmittance(e, visibility_km)
        expected = np.trapezoid(opaque * weight, e) / np.trapezoid(weight, e)
        assert fog.cover(visibility_km) == pytest.approx(expected, rel=1e-3)


def test_thicker_fog_covers_more_of_the_sky() -> None:
    thin, half, thick = (fog.cover(v) for v in (0.99, 0.5, 0.2))
    assert thin < half < thick
    assert thick > 0.99


def test_a_high_sun_dims_through_the_fog_and_a_low_one_vanishes() -> None:
    overhead = fog.sun_transmittance(math.pi / 2, BETA_500M)
    assert overhead == pytest.approx(math.exp(-BETA_500M * fog.DEPTH_M))
    assert fog.sun_transmittance(math.radians(5.0), BETA_500M) < 1e-6
    assert fog.sun_transmittance(-0.1, BETA_500M) == 0.0
