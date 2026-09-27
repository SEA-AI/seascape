import math

import numpy as np
import pytest

from seascape import lwir, waves


def test_published_values_have_not_drifted() -> None:
    # Cox & Munk 1954: RMS slope of a clean sea at 7 m/s, off sun glitter photographs.
    assert waves.wave_slope(7.0) == pytest.approx(0.197, abs=5e-4)
    # Surveying's rule of thumb for the horizon, 3.86 sqrt(h_m) km at k = 0.13.
    rule_m = 3.86e3 * math.sqrt(51.8)
    assert waves.horizon_m(51.8, 0.13) == pytest.approx(rule_m, rel=0.01)
    # Pierson-Moskowitz: a fully developed sea at 7 m/s peaks near 41 m.
    assert waves.wave_length_m(7.0) == pytest.approx(40.8, abs=0.2)
    # Its peak frequency, 0.877 g / U, as a period.
    assert waves.wave_period_s(7.0) == pytest.approx(2 * math.pi * 7.0 / (0.877 * 9.81))
    # Minimum phase speed of a surface wave, where surface tension takes over.
    assert abs(waves.CAPILLARY_WAVELENGTH_M - 0.0173) < 1e-4


def test_the_unresolved_slope_lifts_grazing_emissivity() -> None:
    # Masuda 1988 at this wind speed: near nadir, and at 80 deg where roughness has
    # taken hold, lifting emissivity above flat Fresnel.
    theta, eps = lwir.emissivity_curve(
        t_sea_k=291.0, slope_sigma=waves.unresolved_slope(7.0)
    )
    assert float(np.interp(0.0, theta, eps)) == pytest.approx(0.985, abs=0.005)
    assert float(np.interp(math.radians(80.0), theta, eps)) == pytest.approx(
        0.76, abs=0.03
    )
