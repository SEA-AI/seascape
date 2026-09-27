import math

import numpy as np
import pytest

from seascape import lwir, waves


def field(wind_speed_mps: float, wind_from_deg: float = 0.0) -> tuple[waves.Wave, ...]:
    return waves.components(
        wind_speed_mps, wind_from_deg, lambda p: p, np.random.default_rng(0)
    )


def test_published_values_have_not_drifted() -> None:
    # Cox & Munk 1954: RMS slope of a clean sea at 7 m/s, off sun glitter photographs.
    assert waves.wave_slope(7.0) == pytest.approx(0.197, abs=5e-4)
    # Surveying's rule of thumb for the horizon, 3.86 sqrt(h_m) km at k = 0.13.
    rule_m = 3.86e3 * math.sqrt(51.8)
    assert waves.horizon_m(51.8, 0.13) == pytest.approx(rule_m, rel=0.01)
    # Pierson-Moskowitz: a fully developed sea at 7 m/s peaks near 41 m.
    peak = waves.peak_omega_rad_s(7.0)
    assert 2 * math.pi * waves.GRAVITY_MS2 / peak**2 == pytest.approx(40.8, abs=0.2)


def test_the_unresolved_slope_lifts_grazing_emissivity() -> None:
    # Masuda 1988 at this wind speed: near nadir, and at 80 deg where roughness has
    # taken hold, lifting emissivity above flat Fresnel.
    theta, eps = lwir.emissivity_curve(
        t_sea_k=291.0, slope_sigma=waves.unresolved_slope(7.0, field(7.0))
    )
    assert float(np.interp(0.0, theta, eps)) == pytest.approx(0.985, abs=0.005)
    assert float(np.interp(math.radians(80.0), theta, eps)) == pytest.approx(
        0.76, abs=0.03
    )


@pytest.mark.parametrize("wind_speed_mps", [3.0, 7.0, 12.0, 20.0])
def test_the_field_carries_the_spectrum_s_wave_height(wind_speed_mps) -> None:
    # Pierson-Moskowitz: Hs = 0.209 U^2 / g, nearly all of it in the waves built.
    hs = 4 * math.sqrt(sum(w.amplitude_m**2 / 2 for w in field(wind_speed_mps)))
    assert hs == pytest.approx(0.209 * wind_speed_mps**2 / waves.GRAVITY_MS2, rel=0.01)


def test_waves_run_away_from_the_wind() -> None:
    from_east = field(7.0, wind_from_deg=90.0)
    assert np.mean([w.k_east_rad_m for w in from_east]) < 0.0


def test_a_crest_travels_at_its_phase_speed() -> None:
    wave = waves.Wave(1.0, 1.2, math.radians(30.0), 0.4)
    speed_ms = wave.omega_rad_s / wave.k_rad_m
    east, north = math.sin(wave.toward_rad), math.cos(wave.toward_rad)
    t_s = 2.5
    before = waves.height_m((wave,), np.array([3.0]), np.array([4.0]), 0.0)
    after = waves.height_m(
        (wave,),
        np.array([3.0 + speed_ms * t_s * east]),
        np.array([4.0 + speed_ms * t_s * north]),
        t_s,
    )
    assert after == pytest.approx(before)


def test_a_snapped_field_repeats_after_the_loop() -> None:
    span_s = 40.0
    snapped = waves.components(
        7.0,
        0.0,
        lambda p: span_s / max(1, round(span_s / p)),
        np.random.default_rng(0),
    )
    east, north = np.meshgrid(np.linspace(-50, 50, 9), np.linspace(-50, 50, 9))
    assert waves.height_m(snapped, east, north, span_s) == pytest.approx(
        waves.height_m(snapped, east, north, 0.0), abs=1e-9
    )


def test_a_longer_loop_snaps_the_frequencies_less() -> None:
    def loop(span_s: float) -> float:
        return waves.snap_error(7.0, lambda p: span_s / max(1, round(span_s / p)))

    assert loop(90.0) < loop(40.0) < loop(10.0)
    assert waves.snap_error(7.0, lambda p: p) == pytest.approx(0.0, abs=1e-12)


def test_the_slope_the_waves_leave_out_is_the_rest_of_cox_and_munk() -> None:
    built = field(7.0)
    total = waves.wave_slope(7.0) ** 2
    rest = waves.unresolved_slope(7.0, built) ** 2
    assert waves.resolved_slope_variance(built) + rest == pytest.approx(total)


def test_calm_air_builds_no_waves() -> None:
    assert field(0.0) == ()
    assert waves.unresolved_slope(0.0, ()) == pytest.approx(waves.wave_slope(0.0))


def test_the_slope_is_the_height_s_gradient() -> None:
    built = field(7.0)
    east, north, step = np.array([12.0]), np.array([-7.0]), 1e-4
    d_east, d_north = waves.slope(built, east, north, 1.3)
    ahead = waves.height_m(built, east + step, north, 1.3)
    behind = waves.height_m(built, east - step, north, 1.3)
    assert d_east == pytest.approx((ahead - behind) / (2 * step), rel=1e-4)
    ahead = waves.height_m(built, east, north + step, 1.3)
    behind = waves.height_m(built, east, north - step, 1.3)
    assert d_north == pytest.approx((ahead - behind) / (2 * step), rel=1e-4)
