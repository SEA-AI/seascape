import math

import numpy as np
import pytest

from seascape import lwir, waves


def field(wind_speed_mps: float, wind_from_deg: float = 0.0) -> tuple[waves.Wave, ...]:
    return waves.wind_sea(
        wind_speed_mps, wind_from_deg, lambda p: p, np.random.default_rng(0)
    )


def u10_for(wind_mps: float, height_m: float) -> float:
    """The wind at 10 m that reads about `wind_mps` at `height_m`, to first order."""
    return wind_mps * wind_mps / waves.wind_at_m(wind_mps, height_m)


def test_published_values_have_not_drifted() -> None:
    # Cox & Munk 1954: RMS slope of a clean sea at 7 m/s at 12.5 m, off sun glitter.
    slope = waves.cox_munk_slope(u10_for(7.0, waves.COX_MUNK_WIND_HEIGHT_M))
    assert slope == pytest.approx(0.197, abs=5e-4)
    # Surveying's rule of thumb for the horizon, 3.86 sqrt(h_m) km at k = 0.13.
    rule_m = 3.86e3 * math.sqrt(51.8)
    assert waves.horizon_m(51.8, 0.13) == pytest.approx(rule_m, rel=0.01)
    # Pierson-Moskowitz: a fully developed sea at 7 m/s at 19.5 m peaks near 41 m.
    peak = waves.peak_omega_rad_s(u10_for(7.0, waves.PM_WIND_HEIGHT_M))
    assert 2 * math.pi * waves.GRAVITY_MS2 / peak**2 == pytest.approx(40.8, abs=0.2)


def test_the_wind_rises_with_height_as_the_log_profile_over_the_sea() -> None:
    # From DNV-RP-C205 2010, 2.3.2.4.
    assert waves.wind_at_m(10.0, 10.0) == pytest.approx(10.0)
    assert waves.wind_at_m(10.0, 19.5) == pytest.approx(10.60, abs=0.01)
    assert waves.wind_at_m(10.0, 12.5) == pytest.approx(10.20, abs=0.01)
    assert waves.wind_at_m(3.0, 19.5) == pytest.approx(3 * 1.048, abs=0.01)
    assert waves.wind_at_m(20.0, 19.5) == pytest.approx(20 * 1.071, abs=0.02)
    assert waves.wind_at_m(0.0, 19.5) == 0.0


def test_a_pixel_that_resolves_less_emits_more_at_grazing() -> None:
    """Masuda 1988: unresolved slope lifts grazing emissivity off flat Fresnel."""
    built = field(7.0)
    # lwir takes the slope per axis: half the total variance.
    near, far = (
        math.sqrt(waves.unresolved_slope_variance(7.0, built, (), f) / 2)
        for f in (0.01, 50)
    )
    eps = {}
    for name, sigma in (("near", near), ("far", far)):
        theta, curve = lwir.emissivity_curve(t_sea_k=291.0, slope_sigma=sigma)
        eps[name] = float(np.interp(math.radians(80.0), theta, curve))
        assert float(np.interp(0.0, theta, curve)) == pytest.approx(0.985, abs=0.005)
    assert eps["far"] > eps["near"]


@pytest.mark.parametrize("wind_speed_mps", [3.0, 7.0, 12.0, 20.0])
def test_the_field_carries_the_spectrum_s_wave_height(wind_speed_mps) -> None:
    # Pierson-Moskowitz: Hs = 0.209 U(19.5 m)^2 / g, nearly all of it in the waves.
    hs = 4 * math.sqrt(sum(w.amplitude_m**2 / 2 for w in field(wind_speed_mps)))
    wind = waves.wind_at_m(wind_speed_mps, waves.PM_WIND_HEIGHT_M)
    assert hs == pytest.approx(0.209 * wind**2 / waves.GRAVITY_MS2, rel=0.01)


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
    snapped = waves.wind_sea(
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


def test_a_pixel_leaves_to_roughness_what_it_cannot_resolve() -> None:
    built = field(7.0)
    total = waves.cox_munk_slope(7.0) ** 2
    sharp = waves.unresolved_slope_variance(7.0, built, (), 1e-4)
    assert sharp == pytest.approx(total - waves.slope_variance(built))
    assert waves.unresolved_slope_variance(7.0, built, (), 1e4) == pytest.approx(total)
    footprints = [0.01, 0.1, 1.0, 10.0]
    rest = [waves.unresolved_slope_variance(7.0, built, (), f) for f in footprints]
    assert rest == sorted(rest), "a coarser pixel leaves more"


def test_a_wave_fades_out_before_it_aliases() -> None:
    lo, hi = waves.FADE_FOOTPRINTS
    per_footprint = np.array([lo, 2.5, 3.0, hi, 2 * hi])
    shown = waves.visibility(per_footprint, 1.0)
    assert (shown[0], shown[3], shown[4]) == (0.0, 1.0, 1.0)
    assert np.all(np.diff(shown) >= 0)


def test_calm_air_builds_no_waves() -> None:
    assert field(0.0) == ()
    assert waves.unresolved_slope_variance(0.0, (), (), 1.0) == pytest.approx(
        waves.cox_munk_slope(0.0) ** 2
    )


def test_the_slope_is_the_height_s_gradient() -> None:
    built = field(7.0)
    east, north, step = np.array([12.0]), np.array([-7.0]), 1e-5
    d_east, d_north = waves.slope(built, east, north, 1.3)
    ahead = waves.height_m(built, east + step, north, 1.3)
    behind = waves.height_m(built, east - step, north, 1.3)
    assert d_east == pytest.approx((ahead - behind) / (2 * step), rel=1e-4)
    ahead = waves.height_m(built, east, north + step, 1.3)
    behind = waves.height_m(built, east, north - step, 1.3)
    assert d_north == pytest.approx((ahead - behind) / (2 * step), rel=1e-4)


def test_a_swell_carries_its_height_from_where_it_comes() -> None:
    swell = waves.swell(2.0, 12.0, 90.0, lambda p: p, np.random.default_rng(0))
    assert 4 * math.sqrt(sum(w.amplitude_m**2 / 2 for w in swell)) == pytest.approx(2.0)
    assert all(w.omega_rad_s == pytest.approx(2 * math.pi / 12.0) for w in swell)
    assert np.mean([w.k_east_rad_m for w in swell]) < 0.0


def test_whitecaps_cover_what_monahan_measured() -> None:
    wind = field(15.0)
    fraction = waves.whitecap_fraction(15.0)
    threshold = waves.breaking_threshold_g(wind, fraction)
    east, north = np.meshgrid(np.linspace(0, 3000, 600), np.linspace(0, 3000, 600))
    covered = np.mean(waves.downward_acceleration_g(wind, east, north, 0.0) > threshold)
    assert covered == pytest.approx(fraction, rel=0.15)


@pytest.mark.parametrize("footprint_m", [1e-3, 5.0, 100.0])
def test_a_pixel_whitecaps_over_monahan_s_cover_at_any_footprint(footprint_m) -> None:
    wind = field(15.0)
    fraction = waves.whitecap_fraction(15.0)
    east, north = np.meshgrid(np.linspace(0, 3000, 600), np.linspace(0, 3000, 600))
    cover = waves.whitecap_cover(wind, east, north, 0.0, footprint_m, fraction)
    assert float(np.mean(cover)) == pytest.approx(fraction, rel=0.15)


def test_whitecaps_read_the_wind_at_10_m() -> None:
    # Monahan & O'Muircheartaigh 1980: 3.84e-6 U10^3.41, 0.99% at 10 m/s.
    assert waves.whitecap_fraction(10.0) == pytest.approx(0.0099, abs=1e-4)


def test_calm_air_breaks_nothing() -> None:
    assert waves.whitecap_fraction(0.0) == 0.0
    assert waves.breaking_threshold_g((), 0.0) == math.inf
    # A breath of wind builds no waves but has a whitecap fraction.
    assert waves.breaking_threshold_g((), waves.whitecap_fraction(0.5)) == math.inf


@pytest.mark.parametrize(
    ("toward_deg", "heading_deg", "expected"),
    [
        (0.0, 0.0, "pitch"),
        (90.0, 0.0, "roll"),
        (0.0, 90.0, "roll"),
    ],
)
def test_a_hull_takes_the_slope_of_a_long_wave(toward_deg, heading_deg, expected):
    wave = waves.Wave(0.5, 0.2, math.radians(toward_deg), 1.0)  # 1.5 km long
    east, north, t_s = 30.0, -40.0, 3.0
    pitch, roll = waves.attitude(
        (wave,), east, north, math.radians(heading_deg), 20.0, 5.0, t_s
    )
    d_east, d_north = waves.slope((wave,), np.array(east), np.array(north), t_s)
    heading = math.radians(heading_deg)
    rise_along = d_east * math.sin(heading) + d_north * math.cos(heading)
    rise_starboard = d_east * math.cos(heading) - d_north * math.sin(heading)
    assert (pitch, roll) == pytest.approx(
        (math.atan(rise_along), math.atan(rise_starboard)), abs=1e-5
    )
    assert abs({"pitch": pitch, "roll": roll}[expected]) > 1e-4


def test_calm_water_holds_a_hull_level() -> None:
    assert waves.attitude((), 0.0, 0.0, 0.0, 20.0, 5.0, 0.0) == (0.0, 0.0)


def test_a_hull_rides_over_a_wave_shorter_than_its_samples() -> None:
    ripple = waves.Wave(0.05, math.sqrt(waves.GRAVITY_MS2 * 2 * math.pi), 0.3, 1.0)
    assert waves.attitude((ripple,), 3.0, 4.0, 0.0, 20.0, 5.0, 1.0) == (0.0, 0.0)


def test_a_swell_adds_its_own_unresolved_slope_to_cox_and_munk_s() -> None:
    wind = field(7.0)
    swell = waves.swell(2.0, 12.0, 90.0, lambda p: p, np.random.default_rng(0))
    far = waves.unresolved_slope_variance(7.0, wind, swell, 1e4)
    assert far == pytest.approx(
        waves.cox_munk_slope(7.0) ** 2 + waves.slope_variance(swell)
    )
    near = waves.unresolved_slope_variance(7.0, wind, swell, 1e-3)
    assert near == pytest.approx(waves.unresolved_slope_variance(7.0, wind, (), 1e-3))
