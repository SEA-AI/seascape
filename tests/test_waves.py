import math
from collections.abc import Callable
from statistics import NormalDist

import numpy as np
import pytest

from seascape.physics import earth, lwir, waves
from seascape.physics.wakes import FRESH_FOAM_REFLECTANCE
from seascape.physics.waves import sea, surface
from tests import gerstner


def field(
    wind_speed_mps: float, wind_from_deg: float = 0.0, seed: int = 0
) -> tuple[waves.Wave, ...]:
    return waves.wind_sea(
        wind_speed_mps, wind_from_deg, None, lambda p: p, np.random.default_rng(seed)
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
    assert earth.horizon_m(51.8, 0.13) == pytest.approx(rule_m, rel=0.01)
    # Elfouhaily et al. 1997: a fully developed sea at 7 m/s peaks at
    # 2 pi U^2 / (g 0.84^2), 44.5 m.
    peak = waves.peak_omega_rad_s(7.0, None)
    assert 2 * math.pi * earth.GRAVITY_MS2 / peak**2 == pytest.approx(44.5, abs=0.2)


@pytest.mark.parametrize("north_m", [0.0, 7000.0])
def test_a_ray_stops_where_it_meets_the_curved_sea(north_m: float) -> None:
    """From 20 m over the sea, at the origin and 7 km out, where the sea has dropped
    under the tangent plane.
    """
    radius_m = earth.earth_radius_m(0.13)
    up_m = earth.sea_z_m(0.0, north_m, radius_m) + 20.0
    toward = np.array([0.0, 1.0, -0.01]) / math.hypot(1.0, 0.01)
    run_m = earth.sea_run_m((0.0, north_m, up_m), tuple(toward), radius_m)

    assert run_m is not None
    hit = np.array([0.0, north_m, up_m]) + toward * run_m / toward[1]
    assert hit[2] == pytest.approx(earth.sea_z_m(*hit[:2], radius_m), abs=1e-6)
    assert run_m > 20.0 / 0.01


def test_a_ray_above_the_dip_of_the_horizon_meets_no_sea() -> None:
    radius_m = earth.earth_radius_m(0.13)
    # The horizon dips sqrt(2 h / R), 1.6 mrad from 10 m: 0.1 mrad passes over it.
    toward = (0.0, 1.0, -1e-4)
    assert earth.sea_run_m((0.0, 0.0, 10.0), toward, radius_m) is None


def test_the_wind_rises_with_height_as_the_log_profile_over_the_sea() -> None:
    # From DNV-RP-C205 2010, 2.3.2.4.
    assert waves.wind_at_m(10.0, 10.0) == pytest.approx(10.0)
    assert waves.wind_at_m(10.0, 19.5) == pytest.approx(10.60, abs=0.01)
    assert waves.wind_at_m(10.0, 12.5) == pytest.approx(10.20, abs=0.01)
    assert waves.wind_at_m(3.0, 19.5) == pytest.approx(3 * 1.048, abs=0.01)
    assert waves.wind_at_m(20.0, 19.5) == pytest.approx(20 * 1.071, abs=0.02)
    assert waves.wind_at_m(0.0, 19.5) == 0.0


def test_the_turbulence_intensity_is_the_log_profile_s_shear_at_10_m() -> None:
    """DNV-RP-C205 2.3.2.10: 1 / ln(z / z0), with z0 the log profile's own."""
    h = 1e-4
    for speed in (3.0, 7.0, 15.0):
        above, below = (waves.wind_at_m(speed, 10.0 * math.exp(s)) for s in (h, -h))
        shear = (above - below) / (2 * h)
        assert waves.turbulence_intensity(speed) == pytest.approx(shear / speed)
    assert waves.turbulence_intensity(0.0) == 0.0


@pytest.mark.parametrize("speed", [3.0, 7.0, 15.0])
def test_a_gust_roughens_the_sea_as_cox_and_munk_s_wind_would(speed: float) -> None:
    """Linear in the gust, so the sea's mean roughness keeps its tuning."""
    per_gust = waves.gust_slope_variance(speed)
    for gust in 4 * waves.turbulence_intensity(speed) * np.array([-1.0, 1.0]):
        gusty = waves.cox_munk_slope(speed * (1 + gust)) ** 2
        offset = gusty - waves.cox_munk_slope(speed) ** 2
        assert per_gust * gust == pytest.approx(offset, rel=0.01)


def test_the_gust_field_has_kaimal_s_integral_scale() -> None:
    """One tile holds few of its largest eddies, so the scale is averaged over tiles."""
    scales_m = []
    for seed in range(8):
        field = waves.von_karman_field(
            np.random.default_rng(seed), 1024, 2.0, waves.GUST_LENGTH_M
        )
        assert (field.mean(), field.std()) == pytest.approx((0.0, 1.0), abs=1e-9)
        # Periodic, so the autocorrelation along a row is exact through the FFT.
        power = np.abs(np.fft.fft(field, axis=1)) ** 2
        correlation = np.fft.ifft(power, axis=1).real.mean(axis=0)
        correlation /= correlation[0]
        scales_m.append(np.trapezoid(correlation[: len(correlation) // 2], dx=2.0))
    assert np.mean(scales_m) == pytest.approx(waves.GUST_LENGTH_M, rel=0.15)


def test_a_slick_smooths_the_sea_as_cox_and_munk_measured() -> None:
    for speed in (3.0, 7.0, 15.0):
        wind = waves.wind_at_m(speed, waves.COX_MUNK_WIND_HEIGHT_M)
        slick = waves.SLICK_VARIANCE_INTERCEPT + waves.SLICK_VARIANCE_PER_MPS * wind
        assert waves.cox_munk_slick_slope(speed) ** 2 == pytest.approx(slick)
        assert waves.cox_munk_slick_slope(speed) < waves.cox_munk_slope(speed)
    # Below the fits' crossing, a slick is no rougher than clean water.
    assert waves.cox_munk_slick_slope(0.5) == waves.cox_munk_slope(0.5)


@pytest.mark.parametrize("wind_speed_mps", [0.5, 1.0, 1.5, 2.0, 8.0])
def test_a_slick_roughens_neither_axis(wind_speed_mps: float) -> None:
    slick = waves.cox_munk_slick_variances(wind_speed_mps)
    clean = waves.cox_munk_variances(wind_speed_mps)
    assert all(s <= c for s, c in zip(slick, clean, strict=True))
    total = waves.cox_munk_slick_slope(wind_speed_mps) ** 2
    assert sum(slick) == pytest.approx(total)


def test_a_slick_below_the_fits_crossing_is_the_clean_sea() -> None:
    slick = waves.cox_munk_slick_variances(0.0)
    assert slick == pytest.approx(waves.cox_munk_variances(0.0))


@pytest.mark.parametrize("wind_speed_mps", [7.0, 15.0])
def test_a_slick_damps_the_shortest_waves_first(wind_speed_mps: float) -> None:
    built = field(wind_speed_mps)
    budget = np.array(waves.cox_munk_slick_variances(wind_speed_mps))

    left = waves.slick_survivors(wind_speed_mps, built, 0.0)

    gone = set(built) - set(left)
    next_wave = min(gone, key=lambda w: w.k_rad_m)
    assert max(w.k_rad_m for w in left) < next_wave.k_rad_m
    # Each axis within Cox & Munk's slick fit, and one more wave breaks one.
    assert np.all(np.array(waves.drawn_slope_variances(left, 0.0, 0.0)) <= budget)
    assert np.any(
        np.array(waves.drawn_slope_variances((*left, next_wave), 0.0, 0.0)) > budget
    )


@pytest.mark.parametrize("cover", [0.1, 0.7])
def test_the_gust_tile_is_gaussian_so_its_tail_sets_the_slick_cover(
    cover: float,
) -> None:
    tile = waves.von_karman_field(np.random.default_rng(0), 512, 4.0, 50.0)
    threshold = NormalDist().inv_cdf(1 - cover)
    assert (tile > threshold).mean() == pytest.approx(cover, abs=0.05)


def test_a_pixel_that_resolves_less_emits_more_at_grazing() -> None:
    """Masuda 1988: unresolved slope lifts grazing emissivity off flat Fresnel."""
    built = field(7.0)
    # lwir takes the slope per axis: half the total variance.
    near, far = (
        math.sqrt(sum(waves.unresolved_slope_variances(7.0, 0.0, built, (), f)) / 2)
        for f in (0.01, 50)
    )
    eps = {}
    for name, sigma in (("near", near), ("far", far)):
        theta, curve = lwir.emissivity_curve(t_sea_k=291.0, slope_sigma=sigma)
        eps[name] = float(np.interp(math.radians(80.0), theta, curve))
        assert float(np.interp(0.0, theta, curve)) == pytest.approx(0.985, abs=0.005)
    assert eps["far"] > eps["near"]


@pytest.mark.parametrize("wind_speed_mps", [3.0, 7.0, 12.0, 20.0])
def test_a_fully_developed_sea_is_as_high_as_pierson_moskowitz_s(
    wind_speed_mps,
) -> None:
    # Pierson & Moskowitz 1964: Hs = 0.209 U(19.5 m)^2 / g. JONSWAP's peak enhancement
    # in Elfouhaily et al. 1997 adds some.
    hs = 4 * math.sqrt(sum(w.amplitude_m**2 / 2 for w in field(wind_speed_mps)))
    wind = waves.wind_at_m(wind_speed_mps, 19.5)
    assert hs == pytest.approx(0.209 * wind**2 / earth.GRAVITY_MS2, rel=0.12)


@pytest.mark.parametrize(
    ("wind_speed_mps", "fetch_km"),
    [(5.0, 10.0), (10.0, 3.0), (15.0, 30.0)],
)
def test_a_sea_short_of_fetch_is_as_high_as_jonswap_measured(
    wind_speed_mps: float, fetch_km: float
) -> None:
    """JONSWAP's fetch law holds only while the sea still grows."""
    # Hasselmann et al. 1973: energy 1.6e-7 (g F / U^2) U^4 / g^2, so
    # Hs = 1.6e-3 sqrt(g F / U^2) U^2 / g.
    young = waves.wind_sea(
        wind_speed_mps, 0.0, fetch_km * 1e3, lambda p: p, np.random.default_rng(0)
    )
    hs = 4 * math.sqrt(sum(w.amplitude_m**2 / 2 for w in young))
    g = earth.GRAVITY_MS2
    x = g * fetch_km * 1e3 / wind_speed_mps**2
    assert hs == pytest.approx(1.6e-3 * math.sqrt(x) * wind_speed_mps**2 / g, rel=0.1)


@pytest.mark.parametrize("wind_speed_mps", [5.0, 7.0, 15.0])
def test_the_spectrum_s_slope_is_cox_and_munk_s(wind_speed_mps: float) -> None:
    """Elfouhaily et al. fitted their short waves to Cox & Munk's clean sea."""
    k = np.geomspace(1e-4, 2e4, 200_000)
    mss = np.trapezoid(waves.curvature_spectrum(k, wind_speed_mps, None) / k, k)
    assert mss == pytest.approx(waves.cox_munk_slope(wind_speed_mps) ** 2, rel=0.08)


def test_a_light_wind_still_raises_a_sea() -> None:
    """Below u* = c_m / e, eq. 44's alpha_m turns negative."""
    built = field(1.0)

    assert built
    assert all(w.amplitude_m > 0.0 for w in built)


def test_wind_waves_run_within_a_right_angle_of_downwind_on_both_sides() -> None:
    built = sorted(field(7.0), key=lambda w: w.omega_rad_s)

    # A wind from the north runs its waves toward pi.
    off = np.array([math.remainder(w.toward_rad - math.pi, 2 * math.pi) for w in built])

    assert np.abs(off).max() <= math.pi / 2
    runs = off.reshape(-1, waves.SPREAD_STRATA)
    assert np.all((runs.min(axis=1) < 0.0) & (runs.max(axis=1) > 0.0))


def test_the_tallest_waves_are_the_fewest_carrying_the_share() -> None:
    share = 0.999
    built = field(7.0)
    total = sum(w.amplitude_m**2 for w in built)

    kept = waves.tallest(built, share)

    carried = sum(w.amplitude_m**2 for w in kept)
    smallest = min(w.amplitude_m**2 for w in kept)
    assert carried >= share * total > carried - smallest


def test_waves_run_away_from_the_wind() -> None:
    from_east = field(7.0, wind_from_deg=90.0)
    assert np.mean([w.k_east_rad_m for w in from_east]) < 0.0


def test_a_wind_off_every_axis_runs_its_sea_downwind() -> None:
    """From 60, off both axes, so a sign slip in east or north shows."""
    built = field(7.0, wind_from_deg=60.0)
    k_east = sum(w.k_east_rad_m * w.amplitude_m**2 for w in built)
    k_north = sum(w.k_north_rad_m * w.amplitude_m**2 for w in built)
    # The spread's random strata leave the energy-weighted mean a few degrees off.
    assert math.degrees(math.atan2(k_east, k_north)) == pytest.approx(-120.0, abs=5.0)


def test_a_crest_travels_at_its_phase_speed() -> None:
    wave = waves.Wave(1.0, 1.2, math.radians(30.0), 0.4)
    speed_ms = wave.omega_rad_s / wave.k_rad_m
    east, north = math.sin(wave.toward_rad), math.cos(wave.toward_rad)
    t_s = 2.5
    before = gerstner.height_m((wave,), np.array([3.0]), np.array([4.0]), 0.0)
    after = gerstner.height_m(
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
        None,
        lambda p: span_s / max(1, round(span_s / p)),
        np.random.default_rng(0),
    )
    east, north = np.meshgrid(np.linspace(-50, 50, 9), np.linspace(-50, 50, 9))
    assert gerstner.height_m(snapped, east, north, span_s) == pytest.approx(
        gerstner.height_m(snapped, east, north, 0.0), abs=1e-9
    )


def test_a_longer_loop_snaps_the_frequencies_less() -> None:
    def loop(span_s: float) -> float:
        return waves.snap_error(7.0, None, lambda p: span_s / max(1, round(span_s / p)))

    assert loop(90.0) < loop(40.0) < loop(10.0)
    assert waves.snap_error(7.0, None, lambda p: p) == pytest.approx(0.0, abs=1e-12)


def test_a_pixel_leaves_to_roughness_what_it_cannot_resolve() -> None:
    built = field(7.0)
    total = waves.cox_munk_slope(7.0) ** 2
    sharp = sum(waves.unresolved_slope_variances(7.0, 0.0, built, (), 1e-6))
    assert sharp == pytest.approx(total - waves.slope_variance(built))
    assert sum(
        waves.unresolved_slope_variances(7.0, 0.0, built, (), 1e4)
    ) == pytest.approx(total)
    footprints = [0.01, 0.1, 1.0, 10.0]
    rest = [
        sum(waves.unresolved_slope_variances(7.0, 0.0, built, (), f))
        for f in footprints
    ]
    assert rest == sorted(rest), "a coarser pixel leaves more"


def test_a_heading_band_is_the_closed_form_of_its_integrals() -> None:
    delta, low, high = 0.6, -0.4, 0.5
    theta = np.linspace(low, high, 20_001)
    weight = 1 + delta * np.cos(2 * theta)
    mean = np.trapezoid(weight * np.exp(2j * theta), theta) / np.trapezoid(
        weight, theta
    )
    concentration, axis = sea.heading_band(
        np.array([delta]), np.array([low]), np.array([high])
    )
    assert concentration[0] == pytest.approx(abs(mean), rel=1e-6)
    assert axis[0] == pytest.approx(np.angle(mean) / 2, rel=1e-6)


# A judgement: a grazing pixel, 0.5 m across and a hundred times that along.
GRAZING_ACROSS_M = 0.5
GRAZING_ALONG_M = 100 * GRAZING_ACROSS_M
# A judgement: crests this near the view's line count as running along it.
ALONG_VIEW_RAD = math.radians(1.0)
# A judgement: how far a draw of discrete waves may exceed a continuous spread.
SHARP_EXCESS = 3.0


def _faded_visibility(field: tuple[waves.Wave, ...], view_rad: float) -> np.ndarray:
    """Each wave's visibility to the grazing pixel at the footprint the shader's fade
    gives it.
    """
    axes = [waves.fade_axis(w) for w in field]
    tx = np.array([a.tx for a in axes])
    ty = np.array([a.ty for a in axes])
    band = np.array([a.band for a in axes])
    cos_sq = (tx * math.sin(view_rad) + ty * math.cos(view_rad)) ** 2 + band
    along_sq = GRAZING_ALONG_M**2 - GRAZING_ACROSS_M**2
    footprint = np.sqrt(GRAZING_ACROSS_M**2 + along_sq * cos_sq)
    wavelength = np.array([2 * math.pi / w.k_rad_m for w in field])
    return waves.visibility(wavelength, footprint)


def _slope_variance(built: tuple[waves.Wave, ...]) -> tuple[np.ndarray, np.ndarray]:
    k = np.array([w.k_rad_m for w in built])
    slope = (np.array([w.amplitude_m for w in built]) * k) ** 2 / 2
    return slope, waves.visibility(2 * math.pi / k, GRAZING_ACROSS_M) ** 2 * slope


def _sharp_along_view(built: tuple[waves.Wave, ...], view_rad: float) -> float:
    """The slope variance a grazing pixel keeps in crests along its view, over what
    a square pixel draws of the whole sea.
    """
    slope, square = _slope_variance(built)
    seen = _faded_visibility(built, view_rad)
    toward = np.array([w.toward_rad for w in built])
    crest_off = np.remainder(toward - view_rad, math.pi) - math.pi / 2
    near = np.abs(crest_off) < ALONG_VIEW_RAD
    return float((seen**2 * slope)[near].sum() / square.sum())


def _continuous_sharp_along_view(
    built: tuple[waves.Wave, ...], wind_speed_mps: float, view_rad: float
) -> float:
    """`_sharp_along_view` of `built`'s frequencies spread over Elfouhaily's
    headings continuously, the wind from north.
    """
    steps = 41
    edges = np.linspace(-ALONG_VIEW_RAD, ALONG_VIEW_RAD, steps + 1)
    downwind = math.pi
    # Crests along the view run their waves square to it, folded downwind.
    square_off = math.remainder(view_rad + math.pi / 2 - downwind, math.pi)
    slope, square = _slope_variance(built)
    omega = np.array([w.omega_rad_s for w in built])
    delta = waves.spreading_delta(omega, wind_speed_mps, None)
    kept = 0.0
    for low, high in zip(edges[:-1].tolist(), edges[1:].tolist(), strict=True):
        heading = downwind + square_off + (low + high) / 2
        share = sea.spread_share(delta, square_off + low, square_off + high)
        single = tuple(
            waves.Wave(w.amplitude_m, w.omega_rad_s, heading, 0.0) for w in built
        )
        seen = _faded_visibility(single, view_rad)
        kept += float(np.sum(slope * share * seen**2))
    return kept / float(square.sum())


@pytest.mark.parametrize("wind_speed_mps", [3.0, 12.0])
def test_no_draw_keeps_a_grazing_view_s_crests_sharper_than_a_continuous_sea(
    wind_speed_mps: float,
) -> None:
    for seed in range(10):
        built = field(wind_speed_mps, seed=seed)
        for view_rad in np.radians(np.arange(0.0, 91.0, 15.0)).tolist():
            bound = SHARP_EXCESS * _continuous_sharp_along_view(
                built, wind_speed_mps, view_rad
            )
            assert _sharp_along_view(built, view_rad) < bound, (seed, view_rad)


def test_a_wave_fades_as_a_box_pixel_passes_it() -> None:
    footprints = np.array([0.0, 0.5, 1.0, 1.5, 3.0])
    shown = waves.visibility(np.ones_like(footprints), footprints)
    assert shown[:2] == pytest.approx([1.0, 2 / math.pi])
    assert (shown[2:] == 0.0).all(), "gone at the first zero, side lobes and all"
    assert np.all(np.diff(shown) <= 0)


@pytest.mark.parametrize("wind_speed_mps", [0.0, 14.0])
def test_the_drawn_slope_variance_is_the_filtered_field_s(
    wind_speed_mps: float,
) -> None:
    swell = waves.swell(2.0, 10.0, 0.0, lambda p: p, np.random.default_rng(0))
    sea = field(wind_speed_mps) + swell
    for footprint_m in np.geomspace(1e-3, 1e4, 50):
        assert sum(waves.drawn_slope_variances(sea, footprint_m, 0.0)) == pytest.approx(
            waves.slope_variance(gerstner.filtered(sea, footprint_m)), rel=1e-12
        )


@pytest.mark.parametrize("footprint_m", [0.03, 0.5])
def test_a_wave_no_pixel_resolves_is_gone_at_every_coarser_one(
    footprint_m: float,
) -> None:
    sea = field(14.0)
    kept = waves.resolvable(sea, footprint_m)
    culled = [w for w in sea if w not in kept]
    assert culled, "nothing culled: the assertion below would pass on nothing"
    wavelengths = np.array([2 * math.pi / w.k_rad_m for w in culled])
    for coarser in np.geomspace(footprint_m, 1e4, 50):
        assert (waves.visibility(wavelengths, coarser) == 0.0).all()
    assert all(2 * math.pi / w.k_rad_m > footprint_m for w in kept)


def test_calm_air_builds_no_waves_and_nothing_they_drive() -> None:
    assert field(0.0) == ()
    assert sum(
        waves.unresolved_slope_variances(0.0, 0.0, (), (), 1.0)
    ) == pytest.approx(waves.cox_munk_slope(0.0) ** 2)
    assert waves.whitecap_fraction(0.0) == 0.0
    # Whitecap cover at a breath of wind, over no waves.
    fraction = waves.whitecap_fraction(0.5)
    assert waves.foam_memory_thresholds_g(
        (), (0.0, 0.0), 0.25, 1 / 12, (fraction,)
    ) == [math.inf]
    assert waves.attitude((), waves.Waterplane(0.0, 0.0, 0.0, 20.0, 5.0), 0.0) == (0, 0)
    assert waves.specular_cell_m2(()) == math.inf
    assert waves.twinkle_hz(()) == 0.0
    assert waves.mss_spread(0.0, None, 1.0) == 0.0


def _tall(wind_speed_mps: float) -> tuple[waves.Wave, ...]:
    """A wind sea's waves 4 m and longer, which never fold."""
    return tuple(w for w in field(wind_speed_mps) if 2 * math.pi / w.k_rad_m >= 4.0)


def test_the_water_over_a_point_moved_there_from_its_rest() -> None:
    tall = _tall(14.0)
    east, north = np.meshgrid(np.linspace(-20, 20, 9), np.linspace(-20, 20, 9))
    rest_east, rest_north = waves.gerstner_rest_m(tall, east, north, 2.1)
    moved_east, moved_north, _ = waves.gerstner_offset_m(
        tall, rest_east, rest_north, 2.1
    )
    np.testing.assert_allclose(rest_east + moved_east, east, atol=1e-9)
    np.testing.assert_allclose(rest_north + moved_north, north, atol=1e-9)


def test_a_gerstner_normal_is_square_to_its_surface() -> None:
    tall = _tall(14.0)
    east, north, step = np.array([3.0, -11.0]), np.array([7.0, 2.0]), 1e-5

    def surface(e: np.ndarray, n: np.ndarray) -> np.ndarray:
        moved = waves.gerstner_offset_m(tall, e, n, 0.7)
        return np.array([e + moved[0], n + moved[1], moved[2]])

    normal = np.array(gerstner.normal(tall, east, north, 0.7))
    along_east = surface(east + step, north) - surface(east - step, north)
    along_north = surface(east, north + step) - surface(east, north - step)
    for tangent in (along_east, along_north):
        unit = tangent / np.linalg.norm(tangent, axis=0)
        np.testing.assert_allclose((normal * unit).sum(axis=0), 0.0, atol=1e-8)
    assert (normal[2] > 0).all()


@pytest.mark.parametrize("elevation_deg", [-3.0, -40.0, 4.0])
def test_a_ray_meets_the_water_where_it_first_dips_under(elevation_deg: float) -> None:
    """Down at a grazing and a steep angle from 6 m, and up from just over the water,
    past crests.
    """
    tall = _tall(14.0)
    rays = 100
    rng = np.random.default_rng(3)
    east, north = rng.uniform(-500, 500, (2, rays))
    if elevation_deg < 0:
        up = np.full(rays, 6.0)
    else:
        under = waves.gerstner_rest_m(tall, east, north, 1.7)
        up = waves.gerstner_offset_m(tall, *under, 1.7)[2] + 0.05
    origin = np.vstack([east, north, up])
    bearing = rng.uniform(0, 2 * math.pi, rays)
    e = math.radians(elevation_deg)
    toward = np.array(
        [
            math.cos(e) * np.sin(bearing),
            math.cos(e) * np.cos(bearing),
            np.full(rays, math.sin(e)),
        ]
    )
    *rest, met = gerstner.hit(tall, origin, toward, 1.7, 0.1)
    # Every ray down meets the water; some up clear the crests.
    assert met.all() if elevation_deg < 0 else 0 < met.mean() < 1
    point = np.array([*rest, np.zeros(rays)]) + np.array(
        waves.gerstner_offset_m(tall, *rest, 1.7)
    )
    along = np.sum((point - origin) * toward, axis=0)
    # On the ray, to the micrometre the search halves to.
    off = point - origin - along * toward
    assert np.abs(off[:, met]).max() < 1e-5
    # And nothing on the ray before it is under the water.
    for share in np.linspace(0.0, 0.999, 50):
        at = origin + share * np.where(met, along, 0.0) * toward
        water = np.array(
            waves.gerstner_offset_m(
                tall, *waves.gerstner_rest_m(tall, at[0], at[1], 1.7), 1.7
            )
        )[2]
        assert (at[2] >= water - 1e-6).all()


def test_a_swell_carries_its_height_from_where_it_comes() -> None:
    swell = waves.swell(2.0, 12.0, 90.0, lambda p: p, np.random.default_rng(0))
    assert 4 * math.sqrt(sum(w.amplitude_m**2 / 2 for w in swell)) == pytest.approx(2.0)
    assert all(w.omega_rad_s == pytest.approx(2 * math.pi / 12.0) for w in swell)
    assert np.mean([w.k_east_rad_m for w in swell]) < 0.0


# A light wind on a coarse grid resolves few waves, none of them far past the edge.
@pytest.mark.parametrize(("wind_speed_mps", "cell_m"), [(18.0, 2.0), (4.0, 4.0)])
def test_a_foam_memory_covers_what_monahan_measured(
    wind_speed_mps: float, cell_m: float
) -> None:
    """Off the points its threshold was set on: three stretches of sea 200 cells and
    10 s across.
    """
    wind = field(wind_speed_mps, 300.0)
    drift = waves.foam_drift_mps(wind_speed_mps, math.radians(120.0))
    step = 1 / 12
    fraction = waves.whitecap_fraction(wind_speed_mps)
    threshold, *more = waves.foam_memory_thresholds_g(
        wind, drift, cell_m, step, (fraction, 0.3, 0.5)
    )
    # More cover breaks at a lower threshold.
    assert threshold > more[0] > more[1]
    covers = []
    for east, north, start in ((-4000, 1000, 50), (2500, -3500, 300), (900, 4200, 520)):
        grid = waves.FoamGrid(east, north, cell_m, 200, 200)
        frames = waves.foam_memory(wind, drift, grid, start, step, 120, threshold)
        covers.extend(cover.mean() for cover, _ in frames)
    assert float(np.mean(covers)) == pytest.approx(fraction, rel=0.15)


def test_remembered_foam_fades_at_monahan_s_e_fold() -> None:
    step = 1 / 12
    grid = waves.FoamGrid(0.0, 0.0, 0.5, 64, 64)
    frames = waves.foam_memory(field(13.0), (0.39, 0.0), grid, 0.0, step, 60, 0.3)
    cover, breaking = (np.array(a) for a in zip(*frames, strict=True))
    fading = (breaking[1:] < 1e-6) & (cover[:-1] > 0.1)
    assert fading.any()
    decay = math.exp(-step / waves.WHITECAP_AREA_EFOLD_S)
    np.testing.assert_allclose(cover[1:][fading], cover[:-1][fading] * decay, rtol=1e-5)


def test_whitecaps_read_the_wind_at_10_m() -> None:
    # Monahan & O'Muircheartaigh 1980: 3.84e-6 U10^3.41, 0.99% at 10 m/s.
    assert waves.whitecap_fraction(10.0) == pytest.approx(0.0099, abs=1e-4)


def test_remembered_whitecaps_average_koepke_s_reflectance() -> None:
    """Fresh foam's reflectance while its crest breaks, thin foam's share of it after;
    averaged over the foam, on the finest grid a camera reads, Koepke's 0.22, which he
    gives to half.
    """
    wind_speed_mps = 13.0
    wind = field(wind_speed_mps, 300.0)
    drift = waves.foam_drift_mps(wind_speed_mps, math.radians(120.0))
    step = 1 / 12
    fraction = waves.whitecap_fraction(wind_speed_mps)
    (threshold,) = waves.foam_memory_thresholds_g(wind, drift, 0.25, step, (fraction,))
    grid = waves.FoamGrid(0.0, 0.0, 0.25, 100, 100)
    covered = shown = 0.0
    thin = waves.THIN_FOAM_SHARE
    for cover, breaking in waves.foam_memory(
        wind, drift, grid, 0.0, step, 60, threshold
    ):
        covered += cover.sum()
        shown += (cover * (thin + (1 - thin) * breaking)).sum()
    mean = FRESH_FOAM_REFLECTANCE * shown / covered
    assert mean == pytest.approx(waves.WHITECAP_EFFECTIVE_REFLECTANCE, rel=0.5)
    # Thin foam alone is inside Koepke's half; breaking foam must brighten it.
    assert mean > FRESH_FOAM_REFLECTANCE * thin


def test_a_foam_memory_covers_none_of_the_sea_or_all_of_it() -> None:
    thresholds = waves.foam_memory_thresholds_g(
        field(13.0), (0.0, 0.0), 2.0, 1 / 12, (0.0, 1.0)
    )
    assert list(thresholds) == [math.inf, -math.inf]


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
    hull = waves.Waterplane(east, north, math.radians(heading_deg), 20.0, 5.0)
    pitch, roll = waves.attitude((wave,), hull, t_s)
    d_east, d_north = gerstner.slope((wave,), np.array(east), np.array(north), t_s)
    heading = math.radians(heading_deg)
    rise_along = d_east * math.sin(heading) + d_north * math.cos(heading)
    rise_starboard = d_east * math.cos(heading) - d_north * math.sin(heading)
    # Water rising to starboard rolls the hull port down, a negative roll.
    assert (pitch, roll) == pytest.approx(
        (math.atan(rise_along), -math.atan(rise_starboard)), abs=1e-5
    )
    assert abs({"pitch": pitch, "roll": roll}[expected]) > 1e-4


def test_an_oblique_wave_meets_an_oblique_hull_ahead_and_from_port() -> None:
    """Toward 30 on a heading of 75: 45 to port of the bow."""
    wave = waves.Wave(0.5, 0.2, math.radians(30.0), 1.0)
    hull = waves.Waterplane(heading_rad=math.radians(75.0))
    along, across = surface._along_across((wave,), hull)
    k = wave.k_rad_m * math.sqrt(0.5)
    assert (along[0], across[0]) == pytest.approx((k, -k))


@pytest.mark.parametrize("length_m", [0.0, 20.0, 90.0])
def test_a_hull_floats_at_the_mean_of_the_water_under_it(length_m: float) -> None:
    """Over a low wave, as linear: a cos(phase) sinc(k L / 2) along it."""
    wave = waves.Wave(
        1e-3, math.sqrt(earth.GRAVITY_MS2 * 2 * math.pi / 60.0), math.radians(30.0), 0.7
    )
    east, north, t_s = 12.0, -7.0, 2.5
    hull = waves.Waterplane(east, north, math.radians(30.0), length_m)
    heave = waves.heave_m((wave,), hull, t_s)
    phase = wave.k_east_rad_m * east + wave.k_north_rad_m * north
    phase -= wave.omega_rad_s * t_s - wave.phase_rad
    mean = np.sinc(wave.k_rad_m * length_m / 2 / math.pi)
    assert heave == pytest.approx(wave.amplitude_m * math.cos(phase) * mean, rel=1e-3)


def test_a_point_floats_on_the_gerstner_surface() -> None:
    tall = _tall(14.0)
    rest = waves.gerstner_rest_m(tall, np.array(30.0), np.array(-40.0), 3.0)
    water = waves.gerstner_offset_m(tall, *rest, 3.0)[2]
    heave = waves.heave_m(tall, waves.Waterplane(*map(float, rest), 1.0), 3.0)
    assert heave == pytest.approx(water)


def test_a_hull_pitches_with_the_water_under_it() -> None:
    """At rest phase pi/2 a Gerstner wave's slope is exactly -ak, as the linear one."""
    k = 2 * math.pi / 60.0
    wave = waves.Wave(0.2 / k, math.sqrt(earth.GRAVITY_MS2 * k), 0.0, math.pi / 2)
    # Over the water at rest at the origin, one amplitude back against the wave.
    hull = np.zeros(()), np.array(-wave.amplitude_m)
    rest = waves.gerstner_rest_m((wave,), *hull, 0.0)
    assert np.allclose(rest, 0.0)
    pitch, _ = waves.attitude((wave,), waves.Waterplane(*map(float, rest)), 0.0)
    assert pitch == pytest.approx(math.atan(-0.2))


def test_a_hull_rides_over_a_wave_much_shorter_than_itself() -> None:
    ripple = waves.Wave(0.05, math.sqrt(earth.GRAVITY_MS2 * 2 * math.pi), 0.3, 1.0)
    pitch, roll = waves.attitude(
        (ripple,), waves.Waterplane(3.0, 4.0, 0.0, 20.0, 5.0), 1.0
    )
    own_slope = ripple.amplitude_m * ripple.k_rad_m
    assert max(abs(pitch), abs(roll)) < 0.01 * own_slope


@pytest.mark.parametrize(
    ("omega_rad_s", "length_m", "beam_m"),
    [
        (0.6, 60.0, 10.0),  # 171 m long
        (1.2, 60.0, 10.0),  # 43 m: the hull's lever turns over
        (3.0, 10.0, 10.0),  # 6.8 m: a square turns once the wave bends over it
    ],
)
def test_a_hull_yaws_with_the_water_under_it(
    omega_rad_s: float, length_m: float, beam_m: float
) -> None:
    """The least-squares turn of the Gerstner offset, integrated over a 301 x 101
    grid.
    """
    wave = waves.Wave(
        amplitude_m=0.8 * (0.6 / omega_rad_s) ** 2,
        omega_rad_s=omega_rad_s,
        toward_rad=0.7,
        phase_rad=1.0,
    )
    hull = waves.Waterplane(
        rest_east_m=30.0,
        rest_north_m=-40.0,
        heading_rad=math.radians(100.0),
        length_m=length_m,
        beam_m=beam_m,
    )
    x, y = np.meshgrid(
        np.linspace(-length_m / 2, length_m / 2, 301),
        np.linspace(-beam_m / 2, beam_m / 2, 101),
        indexing="ij",
    )
    sin, cos = math.sin(hull.heading_rad), math.cos(hull.heading_rad)
    east, north, _ = waves.gerstner_offset_m(
        (wave,), 30.0 + x * sin + y * cos, -40.0 + x * cos - y * sin, 2.0
    )
    along, starboard = east * sin + north * cos, east * cos - north * sin

    def integral(f: np.ndarray) -> float:
        return float(np.trapezoid(np.trapezoid(f, axis=1), axis=0))

    # atan2 takes a ratio, so the integrals stand in for the means.
    turn = integral(x * starboard - y * along)
    yaw = waves.yaw_rad((wave,), hull, 2.0)
    assert yaw == pytest.approx(
        math.atan2(turn, integral(x * x + y * y + x * along + y * starboard)), abs=1e-6
    )
    assert abs(yaw) > 1e-4, "an unturned hull would pass vacuously"


def test_a_square_barely_yaws_in_a_long_wave() -> None:
    """The offset is a gradient: over a square, its turn falls with (k L)^2."""
    wave = waves.Wave(amplitude_m=0.8, omega_rad_s=0.6, toward_rad=0.7, phase_rad=1.0)
    square = waves.Waterplane(
        rest_east_m=30.0,
        rest_north_m=-40.0,
        heading_rad=1.7,
        length_m=10.0,
        beam_m=10.0,
    )
    long = square._replace(length_m=60.0)
    # Measured 5e-4, so 1e-3 is a margin over it; a square that turned with the water
    # as the long hull does would sit near 1.
    assert abs(waves.yaw_rad((wave,), square, 2.0)) < 1e-3 * abs(
        waves.yaw_rad((wave,), long, 2.0)
    )


def _jensen(wave: waves.Wave, hull: waves.Waterplane) -> tuple[float, float]:
    """Gain and phase lead of heave and pitch over the hull's static ones: Jensen,
    Mansour & Olsen (2004) eqs. 2.3-2.5, 2.9, 2.10, 2.13 and 4.4, at the frequency
    of encounter's magnitude.
    """
    k, omega = wave.k_rad_m, wave.omega_rad_s
    met = (
        omega - wave.k_east_rad_m * hull.east_mps - wave.k_north_rad_m * hull.north_mps
    )
    alpha, b, t = abs(met) / omega, hull.box_beam_m, hull.box_draught_m
    a = 2 * math.sin(k * b * alpha**2 / 2) * math.exp(-k * t * alpha**2)
    k_along = wave.k_east_rad_m * math.sin(hull.heading_rad) + (
        wave.k_north_rad_m * math.cos(hull.heading_rad)
    )
    smith = math.exp(-abs(k_along) * t)
    excite = complex(1 - k * t, a**2 / (k * b * alpha**3))
    respond = complex(1 - 2 * k * t * alpha**2, -(a**2) / (k * b * alpha**2))
    lead = math.atan2(excite.imag, excite.real) + math.atan2(respond.imag, respond.real)
    return smith * abs(excite) / abs(respond), lead


def _heave(wave: waves.Wave, hull: waves.Waterplane, t_s: float) -> float:
    return waves.heave_m((wave,), hull, t_s)


def _pitch_slope(wave: waves.Wave, hull: waves.Waterplane, t_s: float) -> float:
    return math.tan(waves.attitude((wave,), hull, t_s)[0])


@pytest.mark.parametrize("motion", [_heave, _pitch_slope])
@pytest.mark.parametrize(
    ("toward_deg", "speed_mps", "omega_rad_s"),
    [
        pytest.param(180.0, 18.1, 0.8, id="head"),
        pytest.param(150.0, 18.1, 1.3, id="bow-quartering"),
        pytest.param(20.0, 18.1, 1.2, id="overtaking-following"),
    ],
)
def test_a_hull_moves_as_jensen_s_box_its_static_ride_a_lead_later(
    motion: Callable[[waves.Wave, waves.Waterplane, float], float],
    toward_deg: float,
    speed_mps: float,
    omega_rad_s: float,
) -> None:
    """TMV114 (Jensen et al. 2004, 3.1), sailing north."""
    wave = waves.Wave(0.01, omega_rad_s, math.radians(toward_deg), 0.4)

    def hull(t_s: float) -> waves.Waterplane:
        return waves.Waterplane(
            0.0,
            speed_mps * t_s,
            0.0,
            length_m=96.0,
            beam_m=13.8,
            box_beam_m=13.8 * 0.53,
            box_draught_m=2.5,
            north_mps=speed_mps,
        )

    def still(t_s: float) -> waves.Waterplane:
        """The waterplane alone, with no box."""
        return hull(t_s)._replace(box_beam_m=0.0, box_draught_m=0.0)

    gain, lead = _jensen(wave, hull(0.0))
    met = abs(omega_rad_s - wave.k_north_rad_m * speed_mps)
    times = np.linspace(0.0, 2 * math.pi / met, 7)

    moved = [motion(wave, hull(t), t) for t in times]
    static = [motion(wave, still(t + lead / met), t + lead / met) for t in times]

    assert moved == pytest.approx(gain * np.array(static), abs=1e-9)


def test_a_deep_hull_feels_a_short_wave_less_than_the_water_s_slope() -> None:
    wave = waves.Wave(0.01, 1.5, math.radians(180.0), 0.0)
    times = np.linspace(0.0, 2 * math.pi / 1.5, 32)

    shallow, deep = (
        max(
            abs(waves.attitude((wave,), waves.Waterplane(0, 0, 0, 20, 5, 5, d), t)[0])
            for t in times
        )
        for d in (0.0, 6.0)
    )

    assert deep < 0.5 * shallow


def test_the_closed_form_is_the_least_squares_plane_under_the_hull() -> None:
    sea = field(12.0, wind_from_deg=40.0)
    east, north, heading, length, beam, t = 30.0, -15.0, 0.7, 40.0, 9.0, 2.0
    along, across = np.meshgrid(
        (np.arange(801) + 0.5) / 801 * length - length / 2,
        (np.arange(181) + 0.5) / 181 * beam - beam / 2,
    )
    along, across = along.ravel(), across.ravel()
    # The linear field's plane, centred where the water under the hull was at rest.
    rest_east, rest_north = waves.gerstner_rest_m(
        sea, np.array(east), np.array(north), t
    )
    x = rest_east + along * math.sin(heading) + across * math.cos(heading)
    y = rest_north + along * math.cos(heading) - across * math.sin(heading)
    plane = np.column_stack([np.ones_like(along), along, across])
    _, rise_along, rise_across = np.linalg.lstsq(
        plane, gerstner.height_m(sea, x, y, t), rcond=None
    )[0]
    hull = waves.Waterplane(float(rest_east), float(rest_north), heading, length, beam)
    pitch, roll = waves.attitude(sea, hull, t)
    assert (pitch, roll) == pytest.approx(
        (math.atan(rise_along), -math.atan(rise_across)), abs=1e-5
    )


def test_a_swell_adds_its_own_unresolved_slope_to_cox_and_munk_s() -> None:
    wind = field(7.0)
    swell = waves.swell(2.0, 12.0, 90.0, lambda p: p, np.random.default_rng(0))
    far = sum(waves.unresolved_slope_variances(7.0, 0.0, wind, swell, 1e4))
    assert far == pytest.approx(
        waves.cox_munk_slope(7.0) ** 2 + waves.slope_variance(swell)
    )
    near = sum(waves.unresolved_slope_variances(7.0, 0.0, wind, swell, 1e-3))
    assert near == pytest.approx(
        sum(waves.unresolved_slope_variances(7.0, 0.0, wind, (), 1e-3))
    )


def test_an_isotropic_gaussian_surface_has_e_det_h_of_m4_over_2_sqrt_3() -> None:
    # h_xx, h_yy, h_xy of an isotropic surface of m4 = 1.
    cov = np.array([[3.0, 1.0, 0.0], [1.0, 3.0, 0.0], [0.0, 0.0, 1.0]]) / 8
    h = np.random.default_rng(0).multivariate_normal(np.zeros(3), cov, 1_000_000)
    det = np.abs(h[:, 0] * h[:, 1] - h[:, 2] ** 2).mean()
    assert det == pytest.approx(1 / (2 * math.sqrt(3)), rel=0.01)


@pytest.mark.parametrize("wind_speed_mps", [3.0, 7.0, 15.0])
def test_glitter_follows_the_spectrum_s_tail_to_the_capillary_cutoff(
    wind_speed_mps,
) -> None:
    # m4 = int k^4 S dk = int B k dk; the twinkle is its mean omega^2 = g k.
    cutoff = waves.CAPILLARY_PEAK_RAD_M
    k = np.geomspace(
        waves.peak_omega_rad_s(wind_speed_mps, None) ** 2 / earth.GRAVITY_MS2,
        cutoff,
        100_000,
    )
    weight = waves.curvature_spectrum(k, wind_speed_mps, None) * k
    m4 = np.trapezoid(weight, k)
    omega_sq = np.trapezoid(weight * earth.GRAVITY_MS2 * k, k) / m4
    _, _, variance = waves.sea._bins(wind_speed_mps, None)
    cells, twinkles = [], []
    # Each wave's frequency is drawn in its bin, so the seeds average.
    for seed in range(8):
        sea = field(wind_speed_mps, seed=seed)
        # A light wind's sea is held to Cox & Munk's slope, every wave alike.
        held = sum(w.amplitude_m**2 / 2 for w in sea) / variance.sum()
        cells.append(waves.specular_cell_m2(sea) * held)
        twinkles.append(waves.twinkle_hz(sea))
    assert np.mean(cells) == pytest.approx(2 * math.sqrt(3) / m4, rel=0.1)
    assert np.mean(twinkles) == pytest.approx(
        math.sqrt(omega_sq) / (2 * math.pi), rel=0.1
    )


def test_the_spread_narrows_to_the_peak_and_widens_above_it() -> None:
    peak = waves.peak_omega_rad_s(7.0, None)
    delta = waves.spreading_delta(np.array([peak, 4 * peak]), 7.0, None)
    assert delta[0] == pytest.approx(1.0, abs=1e-3)
    assert delta[1] < 0.5


@pytest.mark.parametrize("wind_speed_mps", [5.0, 10.0])
def test_the_drawn_slope_is_upwind_heavy(wind_speed_mps: float) -> None:
    ratios = [
        np.divide(*waves.drawn_slope_variances(field(wind_speed_mps, seed=seed), 0, 0))
        for seed in range(8)
    ]

    assert 1.2 < np.mean(ratios) < 2.0


def test_drawn_and_unresolved_hold_cox_and_munk_s_slope_on_each_axis() -> None:
    built = field(7.0)
    drawn = waves.drawn_slope_variances(built, 0.05, 0.0)
    left = waves.unresolved_slope_variances(7.0, 0.0, built, (), 0.05)

    assert np.add(drawn, left) == pytest.approx(waves.cox_munk_variances(7.0))


@pytest.mark.parametrize(
    ("wind_speed_mps", "upwind_heavier"),
    # Cox & Munk's fits cross at 2.4 m/s.
    [(0.0, False), (2.0, False), (3.0, True), (15.0, True)],
)
def test_the_axes_split_cox_and_munk_s_total(
    wind_speed_mps: float, upwind_heavier: bool
) -> None:
    up, cross = waves.cox_munk_variances(wind_speed_mps)

    assert up + cross == pytest.approx(waves.cox_munk_slope(wind_speed_mps) ** 2)
    assert (up > cross) == upwind_heavier


@pytest.mark.parametrize("wind_speed_mps", [1.0, 2.0, 3.0, 7.0, 15.0])
def test_no_axis_draws_more_slope_than_cox_and_munk_measured(
    wind_speed_mps: float,
) -> None:
    drawn = waves.drawn_slope_variances(field(wind_speed_mps), 0.0, 0.0)

    assert np.all(
        np.asarray(drawn)
        <= np.asarray(waves.cox_munk_variances(wind_speed_mps)) * (1 + 1e-9)
    )


def test_a_sea_within_cox_and_munk_s_slope_keeps_its_spectrum_s_height() -> None:
    """Elfouhaily's height variance is the bins'; the hold only trims a light wind."""
    _, _, variance = waves.sea._bins(7.0, None)
    built = field(7.0)
    assert sum(w.amplitude_m**2 / 2 for w in built) == pytest.approx(variance.sum())


def test_each_wave_carries_its_own_bin_s_height_off_the_bin_s_centre() -> None:
    edges, _, variance = waves.sea._bins(7.0, None)
    built = sorted(field(7.0), key=lambda w: w.omega_rad_s)
    omega = np.array([w.omega_rad_s for w in built])
    assert np.all((edges[:-1] < omega) & (omega < edges[1:]))
    # On the centres every log step is one bin wide; drawn, they spread over two.
    spacing_spread = np.ptp(np.diff(np.log(omega))) / np.log(edges[1] / edges[0])
    assert spacing_spread > 0.5
    assert [w.amplitude_m**2 / 2 for w in built] == pytest.approx(variance)


def test_a_tile_ranked_onto_a_gaussian_keeps_its_pattern_and_loses_its_skew() -> None:
    tile = np.random.default_rng(0).gamma(2.0, size=(64, 64))
    ranked = waves.gaussian_ranks(tile)
    assert np.array_equal(np.argsort(ranked, axis=None), np.argsort(tile, axis=None))
    assert np.clip(ranked, -1.0, 1.0).mean() == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("share", [0.1, 0.7])
def test_a_foam_lattice_above_its_quantile_covers_that_share(share: float) -> None:
    """The shader keeps `share` of aged foam's area by thresholding the lattice at its
    1 - share quantile; ranked onto a Gaussian, that is exact.
    """
    tile = waves.foam_lattice_field(np.random.default_rng(0), 128, 4.0, (0.5,), 0.6)
    threshold = NormalDist().inv_cdf(1 - share)
    assert (tile > threshold).mean() == pytest.approx(share, abs=1 / tile.size)


def test_thinning_foam_keeps_to_its_cell_walls() -> None:
    """The texels left at the lowest share lie nearer a wall than the tile's average:
    aged foam stays a network as it thins.
    """
    tile = waves.foam_lattice_field(np.random.default_rng(0), 128, 4.0, (0.5,), 0.6)
    walls = waves.foam._cell_walls(np.random.default_rng(0), 128, 8)
    kept = tile > np.quantile(tile, 0.85)
    assert walls[kept].mean() < 0.5 * walls.mean()


def test_the_footprint_s_slope_spread_is_a_gaussian_sea_s() -> None:
    """Isserlis's spread against a Gaussian sea laid on a grid, whose footprints'
    mean |s|^2 is taken over its cells; the mean is the slope spectrum's integral,
    which reaches past the grid's cells.
    """
    speed, footprint_m, cells, cell_m = 7.0, 0.1, 1024, 0.01
    along = footprint_m * math.sqrt(waves.SPREAD_ASPECT)
    across = footprint_m / math.sqrt(waves.SPREAD_ASPECT)

    def kept(k: np.ndarray, off_view: np.ndarray) -> np.ndarray:
        seen_m = np.hypot(across, along * np.cos(off_view))
        return 1 - waves.visibility(2 * math.pi / k, seen_m) ** 2

    axis = np.fft.fftfreq(cells, cell_m) * 2 * math.pi
    k_across, k_along = np.meshgrid(axis, axis)
    k = np.hypot(k_across, k_along)
    k[0, 0] = 1.0
    # Isotropic: height variance per wavenumber area is B(k) k^-4 / 2 pi.
    density = waves.curvature_spectrum(k, speed, None) * k**-4 / (2 * math.pi)
    density *= kept(k, np.arctan2(k_across, k_along))
    density[0, 0] = 0.0
    rng = np.random.default_rng(0)
    # Each cell of wavenumber holds density dk^2, half of it in the real part taken;
    # ifft2 divides by cells^2.
    height = (
        (rng.standard_normal(k.shape) + 1j * rng.standard_normal(k.shape))
        * np.sqrt(density)
        * (axis[1] * cells**2)
    )
    slope_across, slope_along = (
        np.real(np.fft.ifft2(1j * ka * height)) for ka in (k_across, k_along)
    )
    square = slope_across**2 + slope_along**2
    rows, columns = round(along / cell_m), round(across / cell_m)
    trimmed = square[: cells // rows * rows, : cells // columns * columns]
    footprints = trimmed.reshape(cells // rows, rows, -1, columns).mean(axis=(1, 3))

    log_k = np.linspace(math.log(1.0), math.log(10 * waves.CAPILLARY_PEAK_RAD_M), 4000)
    off_view = np.linspace(0, 2 * math.pi, 361)
    wavenumber = np.exp(log_k)[:, None]
    per_log_k = waves.curvature_spectrum(wavenumber, speed, None) * kept(
        wavenumber, off_view[None, :]
    )
    mean = np.trapezoid(np.trapezoid(per_log_k, off_view, axis=1), log_k) / (
        2 * math.pi
    )

    assert footprints.std() / mean == pytest.approx(
        waves.mss_spread(speed, None, footprint_m), rel=0.05
    )
