import math
from collections import defaultdict
from statistics import NormalDist

import numpy as np
import pytest

from seascape import lwir, waves
from seascape.wakes import FRESH_FOAM_REFLECTANCE


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
    assert waves.horizon_m(51.8, 0.13) == pytest.approx(rule_m, rel=0.01)
    # Elfouhaily et al. 1997: a fully developed sea at 7 m/s peaks at
    # 2 pi U^2 / (g 0.84^2), 44.5 m.
    peak = waves.peak_omega_rad_s(7.0, None)
    assert 2 * math.pi * waves.GRAVITY_MS2 / peak**2 == pytest.approx(44.5, abs=0.2)


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


@pytest.mark.parametrize("wind_speed_mps", [7.0, 15.0])
def test_a_slick_damps_the_shortest_waves_first(wind_speed_mps: float) -> None:
    built = field(wind_speed_mps)
    budget = np.array(waves.cox_munk_slick_variances(wind_speed_mps))

    left = waves.slick_survivors(wind_speed_mps, built, 0.0)

    gone = set(built) - set(left)
    k = min(w.k_rad_m for w in gone)
    next_frequency = tuple(w for w in gone if w.k_rad_m == k)
    assert max(w.k_rad_m for w in left) < k
    assert len(next_frequency) == waves.WIND_HEADINGS
    # Each axis within Cox & Munk's slick fit, and one more wavelength breaks one.
    assert np.all(np.array(waves.drawn_slope_variances(left, 0.0, 0.0)) <= budget)
    assert np.any(
        np.array(waves.drawn_slope_variances((*left, *next_frequency), 0.0, 0.0))
        > budget
    )


@pytest.mark.parametrize("cover", [0.1, 0.3, 0.7])
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
def test_a_fully_developed_sea_is_as_high_as_pierson_moskowitz_s(
    wind_speed_mps,
) -> None:
    # Pierson & Moskowitz 1964: Hs = 0.209 U(19.5 m)^2 / g. JONSWAP's peak enhancement
    # in Elfouhaily et al. 1997 adds some.
    hs = 4 * math.sqrt(sum(w.amplitude_m**2 / 2 for w in field(wind_speed_mps)))
    wind = waves.wind_at_m(wind_speed_mps, 19.5)
    assert hs == pytest.approx(0.209 * wind**2 / waves.GRAVITY_MS2, rel=0.12)


@pytest.mark.parametrize(
    ("wind_speed_mps", "fetch_km"),
    [(5.0, 3.0), (5.0, 10.0), (10.0, 3.0), (10.0, 10.0), (10.0, 30.0), (15.0, 30.0)],
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
    g = waves.GRAVITY_MS2
    x = g * fetch_km * 1e3 / wind_speed_mps**2
    assert hs == pytest.approx(1.6e-3 * math.sqrt(x) * wind_speed_mps**2 / g, rel=0.1)


@pytest.mark.parametrize("wind_speed_mps", [5.0, 7.0, 10.0, 15.0])
def test_the_spectrum_s_slope_is_cox_and_munk_s(wind_speed_mps: float) -> None:
    """Elfouhaily et al. fitted their short waves to Cox & Munk's clean sea."""
    k = np.geomspace(1e-4, 2e4, 200_000)
    mss = np.trapezoid(waves.curvature_spectrum(k, wind_speed_mps, None) / k, k)
    assert mss == pytest.approx(waves.cox_munk_slope(wind_speed_mps) ** 2, rel=0.08)


@pytest.mark.parametrize("wind_speed_mps", [1.0, 2.0])
def test_a_light_wind_still_raises_a_sea(wind_speed_mps: float) -> None:
    """Below u* = c_m / e, eq. 44's alpha_m turns negative."""
    built = field(wind_speed_mps)

    assert built
    assert all(w.amplitude_m > 0.0 for w in built)


def test_each_frequency_runs_both_sides_of_the_wind() -> None:
    built = field(7.0)

    # A wind from the north runs its waves toward pi.
    off = defaultdict(list)
    for wave in built:
        off[wave.omega_rad_s].append(
            math.remainder(wave.toward_rad - math.pi, 2 * math.pi)
        )

    assert all(min(o) < 0.0 < max(o) for o in off.values())


@pytest.mark.parametrize("share", [0.9, 0.999])
def test_the_tallest_waves_are_the_fewest_carrying_the_share(share: float) -> None:
    built = field(7.0)
    total = sum(w.amplitude_m**2 for w in built)

    kept = waves.tallest(built, share)

    carried = sum(w.amplitude_m**2 for w in kept)
    smallest = min(w.amplitude_m**2 for w in kept)
    assert carried >= share * total > carried - smallest


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
        None,
        lambda p: span_s / max(1, round(span_s / p)),
        np.random.default_rng(0),
    )
    east, north = np.meshgrid(np.linspace(-50, 50, 9), np.linspace(-50, 50, 9))
    assert waves.height_m(snapped, east, north, span_s) == pytest.approx(
        waves.height_m(snapped, east, north, 0.0), abs=1e-9
    )


def test_a_longer_loop_snaps_the_frequencies_less() -> None:
    def loop(span_s: float) -> float:
        return waves.snap_error(7.0, None, lambda p: span_s / max(1, round(span_s / p)))

    assert loop(90.0) < loop(40.0) < loop(10.0)
    assert waves.snap_error(7.0, None, lambda p: p) == pytest.approx(0.0, abs=1e-12)


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


@pytest.mark.parametrize("wind_speed_mps", [0.0, 14.0])
def test_the_drawn_slope_variance_is_the_filtered_field_s(
    wind_speed_mps: float,
) -> None:
    swell = waves.swell(2.0, 10.0, 0.0, lambda p: p, np.random.default_rng(0))
    sea = field(wind_speed_mps) + swell
    for footprint_m in np.geomspace(1e-3, 1e4, 50):
        assert waves.drawn_slope_variance(sea, footprint_m) == pytest.approx(
            waves.slope_variance(waves.filtered(sea, footprint_m)), rel=1e-12
        )


@pytest.mark.parametrize("footprint_m", [0.02, 0.05, 0.5])
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
    assert all(math.pi / w.k_rad_m > footprint_m for w in kept)


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


# A light wind on a coarse grid resolves few waves, none of them far past the edge.
@pytest.mark.parametrize(
    ("wind_speed_mps", "cell_m"), [(8.0, 2.0), (18.0, 2.0), (4.0, 4.0)]
)
def test_a_foam_memory_covers_what_monahan_measured(
    wind_speed_mps: float, cell_m: float
) -> None:
    """Off the points its threshold was set on: three stretches of sea 200 cells and
    10 s across."""
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


def test_calm_air_breaks_nothing() -> None:
    assert waves.whitecap_fraction(0.0) == 0.0
    # A breath of wind builds no waves but has a whitecap fraction.
    fraction = waves.whitecap_fraction(0.5)
    assert waves.foam_memory_thresholds_g(
        (), (0.0, 0.0), 0.25, 1 / 12, (fraction,)
    ) == [math.inf]


def test_remembered_whitecaps_average_koepke_s_reflectance() -> None:
    """Fresh foam's reflectance while its crest breaks, thin foam's share of it after;
    averaged over the foam, on the finest grid a camera reads, Koepke's 0.22, which he
    gives to half."""
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


def test_a_hull_rides_over_a_wave_much_shorter_than_itself() -> None:
    ripple = waves.Wave(0.05, math.sqrt(waves.GRAVITY_MS2 * 2 * math.pi), 0.3, 1.0)
    pitch, roll = waves.attitude((ripple,), 3.0, 4.0, 0.0, 20.0, 5.0, 1.0)
    own_slope = ripple.amplitude_m * ripple.k_rad_m
    assert max(abs(pitch), abs(roll)) < 0.01 * own_slope


def test_the_closed_form_is_the_least_squares_plane_under_the_hull() -> None:
    sea = field(12.0, wind_from_deg=40.0)
    east, north, heading, length, beam, t = 30.0, -15.0, 0.7, 40.0, 9.0, 2.0
    along, across = np.meshgrid(
        (np.arange(801) + 0.5) / 801 * length - length / 2,
        (np.arange(181) + 0.5) / 181 * beam - beam / 2,
    )
    along, across = along.ravel(), across.ravel()
    x = east + along * math.sin(heading) + across * math.cos(heading)
    y = north + along * math.cos(heading) - across * math.sin(heading)
    plane = np.column_stack([np.ones_like(along), along, across])
    _, rise_along, rise_across = np.linalg.lstsq(
        plane, waves.height_m(sea, x, y, t), rcond=None
    )[0]
    pitch, roll = waves.attitude(sea, east, north, heading, length, beam, t)
    assert (pitch, roll) == pytest.approx(
        (math.atan(rise_along), math.atan(rise_across)), abs=1e-5
    )


def test_a_swell_adds_its_own_unresolved_slope_to_cox_and_munk_s() -> None:
    wind = field(7.0)
    swell = waves.swell(2.0, 12.0, 90.0, lambda p: p, np.random.default_rng(0))
    far = waves.unresolved_slope_variance(7.0, wind, swell, 1e4)
    assert far == pytest.approx(
        waves.cox_munk_slope(7.0) ** 2 + waves.slope_variance(swell)
    )
    near = waves.unresolved_slope_variance(7.0, wind, swell, 1e-3)
    assert near == pytest.approx(waves.unresolved_slope_variance(7.0, wind, (), 1e-3))


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
    cutoff = 2 * math.pi / waves.CAPILLARY_WAVELENGTH_M
    k = np.geomspace(
        waves.peak_omega_rad_s(wind_speed_mps, None) ** 2 / waves.GRAVITY_MS2,
        cutoff,
        100_000,
    )
    sea = field(wind_speed_mps)
    # A light wind's sea is held to Cox & Munk's slope, every wave alike.
    _, variance = waves._bins(wind_speed_mps, None)
    held = sum(w.amplitude_m**2 / 2 for w in sea) / variance.sum()
    weight = held * waves.curvature_spectrum(k, wind_speed_mps, None) * k
    m4 = np.trapezoid(weight, k)
    omega_sq = np.trapezoid(weight * waves.GRAVITY_MS2 * k, k) / m4
    assert waves.specular_cell_m2(sea) == pytest.approx(2 * math.sqrt(3) / m4, rel=0.1)
    assert waves.twinkle_hz(sea) == pytest.approx(
        math.sqrt(omega_sq) / (2 * math.pi), rel=0.1
    )


def test_calm_water_does_not_glitter() -> None:
    assert waves.specular_cell_m2(()) == math.inf
    assert waves.twinkle_hz(()) == 0.0


def test_the_spread_narrows_to_the_peak_and_widens_above_it() -> None:
    peak = waves.peak_omega_rad_s(7.0, None)
    delta = waves.spreading_delta(np.array([peak, 4 * peak]), 7.0, None)
    assert delta[0] == pytest.approx(1.0, abs=1e-3)
    assert delta[1] < 0.5


def test_wind_waves_run_within_a_right_angle_of_downwind() -> None:
    built = field(7.0)
    off = [math.remainder(w.toward_rad - math.pi, 2 * math.pi) for w in built]
    assert max(abs(o) for o in off) <= math.pi / 2


@pytest.mark.parametrize("wind_speed_mps", [5.0, 10.0])
def test_the_drawn_slope_is_upwind_heavy(wind_speed_mps: float) -> None:
    ratios = [
        np.divide(*waves.drawn_slope_variances(field(wind_speed_mps, seed=seed), 0, 0))
        for seed in range(8)
    ]

    assert 1.2 < np.mean(ratios) < 2.0


@pytest.mark.parametrize("footprint_m", [0.05, 1.0])
def test_drawn_and_unresolved_hold_cox_and_munk_s_slope_on_each_axis(
    footprint_m: float,
) -> None:
    built = field(7.0)
    drawn = waves.drawn_slope_variances(built, footprint_m, 0.0)
    left = waves.unresolved_slope_variances(7.0, 0.0, built, (), footprint_m)

    assert np.add(drawn, left) == pytest.approx(waves.cox_munk_variances(7.0))


@pytest.mark.parametrize(
    ("wind_speed_mps", "upwind_heavier"),
    # Cox & Munk's fits cross at 2.4 m/s.
    [(0.0, False), (2.0, False), (3.0, True), (7.0, True), (15.0, True)],
)
def test_the_axes_split_cox_and_munk_s_total(
    wind_speed_mps: float, upwind_heavier: bool
) -> None:
    up, cross = waves.cox_munk_variances(wind_speed_mps)

    assert up + cross == pytest.approx(waves.cox_munk_slope(wind_speed_mps) ** 2)
    assert (up > cross) == upwind_heavier


@pytest.mark.parametrize("footprint_m", [0.01, 1.0, 100.0])
def test_the_axes_left_unresolved_sum_to_the_total(footprint_m: float) -> None:
    built, swell = field(7.0), (waves.Wave(0.5, 0.6, 1.0, 0.0),)

    up, cross = waves.unresolved_slope_variances(7.0, 0.0, built, swell, footprint_m)

    assert up + cross == pytest.approx(
        waves.unresolved_slope_variance(7.0, built, swell, footprint_m), rel=1e-9
    )


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
    _, variance = waves._bins(7.0, None)
    built = field(7.0)
    assert sum(w.amplitude_m**2 / 2 for w in built) == pytest.approx(variance.sum())


def test_a_tile_ranked_onto_a_gaussian_keeps_its_pattern_and_loses_its_skew() -> None:
    tile = np.random.default_rng(0).gamma(2.0, size=(64, 64))
    ranked = waves.gaussian_ranks(tile)
    assert np.array_equal(np.argsort(ranked, axis=None), np.argsort(tile, axis=None))
    assert np.clip(ranked, -1.0, 1.0).mean() == pytest.approx(0.0, abs=1e-12)
