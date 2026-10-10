"""Physics assertions for the LWIR band. No Blender."""

import math
from typing import get_args

import numpy as np
import pytest

from seascape import lwir

# sigma = 2 pi^5 k^4 / (15 h^3 c^2), exactly.
STEFAN_BOLTZMANN = (
    2 * np.pi**5 * lwir.BOLTZMANN_K**4 / (15 * lwir.PLANCK_H**3 * lwir.LIGHT_C**2)
)


@pytest.fixture(scope="module")
def curve() -> tuple[np.ndarray, np.ndarray]:
    return lwir.emissivity_curve()


def eps_at(curve: tuple[np.ndarray, np.ndarray], deg: float) -> float:
    angles, eps = curve
    return float(np.interp(np.radians(deg), angles, eps))


def test_normal_incidence_emissivity_is_about_0_99(curve) -> None:
    """0.98-0.99 is the standard handbook emissivity for water in this band."""
    assert 0.98 <= eps_at(curve, 0.0) <= 0.995


def test_emissivity_falls_with_angle(curve) -> None:
    """Fresnel reflectance rises monotonically with incidence for an absorbing
    medium, so emissivity has to fall. A wobble means the complex sqrt took a wrong
    branch or the band integration picked up a sign error.
    """
    assert eps_at(curve, 60.0) > eps_at(curve, 80.0) > eps_at(curve, 89.0)


def test_grazing_emissivity_collapses(curve) -> None:
    assert eps_at(curve, 89.0) < 0.6


def test_table_reproduces_downing_williams_at_their_own_temperature() -> None:
    """Downing & Williams measured at 300 K, so the table must return what they
    printed."""
    lam, n, k = lwir.optical_constants(300.0)
    j = int(np.argmin(np.abs(lam - 10e-6)))
    assert n[j] == pytest.approx(1.214, abs=1e-3)
    assert k[j] == pytest.approx(0.0534, abs=5e-4)


def test_stefan_boltzmann_matches_its_published_value() -> None:
    """Guards the transcription of h, c and k_B against CODATA's rounded sigma."""
    sigma = (
        STEFAN_BOLTZMANN  # local: ruff reads a bare constant here as a Yoda condition
    )
    assert sigma == pytest.approx(5.670374419e-8, rel=1e-9)


def test_the_shipped_grid_is_unbroken() -> None:
    """A regenerated file with a different span or step changes the physics in
    silence."""
    wavenumbers, temperatures, _ = lwir._table()
    assert np.array_equal(wavenumbers, np.arange(700.0, 1261.0, 20.0))
    assert np.array_equal(temperatures, np.arange(271.0, 312.0, 4.0))


def test_optical_constants_vary_smoothly() -> None:
    """A slipped decimal in the file or a transposed column in the reader moves a
    value off the midpoint of its neighbours; real data stays within a few percent."""
    _, n, k = lwir.optical_constants()
    for values in (n, k):
        midpoint_of_neighbours = (values[:-2] + values[2:]) / 2
        off_by = np.abs(values[1:-1] - midpoint_of_neighbours) / values[1:-1]
        assert off_by.max() < 0.10


def test_emissivity_rises_with_sea_temperature_but_barely() -> None:
    """The epsilon absorbs 1e-17 wobble at 90 degrees, where emissivity is zero."""
    cold = lwir.emissivity_curve(t_sea_k=271.0)[1]
    warm = lwir.emissivity_curve(t_sea_k=311.0)[1]
    assert np.all(warm >= cold - 1e-12)
    assert np.abs(warm - cold).max() < 0.02


def test_temperature_is_clamped_to_the_measured_range() -> None:
    """Extrapolating optical constants past the measurements would invent data."""
    below = lwir.optical_constants(200.0)
    at_edge = lwir.optical_constants(271.0)
    assert all(np.array_equal(a, b) for a, b in zip(below, at_edge, strict=True))


@pytest.mark.parametrize("bad", [-1.0, 0.0, float("nan")])
def test_planck_rejects_impossible_temperatures(bad: float) -> None:
    with pytest.raises(ValueError, match="positive"):
        lwir.planck(1e-5, bad)
    with pytest.raises(ValueError, match="positive"):
        lwir.band_radiance(bad)
    with pytest.raises(ValueError, match="positive"):
        lwir.optical_constants(bad)
    with pytest.raises(ValueError, match="positive"):
        lwir.emissivity_curve(t_sea_k=bad)


def test_band_holds_a_plausible_share_of_total_emission() -> None:
    """Catches a unit error in Planck; wide enough to survive a change of integration
    scheme."""
    total = STEFAN_BOLTZMANN * 288.0**4 / np.pi
    assert 0.35 <= lwir.band_radiance(288.0) / total <= 0.50


def test_sky_meets_ambient_at_the_horizon() -> None:
    assert float(lwir.sky_radiance(0.0, 290.0)) == pytest.approx(
        lwir.band_radiance(290.0)
    )


def test_sky_cools_toward_the_zenith() -> None:
    """Catches a curve that dips in the middle: a bad interpolation or an
    out-of-order table."""
    radiance = lwir.sky_radiance(np.radians([0.0, 5.0, 20.0, 45.0, 90.0]))
    assert np.all(np.diff(radiance) < 0.0)


def test_zenith_sky_is_as_cold_as_a_real_clear_sky() -> None:
    """Published clear-sky zenith brightness temperature spans ~230-265 K in band."""
    zenith = float(lwir.sky_radiance(np.pi / 2, 288.0, "midlatitude_summer"))
    assert lwir.band_radiance(230.0) <= zenith <= lwir.band_radiance(265.0)


def test_wetter_air_warms_the_zenith() -> None:
    """Water vapour is what the band sees overhead, so the tropics glow most."""
    zenith = [
        float(lwir.sky_radiance(np.pi / 2, 288.0, a))
        for a in ("tropical", "midlatitude_summer", "us_standard", "midlatitude_winter")
    ]
    assert zenith == sorted(zenith, reverse=True)


def test_sky_below_the_horizon_holds_at_ambient() -> None:
    """The sea grid is finite, so a ray can pass under the horizon and miss it."""
    assert float(lwir.sky_radiance(np.radians(-20.0))) == pytest.approx(
        float(lwir.sky_radiance(0.0))
    )


def test_sky_rejects_impossible_air_temperatures() -> None:
    with pytest.raises(ValueError, match="positive"):
        lwir.sky_radiance(0.0, -1.0)


def test_brightness_temperature_inverts_band_radiance() -> None:
    """Off the lookup grid, where interpolation error would show."""
    t_k = np.array([250.05, 288.13, 311.37])
    radiance = [lwir.band_radiance(t) for t in t_k]
    assert lwir.brightness_temperature(radiance) == pytest.approx(t_k, abs=0.01)


def test_a_path_of_no_length_keeps_all_its_light() -> None:
    assert lwir.path_optical_depth(0.0, 42.0) == 0.0


def test_below_the_table_the_power_law_meets_it() -> None:
    first = lwir._path_table()[lwir.ATMOSPHERE][0][0]
    below, above = lwir.path_optical_depth([first * 0.999, first * 1.001], 42.0)
    assert below == pytest.approx(above, rel=1e-2)


def test_aerosol_adds_to_water_vapour_which_takes_most_of_the_path() -> None:
    clear, typical, hazy = (lwir.path_optical_depth(10_000.0, v) for v in (None, 42, 5))
    assert clear < typical < hazy
    assert clear / typical > 0.8


def test_every_atmosphere_has_a_sky_a_path_and_a_surface_air() -> None:
    names = set(get_args(lwir.Atmosphere.__value__))
    assert set(lwir._sky_table()[1]) == names
    assert set(lwir._path_table()) == names
    assert set(lwir.SURFACE_AIR_K) == names


def test_a_cloud_overhead_reads_between_the_clear_sky_and_its_base() -> None:
    clear = float(lwir.sky_radiance(np.pi / 2, 290.0))
    cloudy = float(lwir.cloudy_sky_radiance(np.pi / 2, 1.0, 1000.0, 290.0))
    assert clear < cloudy < lwir.band_radiance(290.0)


def test_a_cloud_never_cools_the_sky_it_hides() -> None:
    elevation = np.radians([0.0, 5.0, 45.0, 90.0])
    clear = lwir.sky_radiance(elevation, 290.0)
    cloudy = lwir.cloudy_sky_radiance(elevation, 1.0, 1000.0, 290.0)
    assert np.all(cloudy >= clear)
    assert float(cloudy[0]) == pytest.approx(float(clear[0]))


def test_no_cloud_is_the_clear_sky() -> None:
    elevation = np.radians([0.0, 30.0, 90.0])
    assert lwir.cloudy_sky_radiance(elevation, 0.0, 1000.0) == pytest.approx(
        lwir.sky_radiance(elevation)
    )


@pytest.mark.parametrize("theta_deg", [0.0, 30.0, 60.0])
def test_the_facets_show_the_view_the_mean_surface_s_area(theta_deg: float) -> None:
    """Facets steep enough to face away are past 5 sigma here, so the seen areas sum
    to the mean surface's foreshortened by the view."""
    normal = lwir._facets(0.15)
    theta = math.radians(theta_deg)
    view = np.array([math.sin(theta), 0.0, math.cos(theta)])

    area = lwir._seen_area(normal @ view, normal)

    assert area.mean() == pytest.approx(math.cos(theta), rel=1e-4)


def test_the_facets_carry_their_slope_sigma() -> None:
    """Per axis, as `emissivity_curve` and `reflected_sky` read it."""
    normal = lwir._facets(0.1)
    slope = -normal[:, :2] / normal[:, 2:]
    assert slope.mean(axis=0) == pytest.approx([0.0, 0.0], abs=1e-12)
    assert slope.std(axis=0) == pytest.approx([0.1, 0.1], rel=1e-9)


def test_a_flat_sea_reflects_the_sky_in_its_mirror_direction() -> None:
    elev = np.radians(np.linspace(0.0, 90.0, 91))
    sky = lwir.sky_radiance(elev)
    mirror = np.radians([1.0, 10.0, 45.0])
    assert lwir.reflected_sky(mirror, elev, sky) == pytest.approx(
        lwir.sky_radiance(mirror)
    )


def test_a_rough_sea_seen_grazing_reflects_sky_from_above_its_mirror() -> None:
    elev = np.radians(np.linspace(0.0, 90.0, 91))
    sky = lwir.sky_radiance(elev)
    mirror = np.radians(1.0)
    flat = lwir.reflected_sky(mirror, elev, sky)
    smooth, rough = (
        lwir.reflected_sky(mirror, elev, sky, slope_sigma=s) for s in (0.05, 0.15)
    )
    assert rough < smooth < flat


@pytest.mark.parametrize("mirror_deg", [0.2, 1.0, 3.0, 10.0])
def test_mean_emissivity_and_reflected_sky_add_up_to_the_facets_own_sum(
    mirror_deg: float,
) -> None:
    """The shader takes eps B + (1 - eps) S of facet means; each facet emits and
    reflects in its own proportion."""
    t_sea_k, sigma = lwir.T_SEA_K, 0.1
    elev = np.radians(np.linspace(0.0, 90.0, 181))
    sky = lwir.sky_radiance(elev)
    mirror = math.radians(mirror_deg)
    reflected = lwir.reflected_sky(
        mirror, elev, sky, t_sea_k=t_sea_k, slope_sigma=sigma
    )
    theta, eps = lwir.emissivity_curve(t_sea_k=t_sea_k, slope_sigma=sigma)
    mean_eps = np.interp(math.pi / 2 - mirror, theta, eps)
    hot = lwir.band_radiance(t_sea_k)

    normal = lwir._facets(sigma)
    view = np.array([math.cos(mirror), 0.0, math.sin(mirror)])
    cos_i = normal @ view
    area = lwir._seen_area(cos_i, normal)
    flat_theta, flat = lwir.emissivity_curve(t_sea_k=t_sea_k)
    own = np.interp(np.arccos(np.clip(cos_i, -1.0, 1.0)), flat_theta, flat)
    up = np.arcsin(np.clip(2 * cos_i * normal[:, 2] - view[2], -1.0, 1.0))
    each = own * hot + (1 - own) * np.interp(up, elev, sky)
    exact = (area * each).sum() / area.sum()

    shaded = mean_eps * hot + (1 - mean_eps) * reflected
    assert lwir.brightness_temperature(shaded) == pytest.approx(
        lwir.brightness_temperature(exact), abs=0.006
    )
