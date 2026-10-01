import math

import numpy as np
import pytest

from seascape import clouds, waves


def test_the_bands_add_up_to_one_spectrum_of_unit_variance() -> None:
    bands = clouds.tiles(np.random.default_rng(0), clouds.STRATOCUMULUS_BASE_M)
    assert sum(b.var() for b in bands) == pytest.approx(1.0)
    # Each takes only the waves its coarser neighbour cannot hold.
    for band, spacing_m, coarser_m in zip(
        bands[1:], clouds.BAND_SPACINGS_M[1:], clouds.BAND_SPACINGS_M, strict=False
    ):
        k = 2 * math.pi * np.fft.fftfreq(clouds.CELLS, spacing_m)
        held = np.hypot(k[:, None], k[None, :]) < math.pi / coarser_m
        power = np.abs(np.fft.fft2(band)) ** 2
        assert power[held].sum() < 1e-20 * power.sum()


def test_the_coarsest_tile_spans_the_layer_at_its_default_base() -> None:
    reach_m = clouds.reach_m(clouds.STRATOCUMULUS_BASE_M, waves.earth_radius_m(0.13))
    assert clouds.CELLS * clouds.BAND_SPACINGS_M[0] > 2 * reach_m


def test_the_spectrum_peaks_where_wood_and_hartmann_s_does() -> None:
    base_m = 800.0
    length_m = clouds.integral_scale_m(base_m)
    k0 = math.sqrt(math.pi) * math.gamma(5 / 6) / (math.gamma(1 / 3) * length_m)
    k = np.linspace(1e-6, 10 * k0, 200_001)
    # Summed round each ring.
    peak = k[np.argmax(k * (k0**2 + k**2) ** (-4 / 3))]
    assert 2 * math.pi / peak == pytest.approx(clouds.PEAK_PER_DEPTH * base_m, rel=1e-4)


@pytest.mark.parametrize("cover", [0.1, 0.5, 0.9])
def test_cover_is_the_seen_fraction_and_the_seen_tau_the_marine_mean(
    cover: float,
) -> None:
    threshold, scale = clouds.optical_depth(cover)
    z = np.random.default_rng(0).standard_normal(2_000_000)
    tau = scale * np.maximum(z - threshold, 0) ** clouds.THICKNESS_POWER
    seen = tau > clouds.SEEN_OPTICAL_DEPTH
    assert seen.mean() == pytest.approx(cover, abs=2e-3)
    assert tau[seen].mean() == pytest.approx(clouds.MEAN_OPTICAL_DEPTH, rel=0.01)


def test_marine_sc_has_the_optical_depth_its_water_path_gives() -> None:
    """Stephens 1978, at Han et al.'s mean marine LWP: inside the 8-23 that Wood,
    "Stratocumulus clouds", MWR 140(8) 2373, 2012 (doi:10.1175/MWR-D-11-00121.1), gives
    for marine Sc."""
    assert 8.0 < clouds.MEAN_OPTICAL_DEPTH < 23.0


def test_a_ray_along_the_horizon_meets_the_layer_where_it_ends() -> None:
    earth_m, base_m = waves.earth_radius_m(0.13), clouds.STRATOCUMULUS_BASE_M
    reach_m = clouds.reach_m(base_m, earth_m)
    # From the earth's centre, the point the horizontal ray reaches.
    assert math.hypot(reach_m, earth_m) == pytest.approx(earth_m + base_m)
