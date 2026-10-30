"""Where a photographed sky's sun is. No Blender."""

import math

import numpy as np
import pytest

from seascape.assets import skies
from seascape.physics import lwir

W, H = 2048, 1024


def _sky(bearing_deg: float, elevation_deg: float, radius_deg: float, peak: float):
    """A sky of 1 with a bright spot at the bearing and elevation, as `skies` maps
    pixels to directions."""
    u = (np.arange(W) + 0.5) / W
    rows = np.arange(H)
    b = np.radians((360.0 * u - 90.0) % 360.0)[None, :]
    e = np.radians(90.0 - (rows + 0.5) / H * 180.0)[:, None]
    s_b, s_e = math.radians(bearing_deg), math.radians(elevation_deg)
    cosine = np.sin(e) * math.sin(s_e) + np.cos(e) * math.cos(s_e) * np.cos(b - s_b)
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    lum = 1.0 + peak * np.exp(-0.5 * (angle / radius_deg) ** 2)
    return np.repeat(lum[..., None], 3, axis=-1).astype(np.float32)


@pytest.mark.parametrize(("bearing_deg", "elevation_deg"), [(10.0, 30.0), (250.0, 3.0)])
def test_a_disc_is_found_where_it_is(bearing_deg: float, elevation_deg: float) -> None:
    sun = skies.sun(_sky(bearing_deg, elevation_deg, 0.25, 1e5))
    assert sun.bearing_deg == pytest.approx(bearing_deg, abs=0.1)
    assert sun.elevation_deg == pytest.approx(elevation_deg, abs=0.1)


def test_a_glow_turns_the_sky_but_is_no_disc() -> None:
    sun = skies.sun(_sky(200.0, 10.0, 8.0, 20.0))
    assert sun.elevation_deg is None
    assert sun.bearing_deg == pytest.approx(200.0, abs=2.0)


def test_a_star_is_no_sun() -> None:
    sky = _sky(90.0, 20.0, 8.0, 2.0)
    sky[100, 700] = 1e5
    assert skies.sun(sky).elevation_deg is None


SUN_DEG = (90.0, 20.0)
PHOTO = skies.Photo(
    url="https://example.com/sky.hdr",
    sha256="0" * 64,
    sun_bearing_deg=SUN_DEG[0],
    sun_elevation_deg=SUN_DEG[1],
    sun_floor=1.0,
    licence="CC0-1.0",
    attribution="a test sky",
)
CW, CH = 512, 256
# Below 2 degrees the fit cannot follow a white horizon.
ABOVE_2_DEG = slice(0, round(CH * 88 / 180))


def _clear_sky() -> np.ndarray:
    """Blue overhead, whitening to the horizon and round the sun as a clear sky does."""
    rows = np.arange(H)
    elevation = np.radians(np.clip(90.0 - (rows + 0.5) / H * 180.0, 0.0, 90.0))
    glow = _sky(*SUN_DEG, 15.0, 1.0)[..., 0] - 1.0
    white = 1.0 - np.sin(elevation)[:, None] * (1.0 - glow)
    blue = np.array([0.2, 0.4, 1.0], np.float32)
    sky = blue + white[..., None] * (1.0 - blue)
    return sky[:: H // CH, :: W // CW]


def test_a_clear_sky_hazed_at_the_horizon_and_the_sun_is_no_cloud() -> None:
    assert skies.cloud(_clear_sky(), PHOTO)[ABOVE_2_DEG].max() < 0.05


def test_grey_on_a_clear_sky_is_cloud() -> None:
    sky = _clear_sky()
    sky[40:60, 300:340] = 0.8
    cloud = skies.cloud(sky, PHOTO)
    assert cloud[40:60, 300:340].min() == pytest.approx(1.0)
    assert cloud[ABOVE_2_DEG, :250].max() < 0.05


def test_a_shaded_base_is_cloud_only_past_dark() -> None:
    sky = _clear_sky()
    base = np.s_[50:54, 318:322]  # about BESIDE_DEG across
    faint = np.s_[50:54, 150:154]
    sky[base] *= 0.9 * skies.DARK[1]
    sky[faint] *= 0.95  # over DARK[0]
    # One across the photo's seam.
    sky[50:54, :2] *= 0.9 * skies.DARK[1]
    sky[50:54, -2:] *= 0.9 * skies.DARK[1]
    cloud = skies.cloud(sky, PHOTO)
    assert cloud[base].min() == pytest.approx(1.0)
    assert cloud[50:54, :2].min() == cloud[50:54, -2:].min() == pytest.approx(1.0)
    assert cloud[faint].max() < 0.05
    assert cloud[ABOVE_2_DEG, 200:250].max() < 0.05


def test_a_clear_sky_darker_across_the_photo_is_no_cloud() -> None:
    sky = _clear_sky()
    # Half as bright opposite the sun, smoothly, as a clear sky's dark band is.
    bearing = np.radians(np.arange(CW) * 360.0 / CW - SUN_DEG[0])
    sky *= (0.75 + 0.25 * np.cos(bearing))[None, :, None]
    assert skies.cloud(sky, PHOTO)[ABOVE_2_DEG].max() < 0.05


def test_an_overcast_photo_is_cloud_down_to_the_horizon() -> None:
    cloud = skies.cloud(_clear_sky(), PHOTO.model_copy(update={"overcast": True}))
    assert cloud[: CH // 2].min() == 1.0
    assert cloud[CH // 2 :].max() == 0.0


@pytest.mark.parametrize("offset", [0, 1])
def test_gaps_in_a_mostly_grey_sky_leave_the_grey_cloud(offset: int) -> None:
    sky = _clear_sky()
    gaps = np.zeros(sky.shape[:2], bool)
    # Gaps a pixel wide, on and off the fit's stride, few enough that a fit through
    # every pixel lies near grey.
    s = skies.FIT_STRIDE
    gaps[offset::s, offset :: 20 * s] = True
    sky[~gaps] = 0.8
    # Above 45 degrees the clear sky is blue enough for grey to read whole cloud.
    above_45 = slice(0, CH // 4)
    cloud = skies.cloud(sky, PHOTO)[above_45]
    assert cloud[~gaps[above_45]].min() > 0.95


def test_a_cloud_s_bottom_is_the_foot_of_its_run_down_its_column() -> None:
    share = np.zeros((CH, CW), np.float32)
    share[30:60, 10] = 1.0  # a cloud with clear sky under it
    share[100 : CH // 2, 20] = 1.0  # one down to the horizon
    share[10:20, 30] = share[40:60, 30] = 1.0  # two runs in a column
    share[CH // 2 :] = 1.0
    bottom = skies.cloud_bottom_deg(share)
    foot = 90.0 - 60.0 / CH * 180.0
    assert bottom[30:60, 10] == pytest.approx(foot)
    assert bottom[10:20, 30] == pytest.approx(90.0 - 20.0 / CH * 180.0)
    assert np.isnan(bottom[100 : CH // 2, 20]).all()
    assert np.isnan(bottom[:, 0]).all()
    assert np.isnan(bottom[CH // 2 :]).all()


def test_a_row_s_cloud_height_is_its_share_weighted_side() -> None:
    share = np.zeros((CH, CW), np.float32)
    share[50:60, :10] = 1.0
    share[50:60, 10:20] = 0.5
    share[60 : CH // 2, 20] = 1.0  # bottom out of sight: no weight
    height = skies.cloud_height_m(share, 1000.0)
    elevation = np.radians(90.0 - 55.5 / CH * 180.0)
    bottom = np.radians(90.0 - 60.0 / CH * 180.0)
    assert height[55] == pytest.approx(
        float(lwir.cloud_height_m(elevation, bottom, 1000.0)), rel=1e-5
    )
    assert height[70] == 0.0
    assert height[CH // 2 :].max() == 0.0


def test_the_clear_sky_beside_a_pixel_ignores_the_grid_s_margin() -> None:
    """A strip no whole number of cells across is padded to one; the pad weighs
    nothing, so a sky of one value reads that value everywhere."""
    weight = np.random.default_rng(0).random((37, 101), np.float32)
    value = np.full_like(weight, 2.0)
    assert np.allclose(skies._beside(weight, value, 12.0), 2.0, atol=1e-5)
