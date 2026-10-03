"""Where a photographed sky's sun is. No Blender."""

import math

import numpy as np
import pytest

from seascape import skies

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
