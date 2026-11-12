"""Compass angles."""

import numpy as np
import pytest

from seascape import angles


@pytest.mark.parametrize(
    ("bearing_deg", "elevation_deg", "toward"),
    [
        (0.0, 0.0, (0.0, 1.0, 0.0)),
        (90.0, 0.0, (1.0, 0.0, 0.0)),
        (33.0, 90.0, (0, 0, 1)),
    ],
)
def test_a_direction_points_along_its_compass_bearing(
    bearing_deg: float, elevation_deg: float, toward: tuple[float, float, float]
) -> None:
    got = angles.direction(bearing_deg, elevation_deg)
    np.testing.assert_allclose(got, toward, atol=1e-12)


@pytest.mark.parametrize("bearing_deg", [0.0, 45.0, 135.0, -90.0, 180.0])
def test_a_bearing_reads_back_its_direction(bearing_deg: float) -> None:
    east, north, _ = angles.direction(bearing_deg, 0.0)
    assert angles.bearing_deg(east, north) == pytest.approx(bearing_deg)


@pytest.mark.parametrize(("bearing_deg", "elevation_deg"), [(0.0, 0.0), (30.0, -12.0)])
def test_a_turn_takes_ahead_to_its_direction_and_keeps_starboard_level(
    bearing_deg: float, elevation_deg: float
) -> None:
    turn = angles.turned(bearing_deg, elevation_deg)
    np.testing.assert_allclose(
        turn[:, 1], angles.direction(bearing_deg, elevation_deg), atol=1e-12
    )
    assert turn[2, 0] == pytest.approx(0.0, abs=1e-12)
