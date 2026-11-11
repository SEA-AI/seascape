"""Where a hull is at a time. No Blender."""

import math

import numpy as np
import pytest

from seascape import motion
from seascape.config import Drift, Outputs, Ownship

SPEED_MPS = 400.0
OUTPUTS = Outputs(duration_s=4.0)
STEERED = Ownship(speed_mps=SPEED_MPS, yaw={"amplitude_deg": 30.0, "period_s": 4.0})


def test_the_track_integrates_the_heading_to_the_closed_form() -> None:
    end_s = OUTPUTS.times_s[-1]

    end = motion.track(STEERED, OUTPUTS)(end_s)

    theta = np.linspace(0.0, end_s, 400_001)
    heading = np.radians(30.0) * np.sin(2.0 * np.pi * theta / 4.0)
    east = SPEED_MPS * np.trapezoid(np.sin(heading), theta)
    north = SPEED_MPS * np.trapezoid(np.cos(heading), theta)
    assert (end.east_m, end.north_m) == pytest.approx((east, north), rel=1e-3)


@pytest.mark.parametrize(
    "path",
    [
        motion.track(STEERED, OUTPUTS),
        motion.drifting(
            300.0,
            40.0,
            120.0,
            2.0,
            Drift(sway_m=3.0, surge_m=1.5, period_s=6.0),
            OUTPUTS,
        ),
        motion.orbiting(150.0, -30.0, 40.0),
    ],
    ids=["track", "drifting", "orbiting"],
)
def test_a_path_s_velocity_is_the_rate_its_position_changes(
    path: motion.Path,
) -> None:
    h = 1e-4
    for t_s in (0.3, 1.1, 2.7):
        before, now, after = path(t_s - h), path(t_s), path(t_s + h)
        rate = (after.east_m - before.east_m, after.north_m - before.north_m)
        assert (now.east_mps, now.north_mps) == pytest.approx(
            (rate[0] / (2 * h), rate[1] / (2 * h)), rel=2e-3, abs=1e-6
        )


def test_an_orbit_keeps_its_range_and_its_bow_on_the_tangent() -> None:
    path = motion.orbiting(150.0, -30.0, 40.0)

    for t_s in (0.0, 10.0, 25.0):
        pose = path(t_s)
        assert math.hypot(pose.east_m, pose.north_m) == pytest.approx(150.0)
        out = math.atan2(pose.east_m, pose.north_m)
        assert (pose.heading_rad - out) % (2 * math.pi) == pytest.approx(math.pi / 2)
