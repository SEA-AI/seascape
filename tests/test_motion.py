"""Where a hull is at a time, and a scripted turn. No Blender."""

import math

import numpy as np
import pytest
from pydantic import ValidationError

from seascape import motion
from seascape.config import Drift, Move, Outputs, Ownship, Turn, load
from seascape.physics import turn as kinematics
from tests.scenarios import OPEN_SEA

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
    # Off the track's grid, and past either end of the clip.
    for t_s in (0.3, 1.1, 1.1125, 2.7, -0.5, OUTPUTS.span_s + 0.5):
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


COURSE = Turn(
    by="ownship",
    top_rate_deg_s=30.0,
    accel_deg_s2=30.0,
    moves=[
        Move(to_deg=0.0, hold_s=3.0),
        Move(to_deg=60.0, rate_deg_s=6.0),
        Move(to_deg=-60.0, rate_deg_s=6.0),
        Move(to_deg=0.0, rate_deg_s=6.0, hold_s=3.0),
        Move(to_deg=90.0, hold_s=1.0),
        Move(to_deg=0.0, hold_s=3.0),
    ],
)


def accelerating(accel_deg_s2: float | None) -> Turn:
    return Turn.model_validate({**COURSE.model_dump(), "accel_deg_s2": accel_deg_s2})


def stops(turn: Turn) -> list[tuple[float, float]]:
    """Each move's (time, angle) as it comes to rest."""
    return [
        (leg.start_s + leg.duration_s, leg.start_deg + leg.distance_deg)
        for leg in turn.legs()
        if leg.distance_deg
    ]


@pytest.mark.parametrize("accel_deg_s2", [None, 30.0, 2.0])
def test_a_turn_comes_to_rest_at_each_target(accel_deg_s2: float | None) -> None:
    turn = accelerating(accel_deg_s2)
    turned = motion.turned_deg(turn)

    for t_s, to_deg in stops(turn):
        assert turned(t_s) == pytest.approx(to_deg, abs=1e-9)
    assert turned(turn.duration_s + 5.0) == 0.0


def test_the_rate_keeps_under_its_cap_and_ramps_at_the_acceleration() -> None:
    dt_s = 1e-3
    times = np.arange(0.0, COURSE.duration_s, dt_s)
    rate = np.diff([motion.turned_deg(COURSE)(t) for t in times]) / dt_s
    back_at_bow_s = stops(COURSE)[2][0]

    assert np.abs(rate).max() == pytest.approx(30.0, rel=1e-3)
    sweep = rate[times[1:] <= back_at_bow_s]
    assert np.abs(sweep).max() == pytest.approx(6.0, rel=1e-3)
    assert np.abs(np.diff(rate)).max() / 1e-3 == pytest.approx(30.0, rel=1e-2)


def test_a_short_turn_never_reaches_its_rate() -> None:
    # Up and down at 2 deg/s^2 over 2 deg peaks at 2 deg/s, in 2 s.
    assert kinematics.peak_deg_s(2.0, 6.0, 2.0) == pytest.approx(2.0)
    assert kinematics.duration_s(2.0, 6.0, 2.0) == pytest.approx(2.0)


def test_each_move_stops_after_the_last_and_its_hold() -> None:
    points = stops(accelerating(None))

    # 3 s still, then 60, 120 and 60 deg at 6 deg/s, 3 s still, 90 deg twice at 30.
    assert points == pytest.approx(
        [(13.0, 60.0), (33.0, -60.0), (43.0, 0.0), (49.0, 90.0), (53.0, 0.0)]
    )


def test_an_ownship_that_turns_heads_where_the_turn_points() -> None:
    outputs = Outputs(duration_s=COURSE.duration_s, fps=10)
    path = motion.track(Ownship(), outputs, COURSE)
    turned = motion.turned_deg(COURSE)

    for t_s in outputs.times_s:
        assert math.degrees(path(t_s).heading_rad) == pytest.approx(turned(t_s))


def test_a_turn_must_fit_the_clip_and_never_loops() -> None:
    turn = COURSE.model_dump()
    with pytest.raises(ValidationError, match="the turn takes"):
        load(OPEN_SEA, [{"turn": turn, "outputs": {"duration_s": 10}}])
    with pytest.raises(ValidationError, match="not a loop"):
        load(OPEN_SEA, [{"turn": turn, "outputs": {"duration_s": 60, "loop": True}}])


def test_a_move_faster_than_the_top_rate_is_refused() -> None:
    with pytest.raises(ValidationError, match="faster than top_rate_deg_s"):
        Turn(by="ownship", top_rate_deg_s=5.0, moves=[Move(to_deg=10, rate_deg_s=6)])
