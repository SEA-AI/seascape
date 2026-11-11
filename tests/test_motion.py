"""Where a hull is at a time, and a scripted turn. No Blender."""

import math

import numpy as np
import pytest
from pydantic import ValidationError

from seascape import motion
from seascape.config import Drift, Move, Outputs, Ownship, Turn, load
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


@pytest.mark.parametrize("accel_deg_s2", [None, 30.0, 2.0])
def test_a_turn_comes_to_rest_at_each_target_and_never_past_it(
    accel_deg_s2: float | None,
) -> None:
    turn = accelerating(accel_deg_s2)
    turned = motion.turned_deg(turn)

    for leg, move in zip(turn.legs(), turn.moves, strict=True):
        assert turned(leg.start_s + leg.duration_s) == pytest.approx(move.to_deg)
    swept = [turned(t) for t in np.arange(0.0, turn.duration_s, 1e-2)]
    assert (min(swept), max(swept)) == pytest.approx((-60.0, 90.0))


def test_the_rate_keeps_under_its_cap_and_ramps_at_the_acceleration() -> None:
    dt_s = 1e-3
    times = np.arange(0.0, COURSE.duration_s, dt_s)
    rate = np.diff([motion.turned_deg(COURSE)(t) for t in times]) / dt_s
    # The sweep at 6 deg/s ends as the turn comes back to the bow.
    back = COURSE.legs()[3]

    assert np.abs(rate).max() == pytest.approx(30.0, rel=1e-3)
    sweep = rate[times[1:] <= back.start_s + back.duration_s]
    assert np.abs(sweep).max() == pytest.approx(6.0, rel=1e-3)
    assert np.abs(np.diff(rate)).max() / dt_s == pytest.approx(30.0, rel=1e-2)


def test_each_move_stops_after_the_last_and_its_hold() -> None:
    turned = motion.turned_deg(accelerating(None))

    # 3 s still, then 60, 120 and 60 deg at 6 deg/s, 3 s still, 90 deg twice at 30.
    stops = [(13.0, 60.0), (33.0, -60.0), (43.0, 0.0), (49.0, 90.0), (53.0, 0.0)]
    for t_s, to_deg in stops:
        assert turned(t_s) == pytest.approx(to_deg)
        assert turned(t_s - 0.1) != pytest.approx(to_deg)
    assert turned(60.0) == 0.0


def test_a_sailing_ownship_turns_on_a_circle_of_speed_over_rate() -> None:
    speed_mps, rate_deg_s = 10.0, 6.0
    quarter = Turn(
        by="ownship",
        top_rate_deg_s=rate_deg_s,
        moves=[Move(to_deg=90.0)],
    )
    outputs = Outputs(duration_s=quarter.duration_s + 1.0, fps=10)

    end = motion.track(Ownship(speed_mps=speed_mps), outputs, quarter)(
        quarter.duration_s
    )

    # A quarter turn to starboard from north ends a radius east and a radius north.
    radius_m = speed_mps / math.radians(rate_deg_s)
    assert (end.east_m, end.north_m) == pytest.approx((radius_m, radius_m), rel=1e-3)


@pytest.mark.parametrize(
    ("changes", "error"),
    [
        (
            {"turn": COURSE.model_dump(), "outputs": {"duration_s": 10}},
            "the turn ends at",
        ),
        (
            {"turn": COURSE.model_dump(), "outputs": {"duration_s": 60, "loop": True}},
            "cannot close a scripted turn",
        ),
        (
            {
                "turn": {
                    "by": "ownship",
                    "top_rate_deg_s": 5.0,
                    "moves": [{"to_deg": 10, "rate_deg_s": 6}],
                }
            },
            "exceeds top_rate_deg_s",
        ),
    ],
)
def test_a_turn_that_cannot_run_is_refused(
    changes: dict[str, object], error: str
) -> None:
    with pytest.raises(ValidationError, match=error):
        load(OPEN_SEA, [changes])


@pytest.mark.parametrize(
    ("moves", "fps"),
    [
        (COURSE.model_dump()["moves"], 30),
        # 0.1096 s: rounding the suggestion to 0.01 s left it a frame short.
        ([{"to_deg": 1.096}], 10),
    ],
)
def test_a_turn_too_long_suggests_a_duration_that_fits(
    moves: list[dict[str, object]], fps: int
) -> None:
    turn = {"by": "ownship", "top_rate_deg_s": 10.0, "moves": moves}
    with pytest.raises(ValidationError, match="at least") as refused:
        load(OPEN_SEA, [{"turn": turn, "outputs": {"duration_s": 0.1, "fps": fps}}])
    suggested = float(str(refused.value).split("at least ")[1].split()[0])

    load(OPEN_SEA, [{"turn": turn, "outputs": {"duration_s": suggested, "fps": fps}}])
