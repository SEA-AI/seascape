"""Where a hull is at a time, and the waterplane the waves tilt it by there."""

import bisect
import math
from collections.abc import Callable
from typing import NamedTuple

import numpy as np

from seascape.config import Drift, Outputs, Ownship, Scenario, Turn
from seascape.config.objects import Swing
from seascape.physics import ptu, slew, waves

# Judgement: trapezoid steps per yaw period; the error falls with the square.
STEPS_PER_YAW = 64
# Judgement: the most a turn moves the heading in one integration substep.
STEP_TURN_DEG = 1.0


class Pose(NamedTuple):
    """East and north in the world, the heading clockwise from north, the velocity
    over the ground, and the pitch bow up and roll starboard down it is steered to,
    before the sea tilts it.
    """

    east_m: float
    north_m: float
    heading_rad: float
    east_mps: float = 0.0
    north_mps: float = 0.0
    pitch_rad: float = 0.0
    roll_rad: float = 0.0

    @property
    def speed_mps(self) -> float:
        """Over the ground."""
        return math.hypot(self.east_mps, self.north_mps)


type Path = Callable[[float], Pose]


def sine(mean: float, amplitude: float, period_s: float) -> Callable[[float], float]:
    """A sine of time in seconds."""
    return lambda t: mean + amplitude * math.sin(2.0 * math.pi * t / period_s)


def swung(
    mean_deg: float, swing: Swing | None, outputs: Outputs
) -> Callable[[float], float]:
    """`mean_deg` and the swing about it, in radians at a time."""
    if swing is None:
        return sine(math.radians(mean_deg), 0.0, 1.0)
    return sine(
        math.radians(mean_deg),
        math.radians(swing.amplitude_deg),
        outputs.period_s(swing.period_s),
    )


def _hermite(
    tau: np.ndarray, at: np.ndarray, rate: np.ndarray, t_s: float
) -> tuple[float, float]:
    """A value and its rate at `t_s`, from samples of both on `tau`: the cubic
    Hermite spline between samples, so the rate is the value's slope, and straight
    on at the end's rate past either end.
    """
    if t_s <= tau[0] or t_s >= tau[-1]:
        end = 0 if t_s <= tau[0] else -1
        return float(at[end] + rate[end] * (t_s - tau[end])), float(rate[end])
    i = int(np.searchsorted(tau, t_s, side="right")) - 1
    step = tau[i + 1] - tau[i]
    s = (t_s - tau[i]) / step
    p0, p1, m0, m1 = at[i], at[i + 1], rate[i] * step, rate[i + 1] * step
    value = (
        (2 * s**3 - 3 * s**2 + 1) * p0
        + (s**3 - 2 * s**2 + s) * m0
        + (-2 * s**3 + 3 * s**2) * p1
        + (s**3 - s**2) * m1
    )
    slope = (
        (6 * s**2 - 6 * s) * p0
        + (3 * s**2 - 4 * s + 1) * m0
        + (-6 * s**2 + 6 * s) * p1
        + (3 * s**2 - 2 * s) * m1
    )
    return float(value), float(slope / step)


def turned_deg(turn: Turn) -> Callable[[float], float]:
    """The turn's angle at a time, clockwise from where it starts."""
    legs = turn.legs()
    starts_s = [leg.start_s for leg in legs]

    def at(t_s: float) -> float:
        # `covered_deg` holds a leg's end after it, so the last holds forever.
        leg = legs[max(bisect.bisect_right(starts_s, t_s) - 1, 0)]
        return leg.start_deg + slew.covered_deg(
            leg.distance_deg, leg.rate_deg_s, turn.accel_deg_s2, t_s - leg.start_s
        )

    return at


def panned_deg(turn: Turn | None, rig: str) -> Callable[[float], float]:
    """The pan of the rig named `rig` at a time, clockwise from its yaw: the turn's
    when it turns that rig's ptu, else none.
    """
    if turn is None or not turn.pans(rig):
        return lambda _: 0.0
    return turned_deg(turn)


def track(ownship: Ownship, outputs: Outputs, turn: Turn | None = None) -> Path:
    """The pose at a time: `speed_mps` integrated along the heading that `yaw` swings
    and `turn` turns, with the pitch and roll its swings script.
    """
    times = np.asarray(outputs.times_s)
    swing = ownship.yaw
    # Whole substeps per frame, so every frame lands on the grid; a straight run is
    # exact in one.
    sub = 1
    if swing is not None:
        step_s = outputs.period_s(swing.period_s) / STEPS_PER_YAW
        sub = math.ceil(1.0 / (outputs.fps * step_s))
    yawed = swung(0.0, swing, outputs)
    turned = None
    if turn is not None and turn.ptu is None:
        turned = turned_deg(turn)
        sub = max(sub, math.ceil(turn.top_rate_deg_s / (outputs.fps * STEP_TURN_DEG)))
    tau = np.linspace(0.0, times[-1], (len(times) - 1) * sub + 1)

    def heading_rad(t_s: float) -> float:
        rad = yawed(t_s)
        return rad if turned is None else rad + math.radians(turned(t_s))

    heading = np.array([heading_rad(t) for t in tau])

    def integral(speed: np.ndarray) -> np.ndarray:
        area = np.diff(tau) * (speed[1:] + speed[:-1]) / 2.0
        return np.concatenate(([0.0], np.cumsum(area)))

    speed = ownship.speed_mps
    east = speed * integral(np.sin(heading))
    north = speed * integral(np.cos(heading))
    pitch = swung(ownship.pitch_deg, ownship.pitch, outputs)
    roll = swung(ownship.roll_deg, ownship.roll, outputs)

    def at(t_s: float) -> Pose:
        e, v_e = _hermite(tau, east, speed * np.sin(heading), t_s)
        n, v_n = _hermite(tau, north, speed * np.cos(heading), t_s)
        h = heading_rad(t_s)
        return Pose(e, n, h, v_e, v_n, pitch(t_s), roll(t_s))

    return at


def axis_deg(scenario: Scenario, name: str) -> Callable[[float], float]:
    """The axis of the rig `name` at a time, clockwise from the bow: its yaw and its
    pan.
    """
    yaw_deg, panned = scenario.rigs[name].yaw_deg, panned_deg(scenario.turn, name)
    return lambda t_s: yaw_deg + panned(t_s)


def held_tilt_deg(
    scenario: Scenario,
    name: str,
    attitude_deg: Callable[[float], tuple[float, float]],
) -> np.ndarray:
    """The tilt the pan-tilt unit of rig `name` adds to its mount's pitch at each
    frame, holding its inclination against the ownship's `attitude_deg`.

    Raises:
        ValueError: The rig has no pan-tilt unit.
    """
    rig = scenario.rigs[name]
    if rig.ptu is None:
        raise ValueError(f"rig {name} has no ptu")
    return ptu.panned_tilt_deg(
        attitude_deg,
        axis_deg(scenario, name),
        scenario.outputs.times_s,
        rig.ptu.inclination_deg,
        rig.pitch_deg,
    )


def rig_track(scenario: Scenario, name: str) -> Path:
    """Where the rig `name` is at a time, east and north of the ownship's origin,
    and its axis: the ownship's heading, the rig's yaw and its pan.
    """
    rig = scenario.rigs[name]
    path = track(scenario.ownship, scenario.outputs, scenario.turn)
    axis = axis_deg(scenario, name)

    def at(t_s: float) -> Pose:
        pose = path(t_s)
        h, x, y = pose.heading_rad, rig.offset_x_m, rig.offset_y_m
        return Pose(
            pose.east_m + x * math.cos(h) + y * math.sin(h),
            pose.north_m - x * math.sin(h) + y * math.cos(h),
            h + math.radians(axis(t_s)),
        )

    return at


def drifting(
    range_m: float,
    bearing_deg: float,
    heading_deg: float,
    speed_mps: float,
    drift: Drift | None,
    outputs: Outputs,
) -> Path:
    """From `range_m` out along `bearing_deg` at t = 0, under way along its heading
    and drifting about that pose: a sway across it, and a surge along it at twice the
    rate.
    """
    bearing, heading = math.radians(bearing_deg), math.radians(heading_deg)
    sway_m = surge_m = 0.0
    period_s = 1.0
    if drift is not None:
        sway_m, surge_m = drift.sway_m, drift.surge_m
        period_s = outputs.period_s(drift.period_s)
    sway = sine(0.0, sway_m, period_s)
    surge = sine(0.0, surge_m, period_s / 2.0)
    rate = 2.0 * math.pi / period_s

    def at(t_s: float) -> Pose:
        phase = rate * t_s
        along, across = speed_mps * t_s + surge(t_s), sway(t_s)
        along_mps = speed_mps + 2.0 * rate * surge_m * math.cos(2.0 * phase)
        across_mps = rate * sway_m * math.cos(phase)
        # Starboard of the heading is (cos, -sin).
        sin, cos = math.sin(heading), math.cos(heading)
        return Pose(
            range_m * math.sin(bearing) + along * sin + across * cos,
            range_m * math.cos(bearing) + along * cos - across * sin,
            heading,
            along_mps * sin + across_mps * cos,
            along_mps * cos - across_mps * sin,
        )

    return at


def orbiting(range_m: float, bearing_deg: float, lap_s: float) -> Path:
    """Clockwise round the origin `range_m` out, from `bearing_deg`, once a `lap_s`;
    the bow a right angle ahead of the bearing.
    """
    speed_mps = 2.0 * math.pi * range_m / lap_s

    def at(t_s: float) -> Pose:
        bearing = math.radians(bearing_deg + 360.0 * t_s / lap_s)
        heading = bearing + math.pi / 2.0
        return Pose(
            range_m * math.sin(bearing),
            range_m * math.cos(bearing),
            heading,
            speed_mps * math.sin(heading),
            speed_mps * math.cos(heading),
        )

    return at


def waterplane(
    pose: Pose, tall: tuple[waves.Wave, ...], t_s: float, hull: waves.Waterplane
) -> waves.Waterplane:
    """`hull` at `pose`, centred where the water under it was at rest.

    Args:
        pose: Where it is, where its bow points and how it moves.
        tall: The waves that are geometry, the ones it floats on; the rest only shade.
        t_s: When it is at `pose`.
        hull: Its size; its pose is ignored.
    """
    rest = waves.gerstner_rest_m(
        tall, np.array(pose.east_m), np.array(pose.north_m), t_s
    )
    east, north = map(float, rest)
    return hull._replace(
        rest_east_m=east,
        rest_north_m=north,
        heading_rad=pose.heading_rad,
        east_mps=pose.east_mps,
        north_mps=pose.north_mps,
    )


def rocking(
    pose: Pose,
    hull: waves.Waterplane,
    field: tuple[waves.Wave, ...],
    tall: tuple[waves.Wave, ...],
    t_s: float,
) -> tuple[float, float, float]:
    """The pitch bow up and roll starboard down, in radians, and the heave in metres
    of `hull` at `pose`.

    Args:
        pose: Where it is, where its bow points and how it moves.
        hull: Its size; its pose is ignored.
        field: Every wave, which it pitches and rolls on.
        tall: The waves that are geometry, which alone heave it: the rest are
            shading, which a heaving hull would break its waterline on.
        t_s: The time.
    """
    plane = waterplane(pose, tall, t_s, hull)
    pitch, roll_up = waves.attitude(field, plane, t_s)
    return pitch, -roll_up, waves.heave_m(tall, plane, t_s)


def attitude_deg(
    path: Path,
    hull: waves.Waterplane | None,
    field: tuple[waves.Wave, ...],
    tall: tuple[waves.Wave, ...],
) -> Callable[[float], tuple[float, float]]:
    """The pitch, bow up, and roll, starboard down, at a time: what `path` steers and,
    under `hull`, the sea's.

    Args:
        path: Where it is, where its bow points and how it is tilted.
        hull: Its size, or None for a bare rig, which the sea does not tilt.
        field: Every wave, which a hull pitches and rolls on.
        tall: The waves that are geometry, the ones it floats on.
    """

    def at(t_s: float) -> tuple[float, float]:
        pose = path(t_s)
        pitch, roll = pose.pitch_rad, pose.roll_rad
        if hull is not None:
            sea_pitch, sea_roll, _ = rocking(pose, hull, field, tall, t_s)
            pitch, roll = pitch + sea_pitch, roll + sea_roll
        return math.degrees(pitch), math.degrees(roll)

    return at
