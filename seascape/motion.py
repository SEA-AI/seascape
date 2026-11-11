"""Where a hull is at a time, and the waterplane the waves tilt it by there."""

import math
from collections.abc import Callable
from typing import NamedTuple

import numpy as np

from seascape.config import Drift, Outputs, Ownship
from seascape.config.objects import Swing
from seascape.physics import waves

# Judgement: trapezoid steps per yaw period; the error falls with the square.
STEPS_PER_YAW = 64


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


def track(ownship: Ownship, outputs: Outputs) -> Path:
    """The pose at a time: `speed_mps` integrated along the heading `yaw` swings, and
    the pitch and roll its swings script.
    """
    times = np.asarray(outputs.times_s)
    swing = ownship.yaw
    # Whole substeps per frame, so every frame lands on the grid; a straight run is
    # exact in one.
    sub = 1
    if swing is not None:
        step_s = outputs.period_s(swing.period_s) / STEPS_PER_YAW
        sub = math.ceil(1.0 / (outputs.fps * step_s))
    tau = np.linspace(0.0, times[-1], (len(times) - 1) * sub + 1)
    heading = np.zeros_like(tau)
    if swing is not None:
        omega = 2.0 * math.pi / outputs.period_s(swing.period_s)
        heading = math.radians(swing.amplitude_deg) * np.sin(omega * tau)

    def integral(speed: np.ndarray) -> np.ndarray:
        area = np.diff(tau) * (speed[1:] + speed[:-1]) / 2.0
        return np.concatenate(([0.0], np.cumsum(area)))

    speed = ownship.speed_mps
    east = speed * integral(np.sin(heading))
    north = speed * integral(np.cos(heading))
    pitch = swung(ownship.pitch_deg, ownship.pitch, outputs)
    roll = swung(ownship.roll_deg, ownship.roll, outputs)

    def at(t_s: float) -> Pose:
        e, n, h = (float(np.interp(t_s, tau, c)) for c in (east, north, heading))
        v_e, v_n = speed * math.sin(h), speed * math.cos(h)
        return Pose(e, n, h, v_e, v_n, pitch(t_s), roll(t_s))

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
