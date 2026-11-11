"""Where a hull is at a time, and the waterplane the waves tilt it by there."""

import math
from collections.abc import Callable
from typing import NamedTuple

import numpy as np

from seascape.config import Outputs, Ownship
from seascape.physics import waves

# A judgement: trapezoid steps per yaw period; the error falls with the square.
STEPS_PER_YAW = 64


class Pose(NamedTuple):
    """East and north of the start; the heading is clockwise from north."""

    east_m: float
    north_m: float
    heading_rad: float


class HullSize(NamedTuple):
    """A hull's size as the waves see it: its waterplane, and the box below it. A
    point by default, which follows the water.
    """

    length_m: float = 0.0
    beam_m: float = 0.0
    box_beam_m: float = 0.0
    box_draught_m: float = 0.0


POINT = HullSize()


def track(ownship: Ownship, outputs: Outputs) -> Callable[[float], Pose]:
    """The pose at a time: `speed_mps` integrated along the heading `yaw` swings."""
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

    east = ownship.speed_mps * integral(np.sin(heading))
    north = ownship.speed_mps * integral(np.cos(heading))

    def at(t_s: float) -> Pose:
        return Pose(*(float(np.interp(t_s, tau, c)) for c in (east, north, heading)))

    return at


def waterplane(
    pose: Pose,
    tall: tuple[waves.Wave, ...],
    t_s: float,
    size: HullSize = POINT,
    speed_mps: float = 0.0,
) -> waves.Waterplane:
    """A hull of `size` at `pose`, centred where the water under it was at rest.

    Args:
        pose: Where it is and where its bow points.
        tall: The waves that are geometry, the ones it floats on; the rest only shade.
        t_s: The time the water moved to `pose`.
        size: Waterplane and box; a point, the default, follows the water.
        speed_mps: Its speed along the heading.
    """
    rest = waves.gerstner_rest_m(
        tall, np.array(pose.east_m), np.array(pose.north_m), t_s
    )
    heading = pose.heading_rad
    return waves.Waterplane(
        *map(float, rest),
        heading_rad=heading,
        **size._asdict(),
        east_mps=speed_mps * math.sin(heading),
        north_mps=speed_mps * math.cos(heading),
    )
