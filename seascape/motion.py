"""Where a hull is at a time, and the waterplane the waves tilt it by there."""

import bisect
import math
from collections.abc import Callable
from typing import NamedTuple

import numpy as np

from seascape import angles
from seascape.config import Fishtail, Outputs, Ownship, Scenario, Turn
from seascape.config.objects import Swing
from seascape.physics import earth, ptu, slew, waves

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
# (3, 3) `R_level_ownship` at a time: the hull's tilt, its heading aside.
type Tilt = Callable[[float], np.ndarray]


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
    """The pan of the rig named `rig` at a time, clockwise from its base: the turn's
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


def azimuth_deg(scenario: Scenario, name: str) -> Callable[[float], float]:
    """The azimuth of the head of rig `name` at a time, clockwise from the bow: its
    base's yaw and its pan.
    """
    rig, panned = scenario.rigs[name], panned_deg(scenario.turn, name)
    return lambda t_s: rig.azimuth_deg(panned(t_s))


def head_path(scenario: Scenario, name: str) -> Path:
    """Where the head of rig `name` is at a time, in world east and north, and the
    bearing of its axis, on a hull tilted by the ownship's scripted pitch and roll.

    Limitation: leaves out the sea's rocking and a ptu's tilt; staging and the sheet
    still aim the course along the base's yaw on a level hull.
    """
    rig = scenario.rigs[name]
    path = track(scenario.ownship, scenario.outputs, scenario.turn)
    waterline = waterline_pose(path, at_sea_level)
    T_waterline_ownship = lifted(scenario.ownship.rise_m)
    pan = panned_deg(scenario.turn, name)

    def at(t_s: float) -> Pose:
        T_world_head = waterline(t_s) @ T_waterline_ownship @ rig.pose(pan(t_s))
        east, north, _ = T_world_head[:3, 1]
        return Pose(
            float(T_world_head[0, 3]),
            float(T_world_head[1, 3]),
            math.radians(angles.bearing_deg(east, north)),
        )

    return at


def target_path(
    range_m: float,
    bearing_deg: float,
    heading_deg: float,
    speed_mps: float,
    fishtail: Fishtail | None,
    outputs: Outputs,
) -> Path:
    """From `range_m` out along `bearing_deg` at t = 0, under way along its heading
    and fishtailing about that pose: a sway across it, and a surge along it at twice the
    rate. Bearings are from the ownship's bow at the start.
    """
    start = range_m * angles.direction(bearing_deg, 0.0)[:2]
    # Columns: the hull's starboard and bow, in (east, north).
    R_world_target = angles.turned(heading_deg, 0.0)[:2, :2]
    heading = math.radians(heading_deg)
    sway_m = surge_m = 0.0
    period_s = 1.0
    if fishtail is not None:
        sway_m, surge_m = fishtail.sway_m, fishtail.surge_m
        period_s = outputs.period_s(fishtail.period_s)
    sway = sine(0.0, sway_m, period_s)
    surge = sine(0.0, surge_m, period_s / 2.0)
    rate = 2.0 * math.pi / period_s

    def at(t_s: float) -> Pose:
        phase = rate * t_s
        along, across = speed_mps * t_s + surge(t_s), sway(t_s)
        along_mps = speed_mps + 2.0 * rate * surge_m * math.cos(2.0 * phase)
        across_mps = rate * sway_m * math.cos(phase)
        east, north = start + R_world_target @ (across, along)
        east_mps, north_mps = R_world_target @ (across_mps, along_mps)
        return Pose(
            float(east), float(north), heading, float(east_mps), float(north_mps)
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


class Rocking(NamedTuple):
    """What the sea does to a hull: its pitch, bow up, its roll, starboard down, and
    its heave.
    """

    pitch_rad: float
    roll_rad: float
    heave_m: float


def rocking(
    pose: Pose,
    *,
    hull: waves.Waterplane,
    field: tuple[waves.Wave, ...],
    tall: tuple[waves.Wave, ...],
    t_s: float,
) -> Rocking:
    """How the sea rocks `hull` at `pose`.

    Args:
        pose: Where it is, where its bow points and how it moves.
        hull: Its size; its pose is ignored.
        field: Every wave, which it pitches and rolls on.
        tall: The waves that are geometry, which alone heave it: the rest are
            shading, which a heaving hull would break its waterline on.
        t_s: Seconds on the scenario's clock, when it is at `pose`.
    """
    plane = waterplane(pose, tall, t_s, hull)
    pitch, roll = waves.attitude(field, plane, t_s)
    return Rocking(pitch, roll, waves.heave_m(tall, plane, t_s))


def _pitched(rad: float) -> np.ndarray:
    """Bow up by `rad`: about +x, starboard."""
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rolled(rad: float) -> np.ndarray:
    """Starboard down by `rad`: about +y, the bow."""
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def tilt_rad(pose: Pose, sea: Rocking | None) -> tuple[float, float]:
    """A hull's pitch, bow up, and roll, starboard down, in radians: the sea's tilt
    `sea` outside the trim and list `pose` scripts, which are relative to the water.
    Two tilts composed also yaw the hull; that yaw is dropped, as the hull's heading
    is its track's (Fossen, Handbook of Marine Craft Hydrodynamics, 2011, §2.2: yaw
    of the whole attitude).
    """
    if sea is None:
        return pose.pitch_rad, pose.roll_rad
    R_level_hull = (
        _pitched(sea.pitch_rad)
        @ _rolled(sea.roll_rad)
        @ _pitched(pose.pitch_rad)
        @ _rolled(pose.roll_rad)
    )
    # As `angles.attitude_deg` reads them: the up row, which a yaw leaves alone.
    return (
        math.asin(R_level_hull[2, 1]),
        math.atan2(-R_level_hull[2, 0], R_level_hull[2, 2]),
    )


def tilted(
    path: Path,
    hull: waves.Waterplane | None,
    field: tuple[waves.Wave, ...],
    tall: tuple[waves.Wave, ...],
) -> Tilt:
    """`R_level_ownship` at a time: its `tilt_rad` by the pitch and roll `path` scripts
    and, under `hull`, the sea's.

    Args:
        path: Where it is, where its bow points and how it is tilted.
        hull: Its size, or None for a bare rig, which the sea does not tilt.
        field: Every wave, which a hull pitches and rolls on.
        tall: The waves that are geometry, the ones it floats on.
    """
    return lambda t_s: _afloat(path, hull, field, tall, t_s)[0]


def _afloat(
    path: Path,
    hull: waves.Waterplane | None,
    field: tuple[waves.Wave, ...],
    tall: tuple[waves.Wave, ...],
    t_s: float,
) -> tuple[np.ndarray, float]:
    """`R_level_waterline` and the sea's heave at `t_s`; `tilted`'s arguments."""
    pose = path(t_s)
    if hull is None:
        return _pitched(pose.pitch_rad) @ _rolled(pose.roll_rad), 0.0
    sea = rocking(pose, hull=hull, field=field, tall=tall, t_s=t_s)
    pitch_rad, roll_rad = tilt_rad(pose, sea)
    return _pitched(pitch_rad) @ _rolled(roll_rad), sea.heave_m


def waterline_pose(
    path: Path,
    up_m: Callable[[Pose, float], float],
    hull: waves.Waterplane | None = None,
    field: tuple[waves.Wave, ...] = (),
    tall: tuple[waves.Wave, ...] = (),
) -> Callable[[float], np.ndarray]:
    """(4, 4) `T_world_waterline` at a time: the point on a hull's floating waterline
    it pitches and rolls about, `up_m` high plus the sea's heave, turned by `path`'s
    heading and tilted as `tilted`. The hull's origin is `lifted(rise_m)` above it.

    Args:
        path: Where it is, where its bow points and how it is tilted.
        up_m: The point's height at a pose and time, the sea's heave aside.
        hull: Its size, or None for a bare rig, which the sea does not tilt.
        field: Every wave, which a hull pitches and rolls on.
        tall: The waves that are geometry, the ones it floats on.
    """

    def at(t_s: float) -> np.ndarray:
        pose = path(t_s)
        R_level_waterline, heave_m = _afloat(path, hull, field, tall, t_s)
        T_world_waterline = np.eye(4)
        T_world_waterline[:3, :3] = (
            angles.turned(math.degrees(pose.heading_rad), 0.0) @ R_level_waterline
        )
        T_world_waterline[:3, 3] = pose.east_m, pose.north_m, up_m(pose, t_s) + heave_m
        return T_world_waterline

    return at


def lifted(rise_m: float) -> np.ndarray:
    """(4, 4) a hull's origin on its waterline pivot, `rise_m` up the hull."""
    T = np.eye(4)
    T[2, 3] = rise_m
    return T


def on_sea(radius_m: float) -> Callable[[Pose, float], float]:
    """The curved sea's height under a pose: a hull left at z = 0 flies above it.

    Not tilted to the local vertical: at any range the sea reaches, range / R moves a
    hull's ends far less than its draught.
    """
    return lambda pose, _t_s: earth.sea_z_m(pose.east_m, pose.north_m, radius_m)


def at_sea_level(_pose: Pose, _t_s: float) -> float:
    """A waterline at the world's z = 0, as on a flat sea at the start."""
    return 0.0


def world_elevation_deg(R_level_ownship: np.ndarray, toward: np.ndarray) -> float:
    """The elevation in the world of `toward`, a unit (3,) in the ownship's frame,
    on a hull tilted `R_level_ownship`, (3, 3).
    """
    return float(angles.elevation_deg(R_level_ownship[2] @ toward))


def held_tilt_deg(scenario: Scenario, name: str, tilt: Tilt) -> np.ndarray:
    """The tilt the pan-tilt unit of rig `name` adds to the rig's `pitch_deg` at each
    frame, holding its elevation on a hull tilted `tilt`.

    Raises:
        ValueError: The rig has no pan-tilt unit, or the deck tips its untilted head
            past the zenith, where the IMU's pitch folds and the loop holds the head
            facing aft.
    """
    rig = scenario.rigs[name]
    if rig.ptu is None:
        raise ValueError(f"rig {name} has no ptu")
    pan = panned_deg(scenario.turn, name)

    def head(t_s: float, tilt_deg: float = 0.0) -> np.ndarray:
        """The head's axis on the deck."""
        return rig.pose(pan(t_s), tilt_deg)[:3, 1]

    times = scenario.outputs.times_s
    for t_s in times:
        # Past the zenith, the head's horizontal part turns against its bearing; the
        # deck turns no heading, so the bearing on a level deck is the reference.
        tipped = tilt(t_s) @ head(t_s)
        if tipped[:2] @ head(t_s, -rig.pitch_deg)[:2] < 0.0:
            raise ValueError(
                f"rigs.{name}: at {t_s:g} s the deck and the rig's pitch_deg tip its "
                "ptu past the zenith: lower the ownship's pitch, the sea or pitch_deg"
            )
        # Tilting sweeps the head through the deck's vertical plane at its bearing;
        # the world's up projects onto that plane with length `reach`, the sine of
        # the steepest elevation any tilt gives.
        up = tilt(t_s)[2]
        reach = math.hypot(up @ head(t_s, -rig.pitch_deg), up[2])
        if abs(math.sin(math.radians(rig.ptu.elevation_deg))) > reach:
            raise ValueError(
                f"rigs.{name}: at {t_s:g} s no tilt brings the ptu's head to "
                f"elevation_deg {rig.ptu.elevation_deg:g}: lower the ownship's "
                "roll or the sea"
            )

    # The firmware's IMU sits on the head, so the loop holds the head itself,
    # `pitch_deg` included.
    def head_deg(t_s: float, tilt_deg: float) -> float:
        return world_elevation_deg(tilt(t_s), head(t_s, tilt_deg))

    return ptu.tilt_deg(head_deg, times, rig.ptu.elevation_deg)


def head_pose(
    scenario: Scenario, name: str, tilt: Tilt
) -> Callable[[float], np.ndarray]:
    """(4, 4) `T_ownship_head` of rig `name` at a frame's time: its pan, and a pan-tilt
    unit's tilt holding its elevation on a hull tilted `tilt`.

    Raises:
        KeyError: On a pan-tilt unit, called at a time that is no frame's, where the
            loop holds no tilt.
    """
    rig = scenario.rigs[name]
    pan = panned_deg(scenario.turn, name)
    if rig.ptu is None:
        return lambda t_s: rig.pose(pan(t_s))
    times = scenario.outputs.times_s
    held = dict(
        zip(times, map(float, held_tilt_deg(scenario, name, tilt)), strict=True)
    )
    return lambda t_s: rig.pose(pan(t_s), held[t_s])


def cameras_at_start(scenario: Scenario, name: str) -> dict[str, np.ndarray]:
    """Each camera rig `name` renders, by name, as `R_world_camera` in OpenCV's axes
    at the start: a pan-tilt unit holds its elevation against the ownship's scripted
    tilt on a level sea.
    """
    path = track(scenario.ownship, scenario.outputs, scenario.turn)
    tilt = tilted(path, hull=None, field=(), tall=())
    T_world_head = (
        waterline_pose(path, at_sea_level)(0.0)
        @ lifted(scenario.ownship.rise_m)
        @ head_pose(scenario, name, tilt)(0.0)
    )
    return {
        camera_name: (T_world_head @ camera.pose())[:3, :3]
        for camera_name, camera in scenario.rigs[name].rendered.items()
    }


def ownship_up_m(
    ownship: Ownship, outputs: Outputs, tall: tuple[waves.Wave, ...], radius_m: float
) -> Callable[[Pose, float], float]:
    """The ownship's waterline height at a pose and time, the sea's heave aside: the
    curved sea under it and its scripted heave, and a bare rig's heave on `tall`.

    Args:
        ownship: Its hull, if any, and heave.
        outputs: The periods its heave loops on.
        tall: The waves that are geometry, which heave a bare rig.
        radius_m: Earth's effective radius, which drops it to the curved sea.
    """
    scripted = sine(0.0, 0.0, 1.0)
    if (heave := ownship.heave) is not None:
        scripted = sine(0.0, heave.amplitude_m, outputs.period_s(heave.period_s))
    # A bare rig is a point: it heaves, but would tilt with every ripple.
    bare = ownship.asset is None and bool(tall)
    sea = on_sea(radius_m)

    def up_m(pose: Pose, t_s: float) -> float:
        z = sea(pose, t_s) + scripted(t_s)
        if bare:
            point = waterplane(pose, tall, t_s, waves.Waterplane())
            z += waves.heave_m(tall, point, t_s)
        return z

    return up_m
