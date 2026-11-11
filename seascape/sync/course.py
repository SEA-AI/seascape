"""The sync course: where its buoys and container go for a rig.

The course turns the cameras past still objects, so a delay between a video and its
metadata, or between two videos, moves where they see one.
"""

import itertools
import json
import math
from collections import defaultdict
from collections.abc import Callable
from typing import NamedTuple

from seascape.assets import Buoy, Debris, manifest
from seascape.config import Camera, Crop, Object, Rig, Scenario, Stitch, Turn, sources
from seascape.motion import Pose, track, turned_deg

BUOY, CONTAINER = "mooring_buoy", "container"
BUOY_COLOUR = "orange"
# Judgement: weathered, as a buoy left out a season.
BUOY_WEAR = 0.5
# The smallest box a pair's coarser video is assumed to detect (Confluence, Sensors
# Synchronization).
BUOY_PX = 6.0
# Judgement: the share of the narrowest video the container spans.
CONTAINER_FILL = 1.0 / 3.0
# Judgement: what a buoy keeps clear of a frame's edge, and the horizon of its top.
MARGIN_DEG = 0.5
# The decimals a range, a bearing or an inclination is stored to.
DECIMALS = 1


class View(NamedTuple):
    """A video's span of bearings and elevations from its rig's axis, and its
    pixels.
    """

    name: str
    left_deg: float
    right_deg: float
    down_deg: float
    up_deg: float
    focal_px: float
    width_px: float
    # The narrowest span one of its cameras covers: a stitch's camera's, else its own.
    camera_deg: float


class Ring(NamedTuple):
    """Buoys at one range: 6 px tall in the coarser video of each pair they serve."""

    range_m: float
    videos: frozenset[str]


class Course(NamedTuple):
    """The course's objects, and the inclination its pan-tilt unit holds."""

    inclination_deg: float
    objects: list[Object]


def _rows_deg(camera: Camera, top_px: int, bottom_px: int) -> tuple[float, float]:
    """The elevations of the lowest and highest rows a video keeps of `camera`."""
    half, f = camera.height_px / 2, camera.focal_px
    return (
        camera.pitch_deg - math.degrees(math.atan((half - bottom_px) / f)),
        camera.pitch_deg + math.degrees(math.atan((half - top_px) / f)),
    )


def views(rig: Rig) -> list[View]:
    """Each video, or each camera where the rig records them whole."""
    cameras = rig.rendered
    videos = rig.videos or dict.fromkeys(cameras)
    found = []
    for name, video in videos.items():
        recorded = [cameras[c] for c in (sources(video) if video else (name,))]
        if any(c.lens.fisheye is not None for c in recorded):
            raise ValueError(f"{name} records a fisheye: undistort it")
        if isinstance(video, Stitch):
            pair = sorted(recorded, key=lambda c: c.yaw_deg)
            (left, _), (_, right) = (
                (c.yaw_deg - c.hfov_deg / 2, c.yaw_deg + c.hfov_deg / 2) for c in pair
            )
            # Limitation: `stitch.fused` crops an asymmetric pair to its axis.
            if not math.isclose(left, -right, abs_tol=10.0**-DECIMALS):
                raise ValueError(f"{name} stitches a pair asymmetric about the rig")
            focal = sum(c.focal_px for c in pair) / 2
            width = 2 * focal * math.tan(math.radians(right))
            rows = [_rows_deg(c, 0, 0) for c in pair]
            down, up = max(r[0] for r in rows), min(r[1] for r in rows)
            narrowest = min(c.hfov_deg for c in pair)
            found.append(View(name, left, right, down, up, focal, width, narrowest))
            continue
        (camera,) = recorded
        focal, half = camera.focal_px, camera.width_px / 2
        left_px = right_px = top_px = bottom_px = 0
        if isinstance(video, Crop):
            left_px, right_px = video.left_px, video.right_px
            top_px, bottom_px = video.top_px, video.bottom_px
        left, right = (
            camera.yaw_deg + math.degrees(math.atan(px / focal))
            for px in (left_px - half, half - right_px)
        )
        down, up = _rows_deg(camera, top_px, bottom_px)
        width = camera.width_px - left_px - right_px
        found.append(View(name, left, right, down, up, focal, width, right - left))
    return found


def overlap_deg(a: View, b: View) -> tuple[float, float] | None:
    """The bearings off the rig both videos see, either way round the stern."""
    for turn_deg in (-360.0, 0.0, 360.0):
        low = max(a.left_deg, b.left_deg + turn_deg)
        high = min(a.right_deg, b.right_deg + turn_deg)
        if low < high:
            return low, high
    return None


def rings(videos: list[View]) -> list[Ring]:
    """A ring for each video that is the coarser of a pair whose spans overlap, out
    where a buoy stands `BUOY_PX` tall in it, nearest first.
    """
    served: dict[View, set[str]] = defaultdict(set)
    for a, b in itertools.combinations(videos, 2):
        if overlap_deg(a, b) is not None:
            coarser = min(a, b, key=lambda v: v.focal_px)
            served[coarser] |= {a.name, b.name}
    return sorted(Ring(reach_m(v), frozenset(names)) for v, names in served.items())


def asset[T](name: str, kind: type[T]) -> T:
    """The course's `name` asset, which must be a `kind`."""
    if not isinstance(found := manifest()[name], kind):
        raise TypeError(f"{name} must be a {kind.__name__}")
    return found


def buoy_above_m() -> float:
    """How far the course's buoy stands above the water."""
    b = asset(BUOY, Buoy)
    return b.height_m - b.draught_m


def reach_m(view: View) -> float:
    """How far out a buoy still stands `BUOY_PX` tall in `view`."""
    return buoy_above_m() * view.focal_px / BUOY_PX


def half_deg(range_m: float) -> float:
    """Half the angle across a buoy `range_m` out; judgement: as wide as it stands
    above the water.
    """
    return math.degrees(math.atan(buoy_above_m() / 2 / range_m))


def box_half_deg(range_m: float) -> float:
    """Half the angle across the container `range_m` out, side on."""
    return math.degrees(math.atan(asset(CONTAINER, Debris).length_m / 2 / range_m))


def one_rig(scenario: Scenario) -> tuple[str, Rig, Turn]:
    """The one rig a course is for, and its turn."""
    if scenario.turn is None or len(scenario.rigs) != 1:
        raise ValueError("a course needs a turn and one rig")
    ((name, rig),) = scenario.rigs.items()
    return name, rig, scenario.turn


def _inclination_deg(rig: Rig, videos: list[View], near_m: float) -> float:
    """The pitch that keeps the horizon and the nearest ring's waterline in every
    video, as far from either edge as it can.

    Raises:
        ValueError: No pitch does, or a fixed mount's is not one.
    """
    low_deg = -math.degrees(math.atan(rig.height_m / near_m)) - MARGIN_DEG
    least = max(MARGIN_DEG - v.up_deg for v in videos)
    most = min(low_deg - v.down_deg for v in videos)
    if least > most:
        raise ValueError(
            f"no pitch keeps the horizon and a buoy {near_m:.0f} m out in every "
            "video: the rig's vertical fields are too narrow for its height"
        )
    pitch_deg = round((least + most) / 2, DECIMALS)
    if rig.ptu is not None:
        return round(pitch_deg - rig.pitch_deg, DECIMALS)
    if not least <= rig.pitch_deg <= most:
        raise ValueError(
            f"a buoy {near_m:.0f} m out or the horizon leaves a video: set the rig's "
            f"pitch_deg to {pitch_deg}"
        )
    return 0.0


def _placed(
    asset: str,
    at_m: tuple[float, float],
    bearing_deg: float,
    range_m: float,
    colour: str | None = None,
    wear: float = 0.0,
) -> Object:
    """`asset` `range_m` out along `bearing_deg` from a camera at `at_m`, east and
    north of the ownship's origin, broadside to the line of sight from the origin.
    """
    bearing = math.radians(bearing_deg)
    east = at_m[0] + range_m * math.sin(bearing)
    north = at_m[1] + range_m * math.cos(bearing)
    return Object(
        asset=asset,
        range_m=round(math.hypot(east, north), DECIMALS),
        bearing_deg=round(math.degrees(math.atan2(east, north)), DECIMALS),
        heading_deg=90.0,
        heading_from="line_of_sight",
        colour=colour,
        wear=wear,
    )


def rig_track(scenario: Scenario) -> Callable[[float], Pose]:
    """Where the one rig is at a time, east and north of the ownship's origin, and
    its axis: the ownship's heading, the rig's yaw and a turn by its ptu.
    """
    _, rig, turn = one_rig(scenario)
    path = track(scenario.ownship, scenario.outputs, turn)
    panned = turned_deg(turn) if turn.by == "ptu" else lambda _: 0.0

    def at(t_s: float) -> Pose:
        pose = path(t_s)
        h, x, y = pose.heading_rad, rig.offset_x_m, rig.offset_y_m
        return Pose(
            pose.east_m + x * math.cos(h) + y * math.sin(h),
            pose.north_m - x * math.sin(h) + y * math.cos(h),
            h + math.radians(rig.yaw_deg + panned(t_s)),
        )

    return at


def course(scenario: Scenario) -> Course:
    """The course for `scenario`'s one rig and its turn: a container at the bow, and
    a ring of buoys for each pair of videos, spaced so the turn always has one
    crossing every camera they record, staggered so no ring hides another.

    Raises:
        ValueError: No turn, not one rig, or no pitch that keeps every ring in view.
    """
    _, rig, turn = one_rig(scenario)
    videos = views(rig)
    if not (circles := rings(videos)):
        raise ValueError("no two videos of the rig overlap: a course needs a pair")
    inclination_deg = _inclination_deg(rig, videos, circles[0].range_m)

    rig_at = rig_track(scenario)
    turned = turned_deg(turn)
    times = scenario.outputs.times_s
    pointing = [turned(t) for t in times]

    def camera_at(angle_deg: float) -> tuple[float, float]:
        """Where the rig is when the turn first points it along `angle_deg`."""
        apart = [abs(a - angle_deg) for a in pointing]
        # A staggered ring's end buoy may lie a little past the turn's reach.
        near = min(apart) + turn.top_rate_deg_s / scenario.outputs.fps
        pose = rig_at(next(t for t, d in zip(times, apart, strict=True) if d <= near))
        return pose.east_m, pose.north_m

    narrow = min(videos, key=lambda v: v.right_deg - v.left_deg)
    box = asset(CONTAINER, Debris)
    box_m = box.length_m * narrow.focal_px / (CONTAINER_FILL * narrow.width_px)
    middle_deg = (narrow.left_deg + narrow.right_deg) / 2
    # Limitation: every video must hold it, so only a rig that looks forward fits.
    placed = [_placed(CONTAINER, camera_at(0.0), rig.yaw_deg + middle_deg, box_m)]
    angles = [0.0, *(move.to_deg for move in turn.moves)]
    for k, ring in enumerate(circles):
        # Each camera of a stitch gets its own crossings.
        span_deg = min(v.camera_deg for v in videos if v.name in ring.videos)
        step_deg = span_deg - 2 * (half_deg(ring.range_m) + MARGIN_DEG)
        swept_deg = max(angles) - min(angles)
        count = math.floor(swept_deg / step_deg) + 1
        # Centred on the sweep, each ring shifted a share of a step off the last.
        first_deg = (
            min(angles)
            + (swept_deg - (count - 1) * step_deg) / 2
            + step_deg * (k / len(circles) - (len(circles) - 1) / (2 * len(circles)))
        )
        placed += [
            _placed(
                BUOY,
                camera_at(angle_deg),
                rig.yaw_deg + angle_deg,
                ring.range_m,
                colour=BUOY_COLOUR,
                wear=BUOY_WEAR,
            )
            for angle_deg in (first_deg + i * step_deg for i in range(count))
        ]
    return Course(inclination_deg, placed)


def verify(scenario: Scenario) -> None:
    """Fail unless `scenario` stores the course `course` places for it.

    Raises:
        ValueError: The setting to change, or the `objects` to paste.
    """
    name, rig, _ = one_rig(scenario)
    placed = course(scenario)
    stored_deg = 0.0 if rig.ptu is None else rig.ptu.inclination_deg
    if placed.inclination_deg != stored_deg:
        raise ValueError(
            f"the course needs [rigs.{name}.ptu] "
            f"inclination_deg = {placed.inclination_deg}"
        )
    if placed.objects != scenario.objects:
        raise ValueError(
            f"{objects_toml(placed.objects)}\nthe objects are not the course's for "
            "its rig: replace its objects with the array above, before its first table"
        )


def objects_toml(objects: list[Object]) -> str:
    """`objects` as a scenario's `objects` array, a table a line, each field off its
    default.
    """
    rows = (_toml(o.model_dump(exclude_defaults=True)) for o in objects)
    return "objects = [\n" + "".join(f"    {row},\n" for row in rows) + "]"


def _toml(value: object) -> str:
    """`value` as a TOML value, a table inline."""
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{k} = {_toml(v)}" for k, v in value.items()) + " }"
    if isinstance(value, list):
        return "[" + ", ".join(_toml(v) for v in value) + "]"
    # A JSON string, number or boolean is TOML's too.
    return json.dumps(value)
