"""The sync course: where its buoys and container go for a rig.

The course turns the cameras past moored objects, so a delay between a video and its
metadata, or between two videos, moves where they see one.
"""

import itertools
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from seascape import angles
from seascape.assets import Buoy, Debris, manifest
from seascape.config import (
    Camera,
    Crop,
    Object,
    Rig,
    Scenario,
    Stitch,
    Turn,
    copies,
    sources,
)
from seascape.motion import rig_track

BUOY, CONTAINER = "mooring_buoy", "container"
# Judgement: weathered, as a buoy left out a season.
BUOY_WEAR = 0.5
# Judgement: a buoy straining at its mooring, half a metre to and fro about a
# minute, sideways to the cameras so it changes range more than bearing. Each buoy
# draws its own period, so they fall out of step.
BUOY_DRIFT = {"sway_m": 0.5, "surge_m": 0.25, "period_s": {"uniform": [45.0, 75.0]}}
# The smallest box a pair's coarser video is assumed to detect (Confluence, Sensors
# Synchronization).
BUOY_PX = 6.0
# Judgement: the share of the narrowest video the container spans.
CONTAINER_FILL = 1.0 / 3.0
# Judgement: what a buoy keeps clear of a frame's edge, and the horizon of its top.
MARGIN_DEG = 0.5
# The decimals a range, a bearing or an inclination is stored to.
DECIMALS = 1


@dataclass(frozen=True, slots=True, kw_only=True)
class View:
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


@dataclass(frozen=True, slots=True, kw_only=True)
class Ring:
    """Buoys at one range: `BUOY_PX` tall in the coarser video of each pair they
    serve.
    """

    range_m: float
    videos: frozenset[str]


@dataclass(frozen=True, slots=True, kw_only=True)
class Course:
    """The course's `[[objects]]` entries, a ring of buoys each, and the inclination
    its pan-tilt unit holds, if any.
    """

    inclination_deg: float | None
    entries: list[dict[str, Any]]

    def objects(self, seed: int) -> list[Object]:
        """The entries as a scenario seeded `seed` loads them, a buoy each."""
        return [Object.model_validate(o) for o in copies(self.entries, seed)]


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
        else:
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
            narrowest = right - left
        found.append(
            View(
                name=name,
                left_deg=left,
                right_deg=right,
                down_deg=down,
                up_deg=up,
                focal_px=focal,
                width_px=width,
                camera_deg=narrowest,
            )
        )
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
    found = (
        Ring(range_m=reach_m(v), videos=frozenset(names)) for v, names in served.items()
    )
    return sorted(found, key=lambda r: r.range_m)


def asset[T](name: str, kind: type[T]) -> T:
    """The course's `name` asset, which must be a `kind`."""
    if not isinstance(found := manifest()[name], kind):
        raise TypeError(
            f"{name} must be a {kind.__name__}, not a {type(found).__name__}"
        )
    return found


def buoy_above_m() -> float:
    """How far the course's buoy stands above the water."""
    b = asset(BUOY, Buoy)
    return b.height_m - b.draught_m


def reach_m(view: View) -> float:
    """How far out a buoy still stands `BUOY_PX` tall in `view`."""
    return buoy_above_m() * view.focal_px / BUOY_PX


def stored_m(ring: Ring, rig: Rig) -> float:
    """The range `ring`'s entry stores: round the ownship, `rig`'s offset nearer, so
    no buoy is farther from the cameras than the ring's range.
    """
    return round(ring.range_m - math.hypot(rig.offset_x_m, rig.offset_y_m), DECIMALS)


def half_deg(wide_m: float, range_m: float) -> float:
    """Half the angle across something `wide_m` wide `range_m` out."""
    return math.degrees(math.atan(wide_m / 2 / range_m))


def buoy_half_deg(range_m: float) -> float:
    """Half the angle across a buoy `range_m` out; judgement: as wide as it stands
    above the water.
    """
    return half_deg(buoy_above_m(), range_m)


def one_rig(scenario: Scenario) -> tuple[str, Rig, Turn]:
    """The one rig a course is for, and its turn."""
    if scenario.turn is None or len(scenario.rigs) != 1:
        turn = "a turn" if scenario.turn is not None else "no turn"
        raise ValueError(
            f"a course needs a turn and one rig: the scenario has {turn} and "
            f"{len(scenario.rigs)} rigs"
        )
    ((name, rig),) = scenario.rigs.items()
    return name, rig, scenario.turn


def _inclination_deg(rig: Rig, videos: list[View], near_m: float) -> float | None:
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
        return pitch_deg
    if not least <= rig.pitch_deg <= most:
        raise ValueError(
            f"a buoy {near_m:.0f} m out or the horizon leaves a video: set the rig's "
            f"pitch_deg to {pitch_deg}"
        )
    return None


def course(scenario: Scenario) -> Course:
    """The course for `scenario`'s one rig and its turn: a container at the bow, and
    a ring of buoys round the ownship for each video that is the coarser of a pair,
    spaced within a camera's span so each camera they record passes one, staggered
    so no ring hides another.

    Raises:
        ValueError: No turn, not one rig, an ownship under way, or no pitch that
            keeps every ring in view.
    """
    name, rig, turn = one_rig(scenario)
    # The container is placed once, where the rig starts: a hull under way leaves it.
    if scenario.ownship.speed_mps:
        raise ValueError(
            f"the ownship makes {scenario.ownship.speed_mps:g} m/s: a course needs "
            "it still, so set ownship.speed_mps to 0"
        )
    videos = views(rig)
    if not (circles := rings(videos)):
        raise ValueError("no two videos of the rig overlap: a course needs a pair")
    # A ring is centred on the ownship, so its nearest buoy stands two of the rig's
    # offsets nearer the cameras than the ring's range.
    off_m = math.hypot(rig.offset_x_m, rig.offset_y_m)
    inclination_deg = _inclination_deg(rig, videos, circles[0].range_m - 2 * off_m)

    start = rig_track(scenario, name)(0.0)
    narrow = min(videos, key=lambda v: v.right_deg - v.left_deg)
    box = asset(CONTAINER, Debris)
    box_m = box.length_m * narrow.focal_px / (CONTAINER_FILL * narrow.width_px)
    # Limitation: every video must hold it, so only a rig that looks forward fits.
    middle = math.radians(rig.yaw_deg + (narrow.left_deg + narrow.right_deg) / 2)
    east = start.east_m + box_m * math.sin(middle)
    north = start.north_m + box_m * math.cos(middle)
    # Broadside to the line of sight from the origin.
    entries: list[dict[str, Any]] = [
        {
            "asset": CONTAINER,
            "range_m": round(math.hypot(east, north), DECIMALS),
            "bearing_deg": round(angles.bearing_deg(east, north), DECIMALS),
            "heading_deg": 90.0,
            "heading_from": "line_of_sight",
        }
    ]
    colours = {"choice": sorted(asset(BUOY, Buoy).sold_in)}
    stops_deg = [0.0, *(move.to_deg for move in turn.moves)]
    for k, ring in enumerate(circles):
        # Each camera of a stitch gets its own crossings.
        span_deg = min(v.camera_deg for v in videos if v.name in ring.videos)
        # Down to a stored decimal, so the stored step still fits the camera.
        fits_deg = span_deg - 2 * (buoy_half_deg(ring.range_m) + MARGIN_DEG)
        step_deg = math.floor(fits_deg * 10**DECIMALS) / 10**DECIMALS
        if step_deg <= 0.0:
            raise ValueError(
                f"a camera {span_deg:.1f} deg wide cannot hold a buoy "
                f"{ring.range_m:.0f} m out whole"
            )
        swept_deg = max(stops_deg) - min(stops_deg)
        count = math.floor(swept_deg / step_deg) + 1
        # Centred on the sweep, each ring shifted a share of a step off the last.
        first_deg = (
            min(stops_deg)
            + (swept_deg - (count - 1) * step_deg) / 2
            + step_deg * (k / len(circles) - (len(circles) - 1) / (2 * len(circles)))
        )
        entries.append(
            {
                "asset": BUOY,
                "range_m": stored_m(ring, rig),
                "bearing_deg": round(rig.yaw_deg + first_deg, DECIMALS),
                "count": count,
                "spacing_deg": step_deg,
                # Each buoy draws its own.
                "colour": colours,
                "wear": BUOY_WEAR,
                # Bow across the line of sight, so its sway runs along it.
                "heading_deg": 90.0,
                "heading_from": "line_of_sight",
                "drift": BUOY_DRIFT,
            }
        )
    return Course(inclination_deg=inclination_deg, entries=entries)


def verify(scenario: Scenario) -> None:
    """Fail unless `scenario` stores the course `course` places for it.

    Raises:
        ValueError: The setting to change, or the `objects` to paste.
    """
    name, rig, _ = one_rig(scenario)
    placed = course(scenario)
    stored_deg = None if rig.ptu is None else rig.ptu.inclination_deg
    if placed.inclination_deg != stored_deg:
        raise ValueError(
            f"the course needs [rigs.{name}.ptu] "
            f"inclination_deg = {placed.inclination_deg}"
        )
    if placed.objects(scenario.seed) != scenario.objects:
        raise ValueError(
            f"{objects_toml(placed.entries)}\n\nthe objects are not the course's for "
            "its rig: replace its [[objects]] tables with those above"
        )


def objects_toml(entries: list[dict[str, Any]]) -> str:
    """`entries` as a scenario's `[[objects]]` tables."""
    return "\n\n".join(
        "[[objects]]\n"
        + "\n".join(f"{key} = {_toml(value)}" for key, value in e.items())
        for e in entries
    )


def _toml(value: object) -> str:
    """`value` as a TOML value, a table inline."""
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{k} = {_toml(v)}" for k, v in value.items()) + " }"
    if isinstance(value, list):
        return "[" + ", ".join(_toml(v) for v in value) + "]"
    # A JSON string, number or boolean is TOML's too.
    return json.dumps(value)
