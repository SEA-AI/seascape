"""The sync preview: the frames it renders, and the check that every video sees the
course whole.
"""

import math
from collections import defaultdict
from collections.abc import Callable

from seascape.angles import bearing_deg, wrapped_deg
from seascape.assets import manifest
from seascape.config import Object, Scenario
from seascape.motion import rig_track
from seascape.outputs.labels import Labels
from seascape.sync.course import (
    BUOY,
    BUOY_PX,
    CONTAINER,
    Ring,
    View,
    box_half_deg,
    half_deg,
    one_rig,
    rings,
    views,
)

# Full size and one sample: a buoy stands `BUOY_PX` tall. Limitation: a thermal
# frame's gain and noise follow the frames rendered before it, which the preview
# skips.
PREVIEW = {"outputs": {"samples": {"eo": 1, "ir": 1}}}


def sightlines(scenario: Scenario) -> Callable[[Object, int], tuple[float, float]]:
    """An object's range from the one rig at a frame, and its bearing off the rig's
    axis, wrapped to [-180, 180).
    """
    rig_at = rig_track(scenario, one_rig(scenario)[0])
    seen_from = [rig_at(t) for t in scenario.outputs.times_s]

    def at(o: Object, f: int) -> tuple[float, float]:
        b, rig = math.radians(o.bearing_deg), seen_from[f]
        east = o.range_m * math.sin(b) - rig.east_m
        north = o.range_m * math.cos(b) - rig.north_m
        off = bearing_deg(east, north) - math.degrees(rig.heading_rad)
        return math.hypot(east, north), wrapped_deg(off)

    return at


def nearest_m(circles: list[Ring], range_m: float) -> float:
    """The range of the ring nearest `range_m`, by ratio."""
    return min((abs(math.log(range_m / r.range_m)), r.range_m) for r in circles)[1]


def held(
    scenario: Scenario, objects: list[Object]
) -> dict[tuple[Ring, View], list[bool]]:
    """For each ring and each video it serves, whether at each frame the video holds
    a buoy of that ring whole and clear of the container; a crest aside.
    """
    _, rig, _ = one_rig(scenario)
    videos = views(rig)
    circles = rings(videos)
    sight = sightlines(scenario)
    frames = range(len(scenario.outputs.times_s))
    buoys = [[sight(o, f) for f in frames] for o in objects if o.asset == BUOY]
    boxes = [[sight(o, f) for f in frames] for o in objects if o.asset == CONTAINER]

    def behind(f: int, out_m: float, off: float) -> bool:
        return any(
            box_m < out_m and abs(wrapped_deg(off - box)) < box_half_deg(box_m)
            for box_m, box in (lines[f] for lines in boxes)
        )

    found = {}
    for ring in circles:
        for view in (v for v in videos if v.name in ring.videos):
            middle = (view.left_deg + view.right_deg) / 2
            found[ring, view] = [
                any(
                    nearest_m(circles, out_m) == ring.range_m
                    and abs(wrapped_deg(off - middle))
                    <= (view.right_deg - view.left_deg) / 2 - half_deg(out_m)
                    and not behind(f, out_m, off)
                    for out_m, off in (lines[f] for lines in buoys)
                )
                for f in frames
            ]
    return found


def keyframes(scenario: Scenario) -> list[int]:
    """The frames `check` needs: the first and last, for the container, and for
    each ring and each video it serves the middle of the frames that hold one of its
    buoys.

    Raises:
        ValueError: A video never holds a buoy of a ring it serves.
    """
    frames = {0, len(scenario.outputs.times_s) - 1}
    for (ring, view), whole in held(scenario, scenario.objects).items():
        holding = [f for f, h in enumerate(whole) if h]
        if not holding:
            raise ValueError(
                f"{view.name} never holds a buoy {ring.range_m:.0f} m out whole"
            )
        frames.add(holding[len(holding) // 2])
    return sorted(frames)


def check(seen: Labels, scenario: Scenario) -> None:
    """Fail unless every video sees a buoy of each of its rings whole and at least
    `BUOY_PX` tall at least once, and the container whole in its first and last
    frames. A box cut by the container or a crest is too short.

    Raises:
        ValueError: The first video that fails, and what it missed.
    """
    assets = manifest()
    _sees_every_ring(seen, scenario, assets[BUOY].category_id)
    _keeps_the_container(seen, assets[CONTAINER].category_id)


def _sees_every_ring(seen: Labels, scenario: Scenario, buoy: int) -> None:
    name, rig, _ = one_rig(scenario)
    circles = rings(views(rig))
    images = {i.id: i.camera for i in seen.images}
    met: dict[str, set[float]] = defaultdict(set)
    for a in seen.annotations:
        # A box rounds a buoy to whole rows: one row short is still all of it.
        if a.category_id == buoy and not a.truncated and a.bbox[3] >= BUOY_PX - 1:
            met[images[a.image_id]].add(nearest_m(circles, a.range_m))
    for ring in circles:
        for video in sorted(ring.videos):
            if ring.range_m not in met[f"{name}_{video}"]:
                raise ValueError(
                    f"{name}_{video} never sees a buoy {ring.range_m:.0f} m out whole"
                )


def _keeps_the_container(seen: Labels, box: int) -> None:
    whole: dict[int, set[int]] = defaultdict(set)
    for a in seen.annotations:
        if not a.truncated:
            whole[a.image_id].add(a.category_id)
    frames: dict[str, list[int]] = defaultdict(list)
    for image in sorted(seen.images, key=lambda i: i.time_s):
        frames[image.camera].append(image.id)
    for video, ids in frames.items():
        for end, i in (("first", ids[0]), ("last", ids[-1])):
            if box not in whole[i]:
                raise ValueError(f"{video} misses the container in its {end} frame")
