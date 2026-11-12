"""The sync preview: the frames it renders, and the check that every video sees the
course whole.
"""

import math
from collections import defaultdict
from collections.abc import Callable

from seascape.angles import bearing_deg, wrapped_deg
from seascape.assets import Debris, manifest
from seascape.config import Object, Scenario, recorded_name
from seascape.motion import drifting, rig_track
from seascape.outputs.labels import Labels
from seascape.sync.course import (
    BUOY,
    BUOY_PX,
    CONTAINER,
    Ring,
    View,
    asset,
    buoy_half_deg,
    half_deg,
    one_rig,
    rings,
    stored_m,
    views,
)

# Full size and one sample: a buoy stands `BUOY_PX` tall. Limitation: a thermal
# frame's gain and noise follow the frames rendered before it, which the preview
# skips.
PREVIEW = {"outputs": {"samples": {"eo": 1, "ir": 1}}}


def sightlines(scenario: Scenario) -> Callable[[Object], list[tuple[float, float]]]:
    """An object's range from the one rig at each frame, and its bearing off the
    rig's axis, wrapped to [-180, 180).
    """
    outputs = scenario.outputs
    rig_at = rig_track(scenario, one_rig(scenario)[0])
    seen_from = [rig_at(t) for t in outputs.times_s]

    def at(o: Object) -> list[tuple[float, float]]:
        path = drifting(
            o.range_m, o.bearing_deg, o.course_deg, o.speed_mps, o.drift, outputs
        )
        found = []
        for t, rig in zip(outputs.times_s, seen_from, strict=True):
            pose = path(t)
            east, north = pose.east_m - rig.east_m, pose.north_m - rig.north_m
            off = bearing_deg(east, north) - math.degrees(rig.heading_rad)
            found.append((math.hypot(east, north), wrapped_deg(off)))
        return found

    return at


def nearest_m(circles: list[Ring], range_m: float) -> float:
    """The range of the ring nearest `range_m`, by ratio."""
    return min((abs(math.log(range_m / r.range_m)), r.range_m) for r in circles)[1]


def held(scenario: Scenario) -> dict[tuple[Ring, View], list[bool]]:
    """For each ring and each video it serves, whether at each frame the video holds
    a buoy of that ring whole and clear of the container; a crest aside.
    """
    _, rig, _ = one_rig(scenario)
    videos = views(rig)
    circles = rings(videos)
    sight = sightlines(scenario)
    frames = range(len(scenario.outputs.times_s))
    box_m = asset(CONTAINER, Debris).length_m
    boxes = [sight(o) for o in scenario.objects if o.asset == CONTAINER]
    # A buoy belongs to the ring whose entry stores its range.
    by_ring: dict[float, list[list[tuple[float, float]]]] = defaultdict(list)
    for o in scenario.objects:
        if o.asset == BUOY:
            by_ring[o.range_m].append(sight(o))

    def holds(f: int, ring: Ring, view: View) -> bool:
        """Whether `view` holds a buoy of `ring` whole and clear at frame `f`."""
        middle = (view.left_deg + view.right_deg) / 2
        for lines in by_ring[stored_m(ring, rig)]:
            out_m, off = lines[f]
            room = (view.right_deg - view.left_deg) / 2 - buoy_half_deg(out_m)
            if abs(wrapped_deg(off - middle)) > room:
                continue
            if not any(_behind(box[f], box_m, out_m, off) for box in boxes):
                return True
        return False

    found = {}
    for ring in circles:
        for view in videos:
            if view.name in ring.videos:
                found[ring, view] = [holds(f, ring, view) for f in frames]
    return found


def _behind(box: tuple[float, float], box_m: float, out_m: float, off: float) -> bool:
    """Whether a container `box_m` long at sightline `box` covers any of a buoy
    `out_m` out at bearing `off`.
    """
    near_m, at = box
    reach = half_deg(box_m, near_m) + buoy_half_deg(out_m)
    return near_m < out_m and abs(wrapped_deg(off - at)) < reach


def keyframes(scenario: Scenario) -> list[int]:
    """The frames `check` needs: the first and last, for the container, and as few
    more as hold a buoy of each ring in each video it serves.

    Raises:
        ValueError: A video never holds a buoy of a ring it serves.
    """
    holding = {}
    for (ring, view), whole in held(scenario).items():
        if not any(whole):
            raise ValueError(
                f"{view.name} never holds a buoy {ring.range_m:.0f} m out whole"
            )
        holding[ring, view] = {f for f, h in enumerate(whole) if h}
    frames = {0, len(scenario.outputs.times_s) - 1}
    left = {k for k, fs in holding.items() if not fs & frames}
    # Greedy: each frame added holds the most pairs still missing.
    while left:
        counts: dict[int, int] = defaultdict(int)
        for k in left:
            for f in holding[k]:
                counts[f] += 1
        best = max(counts, key=lambda f: (counts[f], -f))
        frames.add(best)
        left = {k for k in left if best not in holding[k]}
    return sorted(frames)


def check(seen: Labels, scenario: Scenario) -> None:
    """Fail unless every video sees a buoy of each of its rings whole and at least
    `BUOY_PX` tall at least once, and the container whole in its first and last
    frames. A box cut by the container or a crest is too short or too narrow.

    Raises:
        ValueError: A video with no frames, or the first that fails, and what it
            missed.
    """
    name, rig, _ = one_rig(scenario)
    videos = views(rig)
    rendered = {i.camera for i in seen.images}
    # A video with no frames is one the preview did not render: a band left out.
    for video in sorted(recorded_name(name, v.name) for v in videos):
        if video not in rendered:
            raise ValueError(f"{video} has no frames: add its band to outputs.bands")
    assets = manifest()
    _sees_every_ring(seen, name, rings(videos), assets[BUOY].category_id)
    _keeps_the_container(seen, assets[CONTAINER].category_id)


def _whole(wide_px: int, widest_px: int) -> bool:
    """Whether a box `wide_px` wide holds all of what is `widest_px` wide at its
    widest; a box rounds to whole pixels, so one short is still all of it.
    """
    return wide_px >= widest_px - 1


def _sees_every_ring(seen: Labels, name: str, circles: list[Ring], buoy: int) -> None:
    images = {i.id: i.camera for i in seen.images}
    buoys = [a for a in seen.annotations if a.category_id == buoy and not a.truncated]
    # A buoy drifts mostly along the line of sight and the rig barely moves, so a
    # video's widest box of it is the whole buoy: one hidden but for a sliver is
    # narrower.
    widest: dict[tuple[str, str], int] = defaultdict(int)
    for a in buoys:
        key = (images[a.image_id], a.name)
        widest[key] = max(widest[key], a.bbox[2])
    met: dict[str, set[float]] = defaultdict(set)
    for a in buoys:
        video = images[a.image_id]
        if _whole(a.bbox[3], int(BUOY_PX)) and _whole(a.bbox[2], widest[video, a.name]):
            met[video].add(nearest_m(circles, a.range_m))
    for ring in circles:
        for video in sorted(ring.videos):
            recorded = recorded_name(name, video)
            if ring.range_m not in met[recorded]:
                raise ValueError(
                    f"{recorded} never sees a buoy {ring.range_m:.0f} m out whole"
                )


def _keeps_the_container(seen: Labels, box: int) -> None:
    wide: dict[int, int] = defaultdict(int)
    for a in seen.annotations:
        if a.category_id == box and not a.truncated:
            wide[a.image_id] = max(wide[a.image_id], a.bbox[2])
    frames: dict[str, list[int]] = defaultdict(list)
    for image in sorted(seen.images, key=lambda i: i.time_s):
        frames[image.camera].append(image.id)
    for video, ids in frames.items():
        ends = (("first", ids[0]), ("last", ids[-1]))
        # Both ends look along one bearing, so the container is as wide in each; a
        # frame between, off its axis, draws it wider. One hidden but for a sliver
        # is narrower.
        widest = max(wide[i] for _, i in ends)
        for end, i in ends:
            if not wide[i] or not _whole(wide[i], widest):
                raise ValueError(f"{video} misses the container in its {end} frame")
