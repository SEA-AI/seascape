"""The sync preview: its settings, and the check that every video sees the course
whole.
"""

import math
from collections import defaultdict

from seascape.assets import manifest
from seascape.config import Scenario, recorded_name
from seascape.outputs.labels import Labels
from seascape.sync import layout
from seascape.sync.layout import BUOY, BUOY_PX, CONTAINER, Ring, one_rig, rings, views

# Full size, so a buoy stands `BUOY_PX` tall; 1 fps, which holds every buoy in each
# video it serves for several frames at the turn's 6 deg/s; one sample.
PREVIEW = {"outputs": {"fps": 1, "samples": {"eo": 1, "ir": 1}}}


def placed(scenario: Scenario) -> bool:
    """Whether the scenario holds a course to preview: any buoy at all."""
    return any(o.asset == BUOY for o in scenario.objects)


def advice(scenario: Scenario, missed: str) -> str:
    """`missed`, after the course for the scenario's rig to paste where one fits."""
    try:
        printed = layout.suggested(scenario)
    except ValueError as why:
        return f"{missed}; and no course fits its rig: {why}"
    return f"{printed}\n\n{missed}; the course for its rig is above"


def nearest_m(circles: list[Ring], range_m: float) -> float:
    """The range of the ring nearest `range_m`, by ratio."""
    return min((abs(math.log(range_m / r.range_m)), r.range_m) for r in circles)[1]


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
    if not circles:
        raise ValueError("no two videos overlap, so the course has no ring to check")
    images = {i.id: i.camera for i in seen.images}
    buoys = [a for a in seen.annotations if a.category_id == buoy and not a.truncated]
    # A buoy drifts mostly along the line of sight and the rig barely moves, so a
    # video's widest box of it is the whole buoy: one hidden but for a sliver is
    # narrower. Limitation: a buoy hidden in every frame a video holds it sets its
    # own widest; comparing with its drawn width would remove that.
    widest: dict[tuple[str, str], int] = defaultdict(int)
    for a in buoys:
        key = (images[a.image_id], a.name)
        widest[key] = max(widest[key], a.bbox[2])
    met: dict[str, set[float]] = defaultdict(set)
    for a in buoys:
        video = images[a.image_id]
        if a.bbox[3] >= BUOY_PX and _whole(a.bbox[2], widest[video, a.name]):
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
