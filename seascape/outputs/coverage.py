"""Whether a render's videos see what was placed for them, read off its labels."""

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from seascape.outputs.labels import Annotation, Labels


@dataclass(frozen=True, slots=True, kw_only=True)
class Requirement:
    """An object of a category that each of `videos` must hold whole, not truncated
    and at least `min_px` tall: in any one frame, or in its first and in its last.
    """

    category_id: int
    videos: frozenset[str]
    frames: Literal["any", "first+last"]
    # What a failure names, such as "a buoy 40 m out".
    what: str
    min_px: float = 0.0
    # An object counts toward the requirement of its category whose range is nearest
    # its own, by ratio; None takes one at any range.
    range_m: float | None = None


def check(seen: Labels, requirements: Sequence[Requirement]) -> None:
    """Fail unless every video meets every requirement that names it. A box cut by
    another object or a crest is too short or too narrow.

    Raises:
        ValueError: A video with no frames, or the first that fails, and what it
            missed.
    """
    rendered = {i.camera for i in seen.images}
    for video in sorted({v for r in requirements for v in r.videos}):
        if video not in rendered:
            raise ValueError(f"{video} has no frames: add its band to outputs.bands")
    ranges: dict[int, set[float]] = defaultdict(set)
    for r in requirements:
        if r.range_m is not None:
            ranges[r.category_id].add(r.range_m)
    for requirement in requirements:
        whole = [
            a
            for a in seen.annotations
            if a.category_id == requirement.category_id
            and not a.truncated
            and (
                requirement.range_m is None
                or _nearest(a.range_m, ranges[a.category_id]) == requirement.range_m
            )
        ]
        if requirement.frames == "any":
            _in_any_frame(seen, whole, requirement)
        else:
            _at_both_ends(seen, whole, requirement)


def _nearest(range_m: float, ranges_m: set[float]) -> float:
    """Of `ranges_m`, the nearest `range_m` by ratio."""
    return min(ranges_m, key=lambda r: abs(math.log(range_m / r)))


def _whole(wide_px: int, widest_px: int) -> bool:
    """Whether a box `wide_px` wide holds all of what is `widest_px` wide at its
    widest; a box rounds to whole pixels, so one short is still all of it.
    """
    return wide_px >= widest_px - 1


def _in_any_frame(
    seen: Labels,
    whole: list[Annotation],
    requirement: Requirement,
) -> None:
    images = {i.id: i.camera for i in seen.images}
    # A moored object drifts mostly along the line of sight and the rig barely moves,
    # so a video's widest box of it is the whole object, however short a trough
    # leaves it: one hidden but for a sliver is narrower. Limitation: an object hidden
    # in every frame a video holds it sets its own widest; comparing with its drawn
    # width would remove that.
    widest: dict[tuple[str, str], int] = defaultdict(int)
    for a in whole:
        key = (images[a.image_id], a.name)
        widest[key] = max(widest[key], a.bbox[2])
    met = {
        images[a.image_id]
        for a in whole
        if a.bbox[3] >= requirement.min_px
        and _whole(a.bbox[2], widest[images[a.image_id], a.name])
    }
    for video in sorted(requirement.videos):
        if video not in met:
            raise ValueError(f"{video} never sees {requirement.what} whole")


def _at_both_ends(
    seen: Labels, whole: list[Annotation], requirement: Requirement
) -> None:
    wide: dict[int, int] = defaultdict(int)
    for a in whole:
        if a.bbox[3] >= requirement.min_px:
            wide[a.image_id] = max(wide[a.image_id], a.bbox[2])
    frames: dict[str, list[int]] = defaultdict(list)
    for image in sorted(seen.images, key=lambda i: i.time_s):
        frames[image.camera].append(image.id)
    for video in sorted(requirement.videos):
        ids = frames[video]
        ends = (("first", ids[0]), ("last", ids[-1]))
        # Both ends look along one bearing, so the object is as wide in each; a frame
        # between, off its axis, draws it wider.
        widest = max(wide[i] for _, i in ends)
        for end, i in ends:
            if not wide[i] or not _whole(wide[i], widest):
                raise ValueError(
                    f"{video} misses {requirement.what} in its {end} frame"
                )
