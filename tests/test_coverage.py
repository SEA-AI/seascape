"""The check of a render's labels against what each video must see."""

import dataclasses

import pytest

from seascape.outputs.coverage import Requirement, check
from seascape.outputs.labels import Annotation, Image, Labels

BOX, BUOY = 1, 2
NEAR_M, FAR_M = 40.0, 100.0
type Seen = list[tuple[int, float, bool, int] | tuple[int, float, bool, int, int]]
BOXED: Seen = [(BOX, 300.0, False, 40)]


def seen(**videos: tuple[Seen, ...]) -> Labels:
    """Each video's frames, an object per (category, range, cut, tall_px) and its
    wide_px when it is not as wide as tall.
    """
    shots = [(v, frame) for v, frames in videos.items() for frame in frames]
    images = [
        Image.model_construct(id=i, camera=v, time_s=i / 3)
        for i, (v, _) in enumerate(shots)
    ]
    annotations = [
        Annotation.model_construct(
            image_id=i,
            category_id=category,
            range_m=range_m,
            truncated=cut,
            bbox=(0, 0, wide[0] if wide else tall_px, tall_px),
            name=f"{category}-{range_m}",
        )
        for i, (_, frame) in enumerate(shots)
        for category, range_m, cut, tall_px, *wide in frame
    ]
    return Labels.model_construct(images=images, annotations=annotations)


def ring(range_m: float, *videos: str) -> Requirement:
    return Requirement(
        category_id=BUOY,
        videos=frozenset(videos),
        frames="any",
        what=f"a buoy {range_m:.0f} m out",
        min_px=6,
        range_m=range_m,
    )


def box(*videos: str) -> Requirement:
    return Requirement(
        category_id=BOX,
        videos=frozenset(videos),
        frames="first+last",
        what="the container",
    )


NEEDS = [ring(NEAR_M, "T1", "RGB1"), box("T1", "RGB1")]


def passing() -> tuple[Seen, ...]:
    """The container at both ends, and a near buoy whole between."""
    return (BOXED, [(BUOY, NEAR_M, False, 6)], BOXED)


def test_a_container_drawn_wider_off_its_axis_between_the_ends_passes() -> None:
    # A pinhole draws it wider off the frame's axis.
    frames = (BOXED, [(BUOY, NEAR_M, False, 6), (BOX, 300.0, False, 40, 60)], BOXED)
    check(seen(T1=frames, RGB1=frames), NEEDS)


def test_a_buoy_a_pixel_narrower_than_its_widest_box_is_whole() -> None:
    # Box widths round to whole pixels.
    frames: tuple[Seen, ...] = (
        BOXED,
        [(BUOY, NEAR_M, False, 6, 5)],
        [(BUOY, NEAR_M, False, 4, 6)],
        BOXED,
    )
    check(seen(T1=frames, RGB1=frames), NEEDS)


def test_a_container_shorter_than_its_least_height_misses_the_ends() -> None:
    tall = dataclasses.replace(box("T1"), min_px=41)

    with pytest.raises(ValueError, match="T1 misses the container in its first"):
        check(seen(T1=passing()), [tall])


def test_an_object_counts_toward_the_requirement_nearest_its_range() -> None:
    needs = [ring(NEAR_M, "T1"), ring(FAR_M, "T1")]
    # 60 m is nearer 40 than 100 by ratio, so it serves only the near ring.
    buoys: Seen = [(BUOY, NEAR_M, False, 6), (BUOY, 60.0, False, 6)]

    with pytest.raises(ValueError, match="T1 never sees a buoy 100 m out"):
        check(seen(T1=(buoys,)), needs)


def test_each_range_met_passes() -> None:
    needs = [ring(NEAR_M, "T1"), ring(FAR_M, "T1")]
    # 66 m is nearer 100 than 40 by ratio, though not by difference.
    buoys: Seen = [(BUOY, NEAR_M, False, 6), (BUOY, 66.0, False, 6)]
    check(seen(T1=(buoys,)), needs)


def test_a_video_the_render_left_out_fails() -> None:
    with pytest.raises(ValueError, match="RGB1 has no frames"):
        check(seen(T1=passing()), NEEDS)


@pytest.mark.parametrize(
    ("frames", "error"),
    [
        (
            # A pixel short of the smallest box a detector takes.
            (BOXED, [(BUOY, NEAR_M, False, 5)], BOXED),
            "T1 never sees a buoy 40 m out whole",
        ),
        (
            # Full height in one frame, but for a sliver; full width in the next,
            # but sunk in a trough.
            (
                BOXED,
                [(BUOY, NEAR_M, False, 6, 1)],
                [(BUOY, NEAR_M, False, 4, 6)],
                BOXED,
            ),
            "T1 never sees a buoy 40 m out whole",
        ),
        (
            # Two pixels narrower than its widest box: more than rounding.
            (
                BOXED,
                [(BUOY, NEAR_M, False, 6, 4)],
                [(BUOY, NEAR_M, False, 4, 6)],
                BOXED,
            ),
            "T1 never sees a buoy 40 m out whole",
        ),
        (
            ([(BOX, 300.0, True, 40), (BUOY, NEAR_M, False, 6)], BOXED),
            "T1 misses the container in its first frame",
        ),
        (
            ([(BUOY, NEAR_M, False, 6)], []),
            "T1 misses the container in its first frame",
        ),
        (
            # Hidden but for a sliver in the last frame.
            ([*BOXED, (BUOY, NEAR_M, False, 6)], [(BOX, 300.0, False, 40, 4)]),
            "T1 misses the container in its last frame",
        ),
    ],
)
def test_a_video_that_misses_a_requirement_fails(
    frames: tuple[Seen, ...], error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        check(seen(T1=frames, RGB1=passing()), NEEDS)


def test_a_video_s_ends_are_its_first_and_last_in_time() -> None:
    labels = seen(T1=([*BOXED, (BUOY, NEAR_M, False, 6)], []), RGB1=passing())
    labels.images.reverse()

    with pytest.raises(ValueError, match="T1 misses the container in its last frame"):
        check(labels, NEEDS)
