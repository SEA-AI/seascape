"""The sync course's placement, the preview's settings and its check of the labels."""

import itertools
import math
import tomllib
from collections.abc import Callable
from pathlib import Path

import pytest

from seascape.assets import Buoy, manifest
from seascape.config import Drift, Object, Scenario, load, sources
from seascape.outputs.labels import Annotation, Image, Labels
from seascape.sync import course, preview
from tests.scenarios import OPEN_SEA, SYNC_SENTRY, SYNC_WATCHKEEPER

PRODUCTS = [SYNC_SENTRY, SYNC_WATCHKEEPER]


@pytest.mark.parametrize("path", PRODUCTS)
def test_a_product_stores_the_course_placed_for_it(path: Path) -> None:
    course.verify(load(path))


def test_printed_objects_read_back_as_themselves() -> None:
    objects = [
        Object(
            asset=course.BUOY, range_m=40.0, bearing_deg=60.0, colour="orange", wear=0.5
        ),
        Object(
            asset=course.CONTAINER,
            range_m=100.0,
            bearing_deg=0.0,
            drift=Drift(sway_m=1.0, surge_m=0.5, period_s=60.0),
        ),
    ]

    read = tomllib.loads(course.objects_toml(objects))["objects"]

    assert [Object.model_validate(o) for o in read] == objects


def test_each_pair_s_coarser_video_sets_a_ring() -> None:
    (rig,) = load(SYNC_SENTRY).rigs.values()

    rings = course.rings(course.views(rig))

    # Each video pairs with every other; the wide thermal is the coarsest of all,
    # then the wide RGB, then the narrow thermal against the narrow RGB.
    assert [sorted(r.videos) for r in rings] == [
        ["RGBnFoV", "RGBwFoV", "TnFoV", "TwFoV"],
        ["RGBnFoV", "RGBwFoV", "TnFoV"],
        ["RGBnFoV", "TnFoV"],
    ]


def test_videos_either_side_of_the_stern_overlap() -> None:
    stern = course.View(
        name="STERN",
        left_deg=132.5,
        right_deg=227.5,
        down_deg=-20.0,
        up_deg=20.0,
        focal_px=1000.0,
        width_px=1920,
        camera_deg=95.0,
    )
    port = stern._replace(name="PORT", left_deg=-137.5, right_deg=-42.5)

    assert course.overlap_deg(stern, port) == (222.5, 227.5)
    assert (
        course.overlap_deg(port, stern._replace(left_deg=0.0, right_deg=95.0)) is None
    )


@pytest.mark.parametrize("path", PRODUCTS)
def test_every_ring_crosses_every_camera_it_serves(path: Path) -> None:
    scenario = load(path)
    assert scenario.turn is not None
    (rig,) = scenario.rigs.values()
    buoys = [o for o in scenario.objects if o.asset == course.BUOY]
    stops = [0.0, *(move.to_deg for move in scenario.turn.moves)]

    # Each camera of a stitch gets its own crossings.
    span_deg = {
        v: min(rig.rendered[c].hfov_deg for c in sources(video))
        for v, video in rig.videos.items()
    }
    for ring in course.rings(course.views(rig)):
        out_m = pytest.approx(ring.range_m, rel=0.1)
        bearings = sorted(o.bearing_deg for o in buoys if o.range_m == out_m)
        narrowest = min(span_deg[v] for v in ring.videos)
        assert bearings[0] - min(stops) < narrowest
        assert max(stops) - bearings[-1] < narrowest
        assert max(b - a for a, b in itertools.pairwise(bearings)) < narrowest


def test_a_pan_tilt_unit_inclines_to_keep_the_horizon_and_the_nearest_ring() -> None:
    (rig,) = load(SYNC_SENTRY).rigs.values()
    assert rig.ptu is not None
    near_m = course.rings(course.views(rig))[0].range_m
    waterline_deg = -math.degrees(math.atan(rig.height_m / near_m))

    for camera in rig.rendered.values():
        half_deg = math.degrees(math.atan(camera.height_px / 2 / camera.focal_px))
        assert rig.ptu.inclination_deg + camera.pitch_deg + half_deg > 0.0
        assert rig.ptu.inclination_deg + camera.pitch_deg - half_deg < waterline_deg


def test_a_pan_tilt_unit_holds_its_inclination_whatever_its_mount_s_pitch() -> None:
    scenario = load(SYNC_SENTRY)
    (name,) = scenario.rigs
    pitched = load(SYNC_SENTRY, [{"rigs": {name: {"pitch_deg": -3.0}}}])

    # The head holds the inclination itself, whatever its mount's pitch.
    assert course.course(pitched).inclination_deg == pytest.approx(
        course.course(scenario).inclination_deg
    )


def test_a_fixed_mount_that_loses_a_ring_names_the_pitch_to_set() -> None:
    scenario = load(SYNC_WATCHKEEPER)
    (name,) = scenario.rigs

    with pytest.raises(ValueError, match="set the rig's pitch_deg to"):
        course.course(load(SYNC_WATCHKEEPER, [{"rigs": {name: {"pitch_deg": 20.0}}}]))


def test_every_buoy_stands_buoy_px_tall_in_the_coarsest_video_it_serves() -> None:
    scenario = load(SYNC_SENTRY)
    (rig,) = scenario.rigs.values()
    buoy = manifest()[course.BUOY]
    assert isinstance(buoy, Buoy)
    focal_px = {
        v: min(rig.rendered[c].focal_px for c in sources(video))
        for v, video in rig.videos.items()
    }
    rings = course.rings(course.views(rig))

    for o in scenario.objects:
        if o.asset != course.BUOY:
            continue
        # The cameras turn about the rig, not the origin.
        b = math.radians(o.bearing_deg)
        east = o.range_m * math.sin(b) - rig.offset_x_m
        north = o.range_m * math.cos(b) - rig.offset_y_m
        buoy_m = math.hypot(east, north)
        ring = min(rings, key=lambda r: abs(math.log(buoy_m / r.range_m)))
        coarsest_px = min(focal_px[v] for v in ring.videos)
        above_px = (buoy.height_m - buoy.draught_m) * coarsest_px / buoy_m
        assert above_px == pytest.approx(course.BUOY_PX, rel=0.02)


def test_a_course_needs_a_turn() -> None:
    with pytest.raises(ValueError, match="needs a turn and one rig"):
        course.course(load(OPEN_SEA))


@pytest.mark.parametrize("path", PRODUCTS)
def test_a_preview_has_a_frame_where_each_video_holds_each_of_its_rings(
    path: Path,
) -> None:
    quick = load(path, [preview.PREVIEW])

    frames = preview.keyframes(quick)

    for whole in preview.held(quick, quick.objects).values():
        assert any(whole[f] for f in frames)


def test_a_preview_without_buoys_has_no_keyframes() -> None:
    with pytest.raises(ValueError, match="never holds a buoy"):
        preview.keyframes(load(SYNC_WATCHKEEPER, [preview.PREVIEW, {"objects": []}]))


BOX = manifest()[course.CONTAINER].category_id
BUOY = manifest()[course.BUOY].category_id
type Seen = list[tuple[int, float, bool, int]]
BOXED: Seen = [(BOX, 300.0, False, 40)]


def seen(video: str, *frames: Seen) -> Labels:
    """One video, a frame per entry, an object per (category, range, cut, tall_px)."""
    images = [
        Image.model_construct(id=i, camera=f"mast_{video}", time_s=i / 3)
        for i in range(len(frames))
    ]
    annotations = [
        Annotation.model_construct(
            image_id=i,
            category_id=category,
            range_m=range_m,
            truncated=cut,
            bbox=(0, 0, tall_px, tall_px),
        )
        for i, frame in enumerate(frames)
        for category, range_m, cut, tall_px in frame
    ]
    return Labels.model_construct(images=images, annotations=annotations)


def both(t1: Labels, rgb1: Labels) -> Labels:
    """Two videos' labels as one recording's."""
    shift = len(t1.images)
    images = [
        *t1.images,
        *(i.model_copy(update={"id": i.id + shift}) for i in rgb1.images),
    ]
    annotations = [
        *t1.annotations,
        *(
            a.model_copy(update={"image_id": a.image_id + shift})
            for a in rgb1.annotations
        ),
    ]
    return Labels.model_construct(images=images, annotations=annotations)


def passing(near_m: float) -> tuple[Seen, ...]:
    """The container at both ends, and a buoy of the nearest ring whole between."""
    return (BOXED, [(BUOY, near_m, False, 6)], BOXED)


@pytest.fixture(scope="module")
def watchkeeper() -> Scenario:
    return load(SYNC_WATCHKEEPER)


def near_m(scenario: Scenario) -> float:
    """The range of the scenario's nearest ring."""
    return course.rings(course.views(*scenario.rigs.values()))[0].range_m


def test_a_buoy_of_each_ring_and_the_container_at_both_ends_pass(
    watchkeeper: Scenario,
) -> None:
    frames = passing(near_m(watchkeeper))
    preview.check(both(seen("T1", *frames), seen("RGB1", *frames)), watchkeeper)


@pytest.mark.parametrize(
    ("frames", "error"),
    [
        (
            lambda near: (BOXED, [(BUOY, near, True, 6)], BOXED),
            "mast_T1 never sees a buoy {near:.0f} m out whole",
        ),
        (
            # Half hidden behind the container.
            lambda near: (BOXED, [(BUOY, near, False, 3)], BOXED),
            "mast_T1 never sees a buoy {near:.0f} m out whole",
        ),
        (
            lambda near: ([(BOX, 300.0, True, 40), (BUOY, near, False, 6)], BOXED),
            "mast_T1 misses the container in its first frame",
        ),
        (
            lambda near: ([*BOXED, (BUOY, near, False, 6)], []),
            "mast_T1 misses the container in its last frame",
        ),
    ],
)
def test_a_video_that_misses_a_ring_or_the_container_fails(
    frames: Callable[[float], tuple[Seen, ...]], error: str, watchkeeper: Scenario
) -> None:
    near = near_m(watchkeeper)
    with pytest.raises(ValueError, match=error.format(near=near)):
        preview.check(
            both(seen("T1", *frames(near)), seen("RGB1", *passing(near))), watchkeeper
        )
