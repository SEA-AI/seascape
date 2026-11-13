"""The sync course's placement, the preview's settings and its check of the labels."""

import dataclasses
import itertools
import math
import tomllib
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from seascape.assets import Buoy, manifest
from seascape.blender.scene import posed_at_start
from seascape.config import Scenario, load, sources
from seascape.motion import head_track
from seascape.outputs.labels import Annotation, Image, Labels
from seascape.sync import preview, sheet, staging
from tests.scenarios import OPEN_SEA, SYNC_SENTRY, SYNC_WATCHKEEPER

PRODUCTS = [SYNC_SENTRY, SYNC_WATCHKEEPER]


def test_printed_entries_read_back_as_themselves() -> None:
    entries = staging.stage(load(SYNC_SENTRY)).entries

    read = tomllib.loads(staging.objects_toml(entries))["objects"]

    assert read == entries


def test_a_ring_loads_evenly_spaced_with_a_drift_drawn_per_buoy() -> None:
    ring: dict[str, object] = {"asset": staging.BUOY, "range_m": 40.0}
    ring |= {"bearing_deg": -10.0, "count": 3, "spacing_deg": 7.5}
    ring["drift"] = staging.BUOY_DRIFT

    buoys = load(OPEN_SEA, [{"objects": [ring]}]).objects

    assert [b.bearing_deg for b in buoys] == [-10.0, -2.5, 5.0]
    periods = [b.drift.period_s for b in buoys if b.drift is not None]
    assert len(set(periods)) == 3


def test_each_pair_s_coarser_video_sets_a_ring() -> None:
    (rig,) = load(SYNC_SENTRY).rigs.values()

    rings = staging.rings(staging.views(rig))

    # Each video pairs with every other; the wide thermal is the coarsest of all,
    # then the wide RGB, then the narrow thermal against the narrow RGB.
    assert [sorted(r.videos) for r in rings] == [
        ["RGBnFoV", "RGBwFoV", "TnFoV", "TwFoV"],
        ["RGBnFoV", "RGBwFoV", "TnFoV"],
        ["RGBnFoV", "TnFoV"],
    ]


def test_videos_either_side_of_the_stern_overlap() -> None:
    stern = staging.View(
        name="STERN",
        left_deg=132.5,
        right_deg=227.5,
        focal_px=1000.0,
        width_px=1920,
        camera_deg=95.0,
    )
    port = dataclasses.replace(stern, name="PORT", left_deg=-137.5, right_deg=-42.5)
    bow = dataclasses.replace(stern, left_deg=0.0, right_deg=95.0)

    assert staging.overlap_deg(stern, port) == (222.5, 227.5)
    assert staging.overlap_deg(port, bow) is None


@pytest.mark.parametrize("path", PRODUCTS)
def test_every_ring_crosses_every_camera_it_serves(path: Path) -> None:
    scenario = load(path)
    assert scenario.turn is not None
    (rig,) = scenario.rigs.values()
    buoys = [o for o in scenario.objects if o.asset == staging.BUOY]
    stops = [0.0, *(move.to_deg for move in scenario.turn.moves)]

    # Each camera of a stitch gets its own crossings.
    span_deg = {
        v: min(rig.rendered[c].hfov_deg for c in sources(video))
        for v, video in rig.videos.items()
    }
    for ring in staging.rings(staging.views(rig)):
        out_m = staging.stored_m(ring, rig)
        bearings = sorted(o.bearing_deg for o in buoys if o.range_m == out_m)
        narrowest = min(span_deg[v] for v in ring.videos)
        assert bearings[0] - min(stops) < narrowest
        assert max(stops) - bearings[-1] < narrowest
        assert max(b - a for a, b in itertools.pairwise(bearings)) < narrowest


def test_a_pan_tilt_unit_holds_its_inclination_whatever_its_mount_s_pitch() -> None:
    scenario = load(SYNC_SENTRY)
    (name,) = scenario.rigs
    pitched = load(SYNC_SENTRY, [{"rigs": {name: {"pitch_deg": -3.0}}}])

    assert staging.stage(pitched).inclination_deg == pytest.approx(
        staging.stage(scenario).inclination_deg
    )


def test_a_fixed_mount_that_loses_a_ring_names_the_pitch_to_set() -> None:
    scenario = load(SYNC_WATCHKEEPER)
    (name,) = scenario.rigs

    with pytest.raises(ValueError, match="set the rig's pitch_deg to"):
        staging.stage(load(SYNC_WATCHKEEPER, [{"rigs": {name: {"pitch_deg": 20.0}}}]))


def test_every_buoy_stands_buoy_px_tall_or_more_in_its_coarsest_video() -> None:
    scenario = load(SYNC_SENTRY)
    (rig,) = scenario.rigs.values()
    buoy = manifest()[staging.BUOY]
    assert isinstance(buoy, Buoy)
    focal_px = {
        v: min(rig.rendered[c].focal_px for c in sources(video))
        for v, video in rig.videos.items()
    }
    off_m = math.hypot(rig.offset_x_m, rig.offset_y_m)
    # A stored range is rounded to half a last decimal.
    rounding_m = 0.5 * 10.0**-staging.DECIMALS

    for ring in staging.rings(staging.views(rig)):
        coarsest_px = min(focal_px[v] for v in ring.videos)
        for o in scenario.objects:
            if o.asset != staging.BUOY or o.range_m != staging.stored_m(ring, rig):
                continue
            # The cameras turn about the rig, not the origin.
            b = math.radians(o.bearing_deg)
            east = o.range_m * math.sin(b) - rig.offset_x_m
            north = o.range_m * math.cos(b) - rig.offset_y_m
            above_px = (buoy.height_m - buoy.draught_m) * coarsest_px
            above_px /= math.hypot(east, north)
            # From the ring's range, out to rounding, in to two offsets nearer.
            low = staging.BUOY_PX * ring.range_m / (ring.range_m + rounding_m)
            near_m = ring.range_m - 2 * off_m - rounding_m
            assert low <= above_px <= staging.BUOY_PX * ring.range_m / near_m


def test_a_ring_draws_each_buoy_s_colour_from_those_it_comes_in() -> None:
    entries = staging.stage(load(SYNC_SENTRY)).entries
    ring = next(e for e in entries if e["asset"] == staging.BUOY)
    sold_in = staging.asset(staging.BUOY, Buoy).sold_in

    assert ring["colour"] == {"choice": sorted(sold_in)}


def test_a_drawn_spacing_is_refused() -> None:
    ring = {"asset": staging.BUOY, "range_m": 40.0, "bearing_deg": 0.0, "count": 2}
    ring["spacing_deg"] = {"uniform": [5.0, 10.0]}

    with pytest.raises(ValueError, match="spacing_deg is a fixed step"):
        load(OPEN_SEA, [{"objects": [ring]}])


def test_a_spacing_without_a_bearing_is_refused() -> None:
    ring = {"asset": staging.BUOY, "range_m": 40.0, "count": 2, "spacing_deg": 5.0}

    with pytest.raises(ValueError, match="spacing_deg is a fixed step"):
        load(OPEN_SEA, [{"objects": [ring]}])


def test_a_course_needs_a_turn() -> None:
    with pytest.raises(ValueError, match="needs a turn and one rig"):
        staging.stage(load(OPEN_SEA))


def test_a_course_needs_a_still_ownship() -> None:
    under_way = load(PRODUCTS[0], [{"ownship": {"speed_mps": 2.0}}])
    with pytest.raises(ValueError, match="a course needs it still"):
        staging.stage(under_way)


def test_advice_without_a_course_that_fits_says_why() -> None:
    said = preview.advice(load(OPEN_SEA), "missed")

    assert said.startswith("missed; and no course fits its rig: ")


def test_the_sheet_unrolls_each_frame_where_its_camera_looks() -> None:
    scenario = load(SYNC_SENTRY)
    ((name, rig),) = scenario.rigs.items()
    posed = posed_at_start(scenario, name)
    heading_deg = math.degrees(head_track(scenario, name)(0.0).heading_rad)
    n = sheet.EDGE_POINTS
    edges = sheet.frame_edges(rig, posed.cameras)

    for view in staging.views(rig):
        rays = edges[view.name]
        bearings = [np.degrees(np.arctan2(r[:, 0], r[:, 1])) for r in rays]
        elevations = [np.degrees(np.arcsin(r[:, 2])) for r in rays]
        # Clockwise from the top left: the right edge's middle, then the left's.
        left = min(b[3 * n + n // 2] for b in bearings)
        right = max(b[n + n // 2] for b in bearings)
        assert left == pytest.approx(heading_deg + view.left_deg, abs=0.1)
        assert right == pytest.approx(heading_deg + view.right_deg, abs=0.1)
        assert all(e[0] > e[2 * n - 1] for e in elevations)


def test_a_rig_on_a_rolled_deck_leans_and_turns_with_it() -> None:
    """A rig yawed 45 on a deck rolled 20 starboard down points 43.2 from the bow
    and leans to starboard.
    """
    scenario = load(
        SYNC_SENTRY,
        [{"ownship": {"roll_deg": 20.0}, "rigs": {"mast": {"yaw_deg": 45.0}}}],
    )
    rig = scenario.rigs["mast"]
    start = head_track(scenario, "mast")(0.0)

    yaw, roll = math.radians(45.0), math.radians(20.0)
    expected = math.degrees(math.atan2(math.sin(yaw) * math.cos(roll), math.cos(yaw)))
    assert math.degrees(start.heading_rad) == pytest.approx(expected)
    assert (start.east_m, start.north_m) == pytest.approx(
        (rig.height_m * math.sin(roll), rig.offset_y_m)
    )


BOX = manifest()[staging.CONTAINER].category_id
BUOY = manifest()[staging.BUOY].category_id
type Seen = list[tuple[int, float, bool, int] | tuple[int, float, bool, int, int]]
BOXED: Seen = [(BOX, 300.0, False, 40)]


def seen(**videos: tuple[Seen, ...]) -> Labels:
    """Each video's frames, an object per (category, range, cut, tall_px) and its
    wide_px when it is not as wide as tall.
    """
    shots = [(v, frame) for v, frames in videos.items() for frame in frames]
    images = [
        Image.model_construct(id=i, camera=f"mast_{v}", time_s=i / 3)
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


def passing(near_m: float) -> tuple[Seen, ...]:
    """The container at both ends, and a buoy of the nearest ring whole between."""
    return (BOXED, [(BUOY, near_m, False, 6)], BOXED)


@pytest.fixture(scope="module")
def watchkeeper() -> Scenario:
    return load(SYNC_WATCHKEEPER)


def near_m(scenario: Scenario) -> float:
    """The range of the scenario's nearest ring."""
    return staging.rings(staging.views(*scenario.rigs.values()))[0].range_m


def test_a_buoy_of_each_ring_and_the_container_at_both_ends_pass(
    watchkeeper: Scenario,
) -> None:
    frames = passing(near_m(watchkeeper))
    preview.check(seen(T1=frames, RGB1=frames), watchkeeper)


def test_a_container_drawn_wider_off_its_axis_between_the_ends_passes(
    watchkeeper: Scenario,
) -> None:
    near = near_m(watchkeeper)
    # A pinhole draws it wider off the frame's axis.
    frames = (BOXED, [(BUOY, near, False, 6), (BOX, 300.0, False, 40, 60)], BOXED)
    preview.check(seen(T1=frames, RGB1=frames), watchkeeper)


@pytest.mark.parametrize("drop_farthest", [False, True])
def test_every_ring_of_a_rig_with_several_must_be_seen(drop_farthest: bool) -> None:
    sentry = load(SYNC_SENTRY)
    circles = staging.rings(staging.views(*sentry.rigs.values()))
    videos = sorted({v for r in circles for v in r.videos})
    if drop_farthest:
        circles = circles[:-1]

    def turned(video: str) -> tuple[Seen, ...]:
        """The container at both ends, and a buoy of each ring the video serves."""
        held: Seen = [(BUOY, r.range_m, False, 6) for r in circles if video in r.videos]
        return (BOXED, held, BOXED)

    frames = {v: turned(v) for v in videos}

    if drop_farthest:
        with pytest.raises(ValueError, match="never sees a buoy"):
            preview.check(seen(**frames), sentry)
    else:
        preview.check(seen(**frames), sentry)


def test_a_video_the_preview_left_out_fails(watchkeeper: Scenario) -> None:
    only_t1 = seen(T1=passing(near_m(watchkeeper)))
    with pytest.raises(ValueError, match="mast_RGB1 has no frames"):
        preview.check(only_t1, watchkeeper)


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
            # A pixel short of the smallest box a detector takes.
            lambda near: (BOXED, [(BUOY, near, False, 5)], BOXED),
            "mast_T1 never sees a buoy {near:.0f} m out whole",
        ),
        (
            # Full height in one frame, but for a sliver; full width in the next,
            # but sunk in a trough.
            lambda near: (
                BOXED,
                [(BUOY, near, False, 6, 1)],
                [(BUOY, near, False, 4, 6)],
                BOXED,
            ),
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
        (
            # Hidden but for a sliver in the last frame.
            lambda near: (
                [*BOXED, (BUOY, near, False, 6)],
                [(BOX, 300.0, False, 40, 4)],
            ),
            "mast_T1 misses the container in its last frame",
        ),
    ],
)
def test_a_video_that_misses_a_ring_or_the_container_fails(
    frames: Callable[[float], tuple[Seen, ...]], error: str, watchkeeper: Scenario
) -> None:
    near = near_m(watchkeeper)
    with pytest.raises(ValueError, match=error.format(near=near)):
        preview.check(seen(T1=frames(near), RGB1=passing(near)), watchkeeper)
