"""Where the buoys and container go for a rig and its turn."""

import dataclasses
import itertools
import math
import tomllib
from pathlib import Path

import numpy as np
import pytest

from seascape import angles, buoyage
from seascape.assets import Buoy, manifest
from seascape.config import load, recorded_name, sources
from seascape.config.loader import resolve
from seascape.motion import head_path
from seascape.outputs.calibration import CameraCalibration
from tests.scenarios import OPEN_SEA, SWEEP_SENTRY, SWEEP_WATCHKEEPER

PRODUCTS = [SWEEP_SENTRY, SWEEP_WATCHKEEPER]


def test_the_printed_course_reads_back_as_itself() -> None:
    scenario = load(SWEEP_SENTRY)

    assert tomllib.loads(buoyage.suggested(scenario)) == buoyage.course(scenario)


@pytest.mark.parametrize("path", PRODUCTS)
def test_each_product_stores_its_course(path: Path) -> None:
    assert buoyage.holds(resolve(path), load(path))


@pytest.mark.parametrize(
    ("overrides", "turn_deg"),
    [([], 1.0), (["rigs.mast.ptu.elevation_deg = 0.0"], 0.0)],
    ids=["a buoy moved", "the ptu pitched"],
)
def test_a_course_edited_by_hand_is_not_held(
    overrides: list[str], turn_deg: float
) -> None:
    scenario = load(SWEEP_SENTRY)
    edited = resolve(SWEEP_SENTRY, overrides)
    edited["objects"][1]["bearing_deg"] += turn_deg

    assert not buoyage.holds(edited, scenario)


def test_a_sweep_requires_each_ring_in_its_videos_and_the_box_in_every_one() -> None:
    scenario = load(SWEEP_SENTRY)
    (name,) = scenario.rigs
    (rig,) = scenario.rigs.values()
    rings = buoyage.rings(buoyage.views(rig))

    needs = buoyage.requirements(scenario, {"raw"})

    assets = manifest()
    *buoys, box = needs
    assert [n.range_m for n in buoys] == [r.range_m for r in rings]
    assert [n.videos for n in buoys] == [
        {recorded_name(name, v) for v in r.videos} for r in rings
    ]
    assert {n.category_id for n in buoys} == {assets[buoyage.BUOY].category_id}
    assert {n.min_px for n in buoys} == {buoyage.BUOY_PX}
    assert box.category_id == assets[buoyage.CONTAINER].category_id
    assert box.frames == "first+last"
    assert box.videos == {recorded_name(name, v) for v in rig.videos} | {"raw"}


def test_a_ring_loads_evenly_spaced_with_a_drift_drawn_per_buoy() -> None:
    ring: dict[str, object] = {"asset": buoyage.BUOY, "range_m": 40.0}
    ring |= {"bearing_deg": -10.0, "count": 3, "spacing_deg": 7.5}
    ring["fishtail"] = buoyage.BUOY_FISHTAIL

    buoys = load(OPEN_SEA, [{"objects": [ring]}]).objects

    assert [b.bearing_deg for b in buoys] == [-10.0, -2.5, 5.0]
    periods = [b.fishtail.period_s for b in buoys if b.fishtail is not None]
    assert len(set(periods)) == 3


def test_each_pair_s_coarser_video_sets_a_ring() -> None:
    (rig,) = load(SWEEP_SENTRY).rigs.values()

    rings = buoyage.rings(buoyage.views(rig))

    # Each video pairs with every other; the wide thermal is the coarsest of all,
    # then the wide RGB, then the narrow thermal against the narrow RGB.
    assert [sorted(r.videos) for r in rings] == [
        ["RGBnFoV", "RGBwFoV", "TnFoV", "TwFoV"],
        ["RGBnFoV", "RGBwFoV", "TnFoV"],
        ["RGBnFoV", "TnFoV"],
    ]


def test_videos_either_side_of_the_stern_overlap() -> None:
    stern = buoyage.View(
        name="STERN",
        left_deg=132.5,
        right_deg=227.5,
        focal_px=1000.0,
        width_px=1920,
        camera_deg=95.0,
    )
    port = dataclasses.replace(stern, name="PORT", left_deg=-137.5, right_deg=-42.5)
    bow = dataclasses.replace(stern, left_deg=0.0, right_deg=95.0)

    assert buoyage.overlap_deg(stern, port) == (222.5, 227.5)
    assert buoyage.overlap_deg(port, bow) is None


@pytest.mark.parametrize("path", PRODUCTS)
def test_every_ring_crosses_every_camera_it_serves(path: Path) -> None:
    scenario = load(path)
    assert scenario.turn is not None
    (rig,) = scenario.rigs.values()
    buoys = [o for o in scenario.objects if o.asset == buoyage.BUOY]
    stops = [0.0, *(move.to_deg for move in scenario.turn.moves)]

    # Each camera of a stitch gets its own crossings.
    span_deg = {
        v: min(rig.rendered[c].hfov_deg for c in sources(video))
        for v, video in rig.videos.items()
    }
    for ring in buoyage.rings(buoyage.views(rig)):
        out_m = buoyage.stored_m(ring, rig)
        bearings = sorted(o.bearing_deg for o in buoys if o.range_m == out_m)
        narrowest = min(span_deg[v] for v in ring.videos)
        assert bearings[0] - min(stops) < narrowest
        assert max(stops) - bearings[-1] < narrowest
        assert max(b - a for a, b in itertools.pairwise(bearings)) < narrowest


def test_a_pan_tilt_unit_holds_its_elevation_whatever_its_mount_s_pitch() -> None:
    scenario = load(SWEEP_SENTRY)
    (name,) = scenario.rigs
    pitched = load(SWEEP_SENTRY, [{"rigs": {name: {"pitch_deg": -3.0}}}])

    assert buoyage.stage(pitched).elevation_deg == pytest.approx(
        buoyage.stage(scenario).elevation_deg
    )


def test_a_fixed_mount_that_loses_a_ring_names_the_pitch_to_set() -> None:
    scenario = load(SWEEP_WATCHKEEPER)
    (name,) = scenario.rigs

    with pytest.raises(ValueError, match="set the rig's pitch_deg to"):
        buoyage.stage(load(SWEEP_WATCHKEEPER, [{"rigs": {name: {"pitch_deg": 20.0}}}]))


def test_every_buoy_stands_buoy_px_tall_or_more_in_its_coarsest_video() -> None:
    scenario = load(SWEEP_SENTRY)
    (rig,) = scenario.rigs.values()
    buoy = manifest()[buoyage.BUOY]
    assert isinstance(buoy, Buoy)
    focal_px = {
        v: min(rig.rendered[c].focal_px for c in sources(video))
        for v, video in rig.videos.items()
    }
    off_m = math.hypot(rig.offset_x_m, rig.offset_y_m)
    # A stored range is rounded to half a last decimal.
    rounding_m = 0.5 * 10.0**-buoyage.DECIMALS

    for ring in buoyage.rings(buoyage.views(rig)):
        coarsest_px = min(focal_px[v] for v in ring.videos)
        for o in scenario.objects:
            if o.asset != buoyage.BUOY or o.range_m != buoyage.stored_m(ring, rig):
                continue
            # The cameras turn about the rig, not the origin.
            b = math.radians(o.bearing_deg)
            east = o.range_m * math.sin(b) - rig.offset_x_m
            north = o.range_m * math.cos(b) - rig.offset_y_m
            above_px = (buoy.height_m - buoy.draught_m) * coarsest_px
            above_px /= math.hypot(east, north)
            # From the ring's range, out to rounding, in to two offsets nearer.
            low = buoyage.BUOY_PX * ring.range_m / (ring.range_m + rounding_m)
            near_m = ring.range_m - 2 * off_m - rounding_m
            assert low <= above_px <= buoyage.BUOY_PX * ring.range_m / near_m


def test_a_ring_draws_each_buoy_s_colour_from_those_it_comes_in() -> None:
    entries = buoyage.stage(load(SWEEP_SENTRY)).entries
    ring = next(e for e in entries if e["asset"] == buoyage.BUOY)

    buoys = load(OPEN_SEA, [{"objects": [ring]}]).objects

    assert {b.colour for b in buoys} <= set(buoyage.asset(buoyage.BUOY, Buoy).sold_in)


@pytest.mark.parametrize(
    "ring",
    [
        {"bearing_deg": 0.0, "spacing_deg": {"uniform": [5.0, 10.0]}},
        {"spacing_deg": 5.0},
    ],
    ids=["drawn", "without a bearing"],
)
def test_a_spacing_drawn_or_without_a_bearing_is_refused(ring: dict) -> None:
    ring |= {"asset": buoyage.BUOY, "range_m": 40.0, "count": 2}

    with pytest.raises(ValueError, match="spacing_deg is a fixed step"):
        load(OPEN_SEA, [{"objects": [ring]}])


def test_a_course_needs_a_turn() -> None:
    with pytest.raises(ValueError, match="needs a turn and one rig"):
        buoyage.stage(load(OPEN_SEA))


def test_a_course_needs_a_still_ownship() -> None:
    under_way = load(PRODUCTS[0], [{"ownship": {"speed_mps": 2.0}}])
    with pytest.raises(ValueError, match="a course needs it still"):
        buoyage.stage(under_way)


def test_a_rig_on_a_rolled_deck_leans_and_turns_with_it() -> None:
    """A rig yawed 45 on a deck rolled 20 starboard down points 43.2 from the bow
    and leans to starboard.
    """
    scenario = load(
        SWEEP_SENTRY,
        [{"ownship": {"roll_deg": 20.0}, "rigs": {"mast": {"yaw_deg": 45.0}}}],
    )
    rig = scenario.rigs["mast"]
    start = head_path(scenario, "mast")(0.0)

    yaw, roll = math.radians(45.0), math.radians(20.0)
    expected = math.degrees(math.atan2(math.sin(yaw) * math.cos(roll), math.cos(yaw)))
    assert math.degrees(start.heading_rad) == pytest.approx(expected)
    assert (start.east_m, start.north_m) == pytest.approx(
        (rig.height_m * math.sin(roll), rig.offset_y_m)
    )


def test_a_lighter_hull_lifts_its_rig_along_its_tilted_mast() -> None:
    """The hull rolls about its floating waterline, so its rise leans with it."""
    roll_deg = 20.0
    ownship = {"asset": "cargo_ship", "draught_m": 5.24, "roll_deg": roll_deg}
    scenario = load(SWEEP_SENTRY, [{"ownship": ownship}])
    start = head_path(scenario, "mast")(0.0)

    height_m = scenario.head_height_m("mast")
    assert height_m > scenario.rigs["mast"].height_m
    assert start.east_m == pytest.approx(height_m * math.sin(math.radians(roll_deg)))


def test_a_pitched_camera_s_span_is_read_at_the_head_s_level() -> None:
    """Pitched 10 up in its head, a camera's edges cross the head's level, where the
    buoys stand, atan(x cos 10) off its axis, not atan(x).
    """
    pitch = math.radians(10.0)
    R_head_camera = angles.turned(0.0, 10.0) @ angles.TO_CV.T
    T_head_camera = np.eye(4)
    T_head_camera[:3, :3] = R_head_camera
    width, f = 1001, 500.0
    camera = CameraCalibration(
        name="c",
        band="eo",
        image="c.png",
        rig="r",
        width_px=width,
        height_px=width,
        K=((f, 0.0, (width - 1) / 2), (0.0, f, (width - 1) / 2), (0.0, 0.0, 1.0)),
        extrinsics={"head": tuple(map(tuple, T_head_camera))},
    )

    x = (width / 2) / f
    edge = math.degrees(math.atan(x * math.cos(pitch)))
    assert buoyage._span(camera) == pytest.approx((-edge, edge))
