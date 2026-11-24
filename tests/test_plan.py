import math
from pathlib import Path
from typing import Any, cast

import bpy
import cv2
import numpy as np
import pytest
from mathutils import Vector
from PIL import Image

from seascape import angles, plan
from seascape.config import load
from seascape.outputs.plan import (
    GROW,
    MARGIN,
    OBJECT_PX,
    distance,
    grown,
    hud,
    hud_lines,
    label_bearing_deg,
    nice_m,
    overlap,
    ring_ranges_m,
    samples,
    top_view,
)
from tests.scenarios import SWEEP_SENTRY, target, variant


@pytest.mark.parametrize(
    ("ranges_m", "rings_m"),
    [
        # 104 is within 5 % of 100, 120 is not.
        ([200.0, 104.0, 100.0, 120.0], [102.0, 120.0, 200.0]),
        # 108 is within 5 % of 104 but not of 100, the run's smallest.
        ([100.0, 104.0, 108.0], [102.0, 108.0]),
    ],
)
def test_ranges_within_merge_of_the_smallest_share_a_ring(
    ranges_m: list[float], rings_m: list[float]
) -> None:
    assert ring_ranges_m(ranges_m) == pytest.approx(rings_m)


def test_the_top_view_frames_a_lone_hull_whole() -> None:
    """A 40.8 m yacht alone, longer than the floor's frame."""
    view = top_view(np.zeros((1, 2)), 0.3, (720, 720), 20.4)

    assert view.m_per_px * 720 >= 40.8


def test_the_top_view_puts_the_bow_up_and_starboard_right() -> None:
    heading_rad = math.radians(30.0)
    bow = np.array([math.sin(heading_rad), math.cos(heading_rad)])
    starboard = np.array([bow[1], -bow[0]])
    ahead, abeam = 100.0 * bow, 50.0 * starboard
    size_px = (200, 100)

    view = top_view(np.array([[0.0, 0.0], ahead, abeam]), heading_rad, size_px)

    T_camera_world = np.linalg.inv(view.matrix(500.0))
    own, seen, right = (
        T_camera_world @ np.array([*p, 0.0, 1.0]) for p in ([0.0, 0.0], ahead, abeam)
    )
    # A Blender camera's image runs along its +X right and +Y up, looking down -Z.
    assert seen[:2] - own[:2] == pytest.approx([0.0, 100.0], abs=1e-9)
    assert right[:2] - own[:2] == pytest.approx([50.0, 0.0], abs=1e-9)
    assert own[2] < 0.0
    # Centred on the points, the taller span filling the frame less its margins.
    assert (own[0] + right[0]) / 2 == pytest.approx(0.0, abs=1e-9)
    assert (own[1] + seen[1]) / 2 == pytest.approx(0.0, abs=1e-9)
    assert view.m_per_px == pytest.approx(100.0 * (1 + 2 * MARGIN) / size_px[1])


def test_the_top_view_holds_every_hull_whole_as_it_moves() -> None:
    """A 40.8 x 9 m yacht passing close astern of a 12 m ownship: its size, not its
    track, sets the frame.
    """
    bpy.ops.wm.read_factory_settings(use_empty=True)
    own, yacht = (bpy.data.objects.new(n, None) for n in ("own", "yacht"))
    yacht["length_m"], yacht["beam_m"] = 40.8, 9.0
    for frame, east_m in ((0, -10.0), (1, 10.0)):
        yacht.location = (east_m, -25.0, 0.0)
        yacht.keyframe_insert("location", frame=frame)
    for root in (own, yacht):
        bpy.context.scene.collection.objects.link(root)
    scene = plan._Scene(
        roots=[own, yacht],
        fans={},
        rings={},
        path=cast("Any", None),  # the top view never reads it
        heading_rad=0.3,
        length_m=12.0,
        reach_m=400.0,
    )
    lens = plan._camera(scene, "top")

    m_per_px = plan._aim_top(scene, lens, np.array([0, 1]))

    T_camera_world = lens.camera.matrix_world.inverted()
    half_m = np.array(lens.size_px) * m_per_px / 2
    for frame in (0, 1):
        bpy.context.scene.frame_set(frame)
        for root, reach_m in ((own, 6.0), (yacht, math.hypot(40.8, 9.0) / 2)):
            seen = np.array((T_camera_world @ root.matrix_world.translation).xy)
            assert (np.abs(seen) + reach_m <= half_m + 1e-6).all()


@pytest.mark.parametrize(
    ("heading_deg", "shown"),
    [(359.6, "HDG 000°"), (-0.4, "HDG 000°"), (7.0, "HDG 007°")],
)
def test_the_hud_rounds_the_heading_within_a_turn(
    heading_deg: float, shown: str
) -> None:
    _, nav, _ = hud_lines(0.0, math.radians(heading_deg), 0.0, None)

    assert nav.startswith(shown)


def test_the_chase_camera_sits_astern_and_above_along_its_look() -> None:
    heading_deg, length_m = 60.0, 20.0
    T = plan._chase(length_m, 100.0, 50.0, math.radians(heading_deg))

    bow = Vector((*angles.direction(heading_deg, 0.0)[:2], 0.0))
    eye = T.translation
    assert tuple((Vector((100.0, 50.0, 0.0)) - eye).xy.normalized()) == pytest.approx(
        tuple(bow.xy)
    )
    assert eye.z == pytest.approx(plan.CHASE_UP * length_m)
    # A Blender camera looks down its -Z, its +Y up the image.
    look = T.to_3x3() @ Vector((0.0, 0.0, -1.0))
    assert angles.bearing_deg(look.x, look.y) == pytest.approx(heading_deg)
    assert math.degrees(math.asin(look.z)) == pytest.approx(plan.CHASE_PITCH_DEG)
    # Upright and level: +X to starboard of the look; mathutils is single precision.
    starboard = T.to_3x3() @ Vector((1.0, 0.0, 0.0))
    assert tuple(starboard) == pytest.approx((bow.y, -bow.x, 0.0), abs=1e-6)


@pytest.mark.parametrize(
    ("output_fps", "count", "fps", "frames", "play_fps"),
    [
        (25.0, 10, 15.0, [0, 2, 4, 6, 8], 12.5),
        # Faster than the run is rendered: every frame.
        (10.0, 5, 30.0, [0, 1, 2, 3, 4], 10.0),
    ],
)
def test_a_rate_takes_the_nearest_whole_step_of_the_frames(
    output_fps: float, count: int, fps: float, frames: list[int], play_fps: float
) -> None:
    taken, rate = samples(output_fps, count, fps)

    assert taken.tolist() == frames
    assert rate == play_fps


@pytest.mark.parametrize(
    ("m_per_px", "footprint_m", "factor"),
    [
        (0.01, 1.0, 1.0),
        (0.5, 2.0, 0.5 * OBJECT_PX / 2.0),
        (10.0, 1.0, GROW),
    ],
    ids=["near", "between", "far"],
)
def test_an_object_grows_to_its_least_pixels_within_its_most(
    m_per_px: float, footprint_m: float, factor: float
) -> None:
    assert grown(m_per_px, footprint_m) == pytest.approx(factor)


def test_a_fan_reaches_where_a_pixel_spans_its_footprint() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scenario = load(SWEEP_SENTRY)

    fans = [f for _, rig in plan._fans(scenario).values() for f in rig]

    focal_px = [m.camera.focal_px for m in scenario.mounts]
    reach_m = [f.reach_m for f in fans]
    assert reach_m == pytest.approx([plan.FOOTPRINT_M * f for f in focal_px])
    sides_m = [[(far - f.near).length for far in f.far] for f in fans]
    assert sides_m == [pytest.approx([f.reach_m] * 2) for f in fans]
    # Edge to edge, a level pinhole's field; the ptu's tilt widens it a hair.
    widths_deg = [
        (
            angles.bearing_deg(*(f.far[1] - f.near).xy)
            - angles.bearing_deg(*(f.far[0] - f.near).xy)
        )
        % 360.0
        for f in fans
    ]
    hfovs_deg = [m.camera.hfov_deg for m in scenario.mounts]
    assert widths_deg == pytest.approx(hfovs_deg, rel=0.01)


@pytest.mark.parametrize(
    ("m", "shown"), [(104.0, "100 m"), (1500.0, "1.5 km"), (996.0, "1 km")]
)
def test_a_distance_reads_to_two_figures_in_metres_or_kilometres(
    m: float, shown: str
) -> None:
    assert distance(m) == shown


@pytest.mark.parametrize(
    ("most_m", "bar_m"), [(740.0, 500.0), (19.0, 10.0), (2.0, 2.0)]
)
def test_a_scale_bar_is_the_largest_1_2_or_5_that_fits(
    most_m: float, bar_m: float
) -> None:
    assert nice_m(most_m) == bar_m


@pytest.mark.parametrize(
    ("hulls_deg", "label_deg"), [([], 20.0), ([15.0], 40.0), ([350.0, 40.0], 10.0)]
)
def test_a_ring_s_label_sits_farthest_from_every_hull(
    hulls_deg: list[float], label_deg: float
) -> None:
    candidates_deg = np.array([0.0, 10.0, 20.0, 30.0, 40.0])

    assert label_bearing_deg(candidates_deg, hulls_deg) == label_deg


@pytest.mark.parametrize(
    ("b", "shared"),
    [((5, 5, 15, 15), True), ((5, 20, 15, 30), False), ((10, 0, 20, 10), False)],
    ids=["overlapping", "below", "touching"],
)
def test_boxes_overlap_only_where_they_share_area(
    b: tuple[float, float, float, float], shared: bool
) -> None:
    assert overlap((0, 0, 10, 10), b) is shared


def test_a_ring_label_over_an_earlier_one_is_left_out(tmp_path: Path) -> None:
    pictures = [tmp_path / "both.png", tmp_path / "first.png"]
    for path in pictures:
        Image.new("RGB", (720, 720)).save(path)
    lines = ("0.0 s", "HDG 000°", "")
    first = ((360.0, 360.0), 100.0)

    hud(pictures[0], lines, None, [first, ((364.0, 362.0), 1500.0)])
    hud(pictures[1], lines, None, [first])

    both, alone = (np.asarray(Image.open(p)) for p in pictures)
    assert both.any()
    assert np.array_equal(both, alone)


def _frames(path: Path) -> int:
    video, count = cv2.VideoCapture(str(path)), 0
    while video.read()[0]:
        count += 1
    return count


@pytest.mark.render
def test_each_video_holds_a_frame_per_sample(tmp_path: Path) -> None:
    scenario = variant(
        objects=[target("usv_magura", 100.0, 20.0, speed_mps=5.0)],
        outputs={"duration_s": 2.0, "fps": 10},
    )

    drawn = plan.draw(scenario, tmp_path, "usv", 2.0)

    assert all(path.exists() for path in drawn.stills.values())
    # Every fifth of 20 output frames: 0, 0.5, 1 and 1.5 s.
    assert [_frames(path) for path in drawn.videos.values()] == [4, 4]
