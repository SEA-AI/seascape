"""labels.json from an index map and a calibration, without a render."""

import math

import cv2
import numpy as np
import pytest

from seascape import labels, waves
from seascape.calibration import CameraCalibration

RADIUS_M = waves.earth_radius_m(0.13)
FLAT = np.ones((48, 64, 3))


def grey(luminance: np.ndarray) -> np.ndarray:
    return np.repeat(luminance[..., None], 3, axis=2)


def shown(index: np.ndarray) -> np.ndarray:
    return grey(np.where(index > 0, 1.0, 0.5))


def camera(
    width: int = 64,
    height: int = 48,
    hfov_deg: float = 49.0,
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
    at_m: tuple[float, float, float] = (0.0, 0.0, 50.0),
) -> CameraCalibration:
    """Facing north, then pitched about its x and rolled about its z."""
    f = width / 2 / math.tan(math.radians(hfov_deg) / 2)
    level = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
    pitch, _ = cv2.Rodrigues(np.array([math.radians(pitch_deg), 0.0, 0.0]))
    roll, _ = cv2.Rodrigues(np.array([0.0, 0.0, math.radians(roll_deg)]))
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = level @ pitch @ roll, at_m
    return CameraCalibration(
        name="C",
        band="eo",
        image="C.png",
        rig="bow",
        width_px=width,
        height_px=height,
        K=((f, 0, (width - 1) / 2), (0, f, (height - 1) / 2), (0, 0, 1)),
        extrinsics={"world": tuple(map(tuple, pose))},
    )


def target(
    pass_index: int, category: str = "ship", category_id: int = 1
) -> labels.Target:
    return labels.Target(
        pass_index=pass_index,
        name=f"t{pass_index}",
        category_id=category_id,
        category=category,
        supercategory="vessel",
        centre_m=(0.0, 1000.0),
        waterline_m=np.array([[0.0, 990.0], [5.0, 1010.0]]),
        heading_deg=90.0,
        dims_m=(20.0, 5.0, 4.0),
    )


def test_a_box_covers_whole_pixels_from_its_top_left_corner() -> None:
    """COCO's pixel i spans [i, i + 1), so an inclusive box grows by one."""
    index = np.zeros((48, 64), dtype=int)
    index[10:13, 20:25] = 1
    truth = labels.Labels()

    truth.add(camera(), 0.0, index, shown(index), [target(1)], RADIUS_M)

    (found,) = truth.annotations
    assert found.bbox == (20, 10, 5, 3)
    assert found.area == 15
    assert not found.truncated


def test_a_box_on_the_frame_edge_is_truncated() -> None:
    index = np.zeros((48, 64), dtype=int)
    index[40:, 60:] = 1
    truth = labels.Labels()

    truth.add(camera(), 0.0, index, shown(index), [target(1)], RADIUS_M)

    assert truth.annotations[0].truncated


def test_only_a_target_in_frame_is_labelled_or_categorised() -> None:
    index = np.zeros((48, 64), dtype=int)
    index[10, 10] = 1
    truth = labels.Labels()

    truth.add(
        camera(), 0.0, index, shown(index), [target(1), target(2, "buoy", 6)], RADIUS_M
    )

    assert [a.name for a in truth.annotations] == ["t1"]
    assert [c.name for c in truth.categories] == ["ship"]


def test_targets_of_one_category_share_it() -> None:
    index = np.zeros((48, 64), dtype=int)
    index[10, 10], index[20, 20] = 1, 2
    truth = labels.Labels()

    truth.add(camera(), 0.0, index, shown(index), [target(1), target(2)], RADIUS_M)

    assert {a.category_id for a in truth.annotations} == {1}
    assert [(c.name, c.supercategory) for c in truth.categories] == [("ship", "vessel")]


def test_a_category_keeps_its_own_id_whatever_came_first() -> None:
    index = np.zeros((48, 64), dtype=int)
    index[10, 10], index[20, 20] = 1, 2
    truth = labels.Labels()

    truth.add(
        camera(), 0.0, index, shown(index), [target(1, "buoy", 6), target(2)], RADIUS_M
    )

    assert [(c.id, c.name) for c in truth.categories] == [(6, "buoy"), (1, "ship")]
    assert [a.category_id for a in truth.annotations] == [6, 1]


def test_an_image_carries_its_cameras_field_of_view() -> None:
    truth = labels.Labels()

    truth.add(
        camera(hfov_deg=42.0), 0.0, np.zeros((48, 64), dtype=int), FLAT, [], RADIUS_M
    )

    assert truth.images[0].hfov_deg == pytest.approx(42.0)


def test_an_annotation_carries_its_targets_heading_and_dimensions() -> None:
    index = np.zeros((48, 64), dtype=int)
    index[10, 10] = 1
    truth = labels.Labels()

    truth.add(camera(), 0.0, index, shown(index), [target(1)], RADIUS_M)

    (annotation,) = truth.annotations
    assert (annotation.heading_deg, annotation.dims_m) == (90.0, (20.0, 5.0, 4.0))


def test_a_frame_carries_its_time() -> None:
    truth = labels.Labels()

    truth.add(camera(), 2.5, np.zeros((48, 64), dtype=int), FLAT, [], RADIUS_M)

    assert truth.images[0].time_s == 2.5


def test_ranges_run_from_the_camera_to_the_centre_and_the_nearest_waterline() -> None:
    index = np.zeros((48, 64), dtype=int)
    index[10, 10] = 1
    truth = labels.Labels()

    truth.add(
        camera(at_m=(0.0, 0.0, 50.0)), 0.0, index, shown(index), [target(1)], RADIUS_M
    )

    found = truth.annotations[0]
    assert (found.range_m, found.waterline_range_m, found.bearing_deg) == (
        pytest.approx(1000.0),
        pytest.approx(990.0),
        pytest.approx(0.0),
    )


def square() -> np.ndarray:
    index = np.zeros((48, 64), dtype=int)
    index[10:20, 20:30] = 1
    return index


def boxed(
    index: np.ndarray, frame: np.ndarray, band: str = "eo", pass_index: int = 1
) -> bool:
    truth = labels.Labels()
    cam = camera().model_copy(update={"band": band})
    truth.add(cam, 0.0, index, frame, [target(pass_index)], RADIUS_M)
    return bool(truth.annotations)


def eight_bit(index: np.ndarray, values: dict[int, int], rest: int = 120) -> np.ndarray:
    frame = np.full(index.shape, rest, np.uint8)
    for pass_index, value in values.items():
        frame[index == pass_index] = value
    return frame


# Against grey 120, OpenCV puts grey 125 at ΔE*ab 1.98 and 127 at 2.95.
@pytest.mark.parametrize(("value", "seen"), [(125, False), (127, True)])
@pytest.mark.parametrize("band", ["eo", "ir"])
def test_a_target_is_seen_at_the_jnd(value: int, seen: bool, band: str) -> None:
    index = square()
    grey = eight_bit(index, {1: value})
    frame = grey if band == "ir" else cv2.cvtColor(grey, cv2.COLOR_GRAY2RGB)

    assert boxed(index, frame, band) is seen


# Against grey 120, OpenCV puts grey 141 at ΔE*ab 8.38 and 146 at 10.29.
@pytest.mark.parametrize(("value", "seen"), [(141, False), (146, True)])
def test_one_pixel_needs_the_contrast_of_four(value: int, seen: bool) -> None:
    index = np.zeros((48, 64), dtype=int)
    index[20, 30] = 1

    assert boxed(index, eight_bit(index, {1: value})) is seen


@pytest.mark.parametrize(("swapped", "seen"), [(False, False), (True, True)])
def test_each_row_is_judged_against_its_own_background(
    swapped: bool, seen: bool
) -> None:
    """Across the horizon a target the colour of its row's sky or sea is unseen, though
    it differs from the mean around it."""
    index = square()
    sky, sea = (60, 200) if swapped else (200, 60)
    frame = np.full(index.shape, 200, np.uint8)
    frame[15:] = 60
    frame[10:15][index[10:15] == 1] = sky
    frame[15:20][index[15:20] == 1] = sea

    assert boxed(index, frame) is seen


def test_another_object_in_the_flank_is_not_background() -> None:
    """Counted, the neighbour would halve the target's ΔE*ab to under the JND."""
    index = square()
    index[10:20, 30:38] = 2

    assert boxed(index, eight_bit(index, {1: 128, 2: 128}))


def test_a_target_as_bright_as_its_background_is_seen_by_its_colour() -> None:
    index = square()
    frame = cv2.cvtColor(eight_bit(index, {}), cv2.COLOR_GRAY2RGB)
    frame[index == 1] = (182, 91, 120)  # BT.709 luminance within 0.01 % of grey 120

    assert boxed(index, frame)


def test_a_target_with_no_background_to_measure_keeps_its_box() -> None:
    index = np.ones((48, 64), dtype=int)

    assert boxed(index, eight_bit(index, {}))


def test_a_float_frame_keeps_every_box() -> None:
    index = square()

    assert boxed(index, np.ones((*index.shape, 3), np.float32))


def test_a_frame_is_read_as_rgb() -> None:
    """As BGR the difference would be ΔE*ab 0.69, under the JND; as RGB it is 2.63."""
    index = square()
    frame = np.empty((*index.shape, 3), np.uint8)
    frame[:] = (150, 60, 30)
    frame[index == 1] = (150, 60, 35)

    assert boxed(index, frame)


@pytest.mark.parametrize(("value", "seen"), [(60, False), (200, True)])
def test_a_row_with_no_flank_takes_the_nearest_rows_background(
    value: int, seen: bool
) -> None:
    """Sky over sea, the target the colour of each row but its flankless sea row."""
    index = square()
    index[17, 12:20] = index[17, 30:38] = 2
    frame = np.full(index.shape, 200, np.uint8)
    frame[15:] = 60
    frame[17][index[17] == 1] = value

    assert boxed(index, frame) is seen


@pytest.mark.parametrize(("value", "seen"), [(125, False), (127, True)])
def test_a_target_on_the_left_edge_is_judged_on_its_flanks(
    value: int, seen: bool
) -> None:
    index = np.zeros((48, 64), dtype=int)
    index[10:20, 2:12] = 1
    index[10:20, 12:20] = 2

    assert boxed(index, eight_bit(index, {1: value})) is seen


def grazing_circle_px(cam: CameraCalibration, radius_m: float) -> np.ndarray:
    """The horizon another way: the circle where rays from C touch the paraboloid,
    (x - cx)^2 + (y - cy)^2 = 2R cz + cx^2 + cy^2, projected. COCO pixels, by u."""
    pose = np.array(cam.extrinsics["world"])
    rotation, centre = pose[:3, :3], pose[:3, 3]
    cx, cy, cz = centre
    rho = math.sqrt(2 * radius_m * cz + cx * cx + cy * cy)
    phi = np.radians(np.arange(-180.0, 180.0, 0.001))
    x, y = cx + rho * np.sin(phi), cy + rho * np.cos(phi)
    ahead = rotation.T @ (
        np.stack([x, y, -(x * x + y * y) / (2 * radius_m)]) - centre[:, None]
    )
    ahead = ahead[:, ahead[2] > 0]
    u, v, _ = np.array(cam.K) @ ahead / ahead[2]
    order = np.argsort(u)
    return np.stack([u[order], v[order]], axis=1) + 0.5


@pytest.mark.parametrize(
    ("pitch_deg", "roll_deg", "at_m"),
    [(0.0, 0.0, (0.0, 0.0, 50.0)), (-6.0, 2.0, (21.5, -68.2, 51.8))],
)
def test_the_horizon_lies_where_rays_graze_the_sea(
    pitch_deg: float, roll_deg: float, at_m: tuple[float, float, float]
) -> None:
    cam = camera(3840, 2160, 49.0, pitch_deg, roll_deg, at_m)
    reference = grazing_circle_px(cam, RADIUS_M)

    points = np.array(labels.horizon_px(cam, RADIUS_M))

    assert len(points) == labels.HORIZON_POINTS
    expected = np.interp(points[:, 0], reference[:, 0], reference[:, 1])
    assert points[:, 1] == pytest.approx(expected, abs=1e-3)


def test_no_horizon_in_a_frame_that_looks_at_the_sea_only() -> None:
    assert labels.horizon_px(camera(pitch_deg=-60.0), RADIUS_M) == []


def test_a_merge_renumbers_reaches_every_file_and_keeps_each_scenario(
    tmp_path,
) -> None:
    folders = []
    for seed, category in ((7, ("ship", 1)), (8, ("buoy", 6))):
        index = np.zeros((48, 64), dtype=int)
        index[10, 10] = 1
        truth = labels.Labels(info={"scenario": {"seed": seed}, "version": "x"})
        truth.add(camera(), 0.0, index, shown(index), [target(1, *category)], RADIUS_M)
        folders.append(tmp_path / str(seed))
        folders[-1].mkdir()
        truth.write(folders[-1])

    merged = labels.merge(tmp_path, folders)

    assert [(i.id, i.file_name) for i in merged.images] == [
        (1, "7/C.png"),
        (2, "8/C.png"),
    ]
    assert [(a.id, a.image_id, a.category_id) for a in merged.annotations] == [
        (1, 1, 1),
        (2, 2, 6),
    ]
    assert [c.id for c in merged.categories] == [1, 6]
    assert merged.info == {
        "version": "x",
        "scenarios": {"7": {"seed": 7}, "8": {"seed": 8}},
    }


def test_a_merge_refuses_two_names_for_one_category(tmp_path) -> None:
    folders = []
    for name in ("ship", "buoy"):
        index = np.zeros((48, 64), dtype=int)
        index[10, 10] = 1
        truth = labels.Labels(info={"scenario": {}})
        truth.add(camera(), 0.0, index, shown(index), [target(1, name, 1)], RADIUS_M)
        folders.append(tmp_path / name)
        folders[-1].mkdir()
        truth.write(folders[-1])

    with pytest.raises(ValueError, match="category 1 is ship"):
        labels.merge(tmp_path, folders)
