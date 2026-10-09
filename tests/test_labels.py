"""labels.json from an index map and a calibration, without a render."""

import math

import cv2
import numpy as np
import pytest

from seascape import labels, waves
from seascape.calibration import CameraCalibration

RADIUS_M = waves.earth_radius_m(0.13)
FLAT = np.ones((48, 64))


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

    truth.add(camera(), 0.0, index, FLAT, [target(1)], RADIUS_M)

    (found,) = truth.annotations
    assert found.bbox == (20, 10, 5, 3)
    assert found.area == 15
    assert not found.truncated


def test_a_box_on_the_frame_edge_is_truncated() -> None:
    index = np.zeros((48, 64), dtype=int)
    index[40:, 60:] = 1
    truth = labels.Labels()

    truth.add(camera(), 0.0, index, FLAT, [target(1)], RADIUS_M)

    assert truth.annotations[0].truncated


def test_only_a_target_in_frame_is_labelled_or_categorised() -> None:
    index = np.zeros((48, 64), dtype=int)
    index[10, 10] = 1
    truth = labels.Labels()

    truth.add(camera(), 0.0, index, FLAT, [target(1), target(2, "buoy", 6)], RADIUS_M)

    assert [a.name for a in truth.annotations] == ["t1"]
    assert [c.name for c in truth.categories] == ["ship"]


def test_targets_of_one_category_share_it() -> None:
    index = np.zeros((48, 64), dtype=int)
    index[10, 10], index[20, 20] = 1, 2
    truth = labels.Labels()

    truth.add(camera(), 0.0, index, FLAT, [target(1), target(2)], RADIUS_M)

    assert {a.category_id for a in truth.annotations} == {1}
    assert [(c.name, c.supercategory) for c in truth.categories] == [("ship", "vessel")]


def test_a_category_keeps_its_own_id_whatever_came_first() -> None:
    index = np.zeros((48, 64), dtype=int)
    index[10, 10], index[20, 20] = 1, 2
    truth = labels.Labels()

    truth.add(camera(), 0.0, index, FLAT, [target(1, "buoy", 6), target(2)], RADIUS_M)

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

    truth.add(camera(), 0.0, index, FLAT, [target(1)], RADIUS_M)

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

    truth.add(camera(at_m=(0.0, 0.0, 50.0)), 0.0, index, FLAT, [target(1)], RADIUS_M)

    found = truth.annotations[0]
    assert (found.range_m, found.waterline_range_m, found.bearing_deg) == (
        pytest.approx(1000.0),
        pytest.approx(990.0),
        pytest.approx(0.0),
    )


def test_a_target_like_its_background_has_no_contrast() -> None:
    index = np.zeros((48, 64), dtype=int)
    index[10:20, 20:30] = 1
    truth = labels.Labels()

    truth.add(camera(), 0.0, index, FLAT, [target(1)], RADIUS_M)

    assert truth.annotations[0].contrast == 0.0


@pytest.mark.parametrize(("target_l", "contrast"), [(0.3, 0.25), (1.0, 1.5)])
def test_a_uniform_targets_contrast_is_webers_unsigned(
    target_l: float, contrast: float
) -> None:
    index = np.zeros((48, 64), dtype=int)
    index[10:20, 20:30] = 1
    frame = np.where(index == 1, target_l, 0.4)
    truth = labels.Labels()

    truth.add(camera(), 0.0, index, frame, [target(1)], RADIUS_M)

    assert truth.annotations[0].contrast == pytest.approx(contrast)


def test_a_dark_and_a_bright_half_do_not_cancel() -> None:
    index = np.zeros((48, 64), dtype=int)
    index[10:20, 20:30] = 1
    frame = np.full((48, 64), 0.4)
    frame[10:15, 20:30], frame[15:20, 20:30] = 0.2, 0.6
    truth = labels.Labels()

    truth.add(camera(), 0.0, index, frame, [target(1)], RADIUS_M)

    assert truth.annotations[0].contrast == pytest.approx(0.5)


def test_contrasts_background_is_no_other_target() -> None:
    index = np.zeros((48, 64), dtype=int)
    index[10:20, 20:30], index[10:20, 30:40] = 1, 2
    frame = np.where(index == 1, 0.8, np.where(index == 2, 9.0, 0.4))
    truth = labels.Labels()

    truth.add(camera(), 0.0, index, frame, [target(1)], RADIUS_M)

    assert truth.annotations[0].contrast == pytest.approx(1.0)


def test_contrasts_background_skips_the_pixels_the_filter_blurs_into() -> None:
    index = np.zeros((48, 64), dtype=int)
    index[10:20, 20:30] = 1
    width = 2 * labels.RING_GAP_PX + 1
    blurred = cv2.dilate(index.astype(np.uint8), np.ones((width, width), np.uint8))
    frame = np.where(blurred == 1, 0.8, 0.4)
    truth = labels.Labels()

    truth.add(camera(), 0.0, index, frame, [target(1)], RADIUS_M)

    assert truth.annotations[0].contrast == pytest.approx(1.0)


def test_contrasts_background_skips_another_targets_blur() -> None:
    index = np.zeros((48, 64), dtype=int)
    index[10:20, 20:30], index[10:20, 34:44] = 1, 2
    width = 2 * labels.RING_GAP_PX + 1
    blurred = cv2.dilate(
        (index == 2).astype(np.uint8), np.ones((width, width), np.uint8)
    )
    frame = np.where(index == 1, 0.8, np.where(blurred == 1, 9.0, 0.4))
    truth = labels.Labels()

    truth.add(camera(), 0.0, index, frame, [target(1)], RADIUS_M)

    assert truth.annotations[0].contrast == pytest.approx(1.0)


def test_a_target_with_no_background_around_it_has_no_contrast() -> None:
    truth = labels.Labels()

    truth.add(camera(), 0.0, np.ones((48, 64), dtype=int), FLAT, [target(1)], RADIUS_M)

    assert truth.annotations[0].contrast is None


def test_an_8bit_frame_is_decoded_from_srgb_to_luminance() -> None:
    """sRGB 188 is linear 0.5, and grey is its own luminance."""
    index = np.zeros((48, 64), dtype=int)
    index[10:20, 20:30] = 1
    frame = np.full((48, 64, 3), 255, np.uint8)
    frame[index == 1] = 188
    truth = labels.Labels()

    truth.add(camera(), 0.0, index, frame, [target(1)], RADIUS_M)

    assert truth.annotations[0].contrast == pytest.approx(0.5, abs=0.003)


def test_8bit_primaries_weigh_as_bt709() -> None:
    """A BGR frame read as RGB swaps red's weight for blue's."""
    primaries = np.eye(3, dtype=np.uint8)[None] * 255

    assert labels.luminance(primaries)[0] == pytest.approx(labels.BT709)


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
        truth.add(camera(), 0.0, index, FLAT, [target(1, *category)], RADIUS_M)
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
        truth.add(camera(), 0.0, index, FLAT, [target(1, name, 1)], RADIUS_M)
        folders.append(tmp_path / name)
        folders[-1].mkdir()
        truth.write(folders[-1])

    with pytest.raises(ValueError, match="category 1 is ship"):
        labels.merge(tmp_path, folders)
