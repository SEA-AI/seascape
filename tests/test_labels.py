"""labels.json from an index map and a calibration, without a render."""

import math

import cv2
import numpy as np
import pytest

from seascape import labels, waves
from seascape.calibration import CameraCalibration

RADIUS_M = waves.earth_radius_m(0.13)


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
        width_px=width,
        height_px=height,
        K=((f, 0, (width - 1) / 2), (0, f, (height - 1) / 2), (0, 0, 1)),
        extrinsics={"world": tuple(map(tuple, pose))},
    )


def target(pass_index: int, category: str = "ship") -> labels.Target:
    return labels.Target(
        pass_index=pass_index,
        name=f"t{pass_index}",
        category=category,
        supercategory="vessel",
        centre_m=(0.0, 1000.0),
        waterline_m=np.array([[0.0, 990.0], [5.0, 1010.0]]),
    )


def test_a_box_covers_whole_pixels_from_its_top_left_corner() -> None:
    """COCO's pixel i spans [i, i + 1), so an inclusive box grows by one."""
    index = np.zeros((48, 64), dtype=int)
    index[10:13, 20:25] = 1
    truth = labels.Labels()

    truth.add(camera(), 0.0, index, [target(1)], RADIUS_M)

    (found,) = truth.annotations
    assert found.bbox == (20, 10, 5, 3)
    assert found.area == 15
    assert not found.truncated


def test_a_box_on_the_frame_edge_is_truncated() -> None:
    index = np.zeros((48, 64), dtype=int)
    index[40:, 60:] = 1
    truth = labels.Labels()

    truth.add(camera(), 0.0, index, [target(1)], RADIUS_M)

    assert truth.annotations[0].truncated


def test_only_a_target_in_frame_is_labelled_or_categorised() -> None:
    index = np.zeros((48, 64), dtype=int)
    index[10, 10] = 1
    truth = labels.Labels()

    truth.add(camera(), 0.0, index, [target(1), target(2, "buoy")], RADIUS_M)

    assert [a.name for a in truth.annotations] == ["t1"]
    assert [c.name for c in truth.categories] == ["ship"]


def test_targets_of_one_category_share_it() -> None:
    index = np.zeros((48, 64), dtype=int)
    index[10, 10], index[20, 20] = 1, 2
    truth = labels.Labels()

    truth.add(camera(), 0.0, index, [target(1), target(2)], RADIUS_M)

    assert {a.category_id for a in truth.annotations} == {1}
    assert [(c.name, c.supercategory) for c in truth.categories] == [("ship", "vessel")]


def test_a_frame_carries_its_time() -> None:
    truth = labels.Labels()

    truth.add(camera(), 2.5, np.zeros((48, 64), dtype=int), [], RADIUS_M)

    assert truth.images[0].time_s == 2.5


def test_ranges_run_from_the_camera_to_the_centre_and_the_nearest_waterline() -> None:
    index = np.zeros((48, 64), dtype=int)
    index[10, 10] = 1
    truth = labels.Labels()

    truth.add(camera(at_m=(0.0, 0.0, 50.0)), 0.0, index, [target(1)], RADIUS_M)

    found = truth.annotations[0]
    assert (found.range_m, found.waterline_range_m, found.bearing_deg) == (
        pytest.approx(1000.0),
        pytest.approx(990.0),
        pytest.approx(0.0),
    )


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
