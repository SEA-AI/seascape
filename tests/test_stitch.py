"""Two pinholes stitched as the recorder's horizontal stitching does."""

import math

import numpy as np
import pytest

from seascape.outputs import stitch
from seascape.outputs.calibration import TO_CV, CameraCalibration

# Core-Backend's cameras take -fx and -fy: OpenCV's frame half a turn about the axis.
HALF_TURN = np.diag([-1.0, -1.0, 1.0])


def camera(
    name: str, k: tuple[float, float, float, float], turn: np.ndarray
) -> CameraCalibration:
    fx, fy, cx, cy = k
    pose = np.eye(4)
    pose[:3, :3] = turn
    return CameraCalibration(
        name=name,
        band="ir",
        image=f"{name}.png",
        rig="bow",
        width_px=640,
        height_px=512,
        K=((fx, 0.0, cx), (0.0, fy, cy), (0.0, 0.0, 1.0)),
        extrinsics={"rig": tuple(map(tuple, pose)), "world": tuple(map(tuple, pose))},
    )


def core(rpy_deg: tuple[float, float, float]) -> np.ndarray:
    """A camera's turn from Core-Backend's roll, pitch and yaw, Ry Rx Rz."""
    roll, pitch, yaw = map(math.radians, rpy_deg)
    c, s = math.cos, math.sin
    rx = np.array([[1, 0, 0], [0, c(pitch), -s(pitch)], [0, s(pitch), c(pitch)]])
    ry = np.array([[c(yaw), 0, s(yaw)], [0, 1, 0], [-s(yaw), 0, c(yaw)]])
    rz = np.array([[c(roll), -s(roll), 0], [s(roll), c(roll), 0], [0, 0, 1]])
    return TO_CV.T @ HALF_TURN @ ry @ rx @ rz @ HALF_TURN


def toward(yaw_deg: float) -> np.ndarray:
    """A level camera's turn into the rig, `yaw_deg` to starboard."""
    a = math.radians(-yaw_deg)
    return (
        np.array(
            [[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]]
        )
        @ TO_CV.T
    )


def test_the_fused_camera_is_the_one_the_recorder_stitches_onto() -> None:
    # A Watchkeeper's calibrated Bosons, and the T1 its recorder made of them.
    left = camera(
        "t2",
        (1492.8727451179086, 1492.8825192178263, 319.5, 255.5),
        core((0.12533663815589705, -0.39883822700015736, 11.285691028542576)),
    )
    right = camera(
        "t1",
        (1484.9256526681734, 1484.7724256583147, 319.5, 255.5),
        core((0.0068806206480282, -0.43073665892759266, -11.520948984920224)),
    )

    fused = stitch.fused(left, right, "T1", "T1.png")

    assert (fused.width_px, fused.height_px) == (1284, 498)
    assert fused.K[0] == pytest.approx((1488.899198893041, 0.0, 642.2806662491481))
    assert fused.K[1] == pytest.approx((0.0, 1488.8274724380703, 247.79129345491148))


def test_a_stitch_puts_each_ray_where_the_fused_camera_sees_it() -> None:
    f = 320 / math.tan(math.radians(12.15))
    k = (f, f, 319.5, 255.5)
    cameras = (camera("l", k, toward(-11.4)), camera("r", k, toward(11.4)))

    def bearings(c: CameraCalibration) -> np.ndarray:
        """Each pixel's bearing in the rig, degrees to starboard."""
        v, u = np.mgrid[: c.height_px, : c.width_px]
        rays = np.stack([u, v, np.ones_like(u)], axis=-1) @ np.linalg.inv(c.K).T
        x, y, _ = np.moveaxis(rays @ np.array(c.extrinsics["rig"])[:3, :3].T, -1, 0)
        return np.degrees(np.arctan2(x, y))

    def counts(c: CameraCalibration) -> np.ndarray:
        """How far each pixel looks off the rig's axis, as a 16-bit frame in
        millidegrees: alike either side, so the frames' levels already meet.
        """
        return np.rint(1000 * np.abs(bearings(c))).astype(np.uint16)

    fused = stitch.fused(*cameras, "T1", "T1.png")
    frames = (counts(cameras[0]), counts(cameras[1]))
    indices = (np.full((512, 640), 1), np.full((512, 640), 2))

    frame, index = stitch.stitch(frames, indices, cameras, fused)

    # Bilinear sampling rounds off the kink at the axis by a few millidegrees.
    assert frame.astype(int) == pytest.approx(counts(fused).astype(int), abs=5)
    assert (index[:, 0] == 1).all() and (index[:, -1] == 2).all()
    # The seam is at the rig's axis, where the frames meet.
    seam = (index == 2).argmax(axis=1)
    assert np.abs(bearings(fused)[0, seam]).max() < 0.1


def test_two_frames_meet_halfway_between_their_levels() -> None:
    f = 320 / math.tan(math.radians(12.15))
    k = (f, f, 319.5, 255.5)
    cameras = (camera("l", k, toward(-11.4)), camera("r", k, toward(11.4)))
    fused = stitch.fused(*cameras, "T1", "T1.png")
    frames = (np.full((512, 640), 100, np.uint8), np.full((512, 640), 120, np.uint8))
    indices = (np.zeros((512, 640), int), np.zeros((512, 640), int))

    frame, _ = stitch.stitch(frames, indices, cameras, fused)

    assert (frame == 110).all()
