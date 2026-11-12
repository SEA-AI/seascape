"""Two pinholes' frames stitched side by side, as the recorder's horizontal stitching
does (Core-Backend's `ThermalFusion`).
"""

import math

import cv2
import numpy as np

from seascape import angles
from seascape.outputs.calibration import CameraCalibration


def _turn(camera: CameraCalibration) -> np.ndarray:
    return np.array(camera.extrinsics["rig"])[:3, :3]


def homography(source: CameraCalibration, onto: CameraCalibration) -> np.ndarray:
    """Maps `onto`'s pixels to `source`'s, as cameras that turn about one point."""
    turn = _turn(source).T @ _turn(onto)
    return np.array(source.K) @ turn @ np.linalg.inv(onto.K)


def fused(
    left: CameraCalibration, right: CameraCalibration, name: str, image: str
) -> CameraCalibration:
    """The pinhole the recorder stitches `left` and `right` onto: their mean focal
    length and pitch, along the rig's axis, cropped to where both frames reach and
    rounded down to even.

    Args:
        left: The camera whose frame takes the left half.
        right: The camera whose frame takes the right half.
        name: The stitched camera's name.
        image: Its frame, relative to the calibration file's folder.
    """
    fx = (left.K[0][0] + right.K[0][0]) / 2
    fy = (left.K[1][1] + right.K[1][1]) / 2
    w, h = 2 * left.width_px, left.height_px
    # Each optical axis's elevation over the rig's horizontal.
    pitch = np.mean([math.asin(_turn(c)[2, 2]) for c in (left, right)])
    c, s = math.cos(pitch), math.sin(pitch)
    turn = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]) @ angles.TO_CV.T
    at = (np.array(left.extrinsics["rig"]) + np.array(right.extrinsics["rig"])) / 2
    rig = np.eye(4)
    rig[:3, :3], rig[:3, 3] = turn, at[:3, 3]
    # The rig's pose in each frame, from the left camera's.
    to_rig = np.linalg.inv(left.extrinsics["rig"])
    extrinsics = {
        frame: tuple(map(tuple, np.array(pose) @ to_rig @ rig))
        for frame, pose in left.extrinsics.items()
    }
    whole = left.model_copy(
        update={
            "name": name,
            "image": image,
            "width_px": w,
            "height_px": h,
            "K": ((fx, 0.0, w / 2), (0.0, fy, h / 2), (0.0, 0.0, 1.0)),
            "extrinsics": extrinsics,
        }
    )
    # The frames' corners in the whole.
    edges = np.array([[0, 0], [left.width_px, 0], [0, h], [left.width_px, h]], float)
    corners = np.stack(
        [
            cv2.perspectiveTransform(
                edges.reshape(-1, 1, 2), np.linalg.inv(homography(c, whole))
            ).reshape(4, 2)
            for c in (left, right)
        ]
    )
    top, bottom = float(corners[:, :2, 1].max()), float(corners[:, 2:, 1].min())
    (top_left, _, bottom_left, _), (_, right_top, _, right_bottom) = corners

    def across(a: np.ndarray, b: np.ndarray, y: float) -> float:
        return a[0] + (y - a[1]) * (b[0] - a[0]) / (b[1] - a[1])

    # Left and right alike, so the rig's axis stays at the centre.
    side = max(
        *(across(top_left, bottom_left, y) for y in (top, bottom)),
        *(w - across(right_top, right_bottom, y) for y in (top, bottom)),
    )
    width, height = int(w - 2 * side), int(bottom - top)
    return whole.model_copy(
        update={
            "width_px": width - width % 2,
            "height_px": height - height % 2,
            "K": ((fx, 0.0, w / 2 - side), (0.0, fy, h / 2 - top), (0.0, 0.0, 1.0)),
        }
    )


def stitch(
    frames: tuple[np.ndarray, np.ndarray],
    indices: tuple[np.ndarray, np.ndarray],
    cameras: tuple[CameraCalibration, CameraCalibration],
    onto: CameraCalibration,
) -> tuple[np.ndarray, np.ndarray]:
    """Two integer frames and their object-index passes as `onto` sees them. The
    frames blend linearly across each row's overlap; the index takes the nearer
    frame's.

    Args:
        frames: The left and right frames, top row first.
        indices: Their object-index passes.
        cameras: The cameras that took them.
        onto: The stitched camera, as `fused` makes it.
    """
    size = (onto.width_px, onto.height_px)

    def warp(image: np.ndarray, camera: CameraCalibration, flags: int) -> np.ndarray:
        return cv2.warpPerspective(
            image, homography(camera, onto), size, flags=flags | cv2.WARP_INVERSE_MAP
        )

    seen = [
        warp(np.ones((c.height_px, c.width_px), np.uint8), c, cv2.INTER_NEAREST)
        for c in cameras
    ]
    # Each row's overlap runs from the right frame's first column to the left's last.
    start = seen[1].argmax(axis=1)[:, None]
    end = onto.width_px - 1 - seen[0][:, ::-1].argmax(axis=1)[:, None]
    columns = np.arange(onto.width_px)
    share = np.clip((end - columns) / np.maximum(end - start, 1), 0.0, 1.0)
    # Each camera's AGC tones it alone, so the frames meet halfway between their
    # levels, measured where the recorder measures them.
    h, w = frames[0].shape[:2]
    rows = slice(h // 4, h - h // 4)
    step = (frames[0][rows, -(w // 8) :].mean() - frames[1][rows, : w // 8].mean()) / 2
    left, right = (
        warp(f.astype(np.float32) + shift, c, cv2.INTER_LINEAR)
        for f, c, shift in zip(frames, cameras, (-step, step), strict=True)
    )
    weight = share[..., None] if left.ndim == 3 else share
    blend = np.rint(weight * left + (1 - weight) * right)
    bounds = np.iinfo(frames[0].dtype)
    frame = np.clip(blend, bounds.min, bounds.max).astype(frames[0].dtype)
    nearest = [
        warp(i.astype(np.float32), c, cv2.INTER_NEAREST)
        for i, c in zip(indices, cameras, strict=True)
    ]
    index = np.where(share >= 0.5, *nearest).astype(indices[0].dtype)
    return frame, index
