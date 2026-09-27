"""Stitch a render's frames into panoramas, from its calibration.json.

The poses are known, so this is OpenCV's stitching pipeline with its estimation
stages skipped. No Blender.

Frames are stitched as they are, with no exposure compensation.
"""

import math
from functools import cache
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

from seascape import agc
from seascape.calibration import Calibration, CameraCalibration
from seascape.montage import INK, MATTE

# CLI name to cv2.PyRotationWarper type.
PROJECTIONS = {
    "rectilinear": "plane",
    "cylindrical": "cylindrical",
    "equirectangular": "spherical",
}

TICK_DEG = 10
LABEL_DEG = 30

# The stitcher's frame is +X right, +Y down, +Z ahead.
_TO_CV = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def rotation(camera: CameraCalibration, frame: str) -> np.ndarray:
    return np.array(camera.extrinsics[frame])[:3, :3]


def axis(cameras: list[CameraCalibration], frame: str) -> float:
    """Bearing of the summed optical axes, which cannot wrap as a mean of angles can."""
    x, y, _ = np.sum([rotation(camera, frame)[:, 2] for camera in cameras], axis=0)
    return float(np.arctan2(x, y))


def pose(
    camera: CameraCalibration, frame: str, bearing: float
) -> tuple[np.ndarray, np.ndarray]:
    """K and R as the stitcher takes them, turned so `bearing` is straight ahead."""
    c, s = np.cos(bearing), np.sin(bearing)
    turn = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return (
        np.array(camera.K, np.float32),
        np.array(_TO_CV @ turn @ rotation(camera, frame), np.float32),
    )


class Layout(NamedTuple):
    """Where a stitched panorama sits in its projection."""

    kind: str
    scale: float
    axis: float  # the bearing the panorama is turned to, radians, in its frame
    left: int  # projection x of column 0
    top: int  # projection y of row 0
    resize: float  # the final shrink to max_width, 1.0 when none


def column(layout: Layout, bearing_deg: float) -> float | None:
    """The column a ray at `bearing_deg` lands in, at any elevation, or None where
    the projection cannot show it.

    A camera with K = I looks along its rotation's third column, so a turn about the
    stitcher's vertical aims it at the bearing, and the warper projects it exactly
    as it projected the frames.
    """
    b = math.radians(bearing_deg) - layout.axis
    if layout.kind == "plane" and math.cos(b) <= 0.0:
        return None
    c, s = math.cos(b), math.sin(b)
    r = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], np.float32)
    u, _ = _warper(layout.kind, layout.scale).warpPoint(
        (0.0, 0.0), np.eye(3, dtype=np.float32), r
    )
    # Pixel centres sit at integers, so the resize scales about -0.5.
    return (u - layout.left + 0.5) * layout.resize - 0.5


@cache
def _warper(kind: str, scale: float) -> cv2.PyRotationWarper:
    return cv2.PyRotationWarper(kind, scale)


def ruled(image: np.ndarray, layout: Layout, true: bool = False) -> np.ndarray:
    """`image` over a strip of bearing ticks, in degrees of its frame: 000-359 when
    `true`, as a chart writes true bearings, else signed about the frame's axis."""
    w = image.shape[1]
    size = max(1.0, w / 2000)
    strip = np.full((round(36 * size), w, 3), MATTE, np.uint8)
    tick = round(8 * size)
    font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.5 * size
    line = max(1, round(size))
    for bearing in range(-180, 180, TICK_DEG):
        x = column(layout, bearing)
        if x is None or not 0 <= x < w:
            continue
        x = round(x)
        labelled = bearing % LABEL_DEG == 0
        cv2.line(strip, (x, 0), (x, tick * (2 if labelled else 1)), INK, line)
        if labelled:
            if true:
                text = f"{bearing % 360:03d}"
            else:
                text = f"{bearing:+d}" if bearing else "0"
            (tw, th), _ = cv2.getTextSize(text, font, scale, line)
            origin = (x - tw // 2, 2 * tick + th + round(4 * size))
            cv2.putText(strip, text, origin, font, scale, INK, line, cv2.LINE_AA)
    return np.vstack([image, strip])


def stitch(
    folder: Path,
    cameras: list[CameraCalibration],
    projection: str,
    frame: str,
    max_width: int | None = None,
) -> tuple[np.ndarray, Layout]:
    """At native resolution unless that is wider than `max_width`. Native is the
    largest fx: pixels per radian on axis, where every projection here runs at one
    unit per radian."""
    if max_width is not None and max_width < 1:
        raise ValueError(f"max width is {max_width}: it must be a pixel or more")
    if missing := [c.name for c in cameras if frame not in c.extrinsics]:
        raise ValueError(f"no {frame!r} extrinsics for {', '.join(missing)}")
    kind = PROJECTIONS[projection]
    bearing = axis(cameras, frame)
    poses = [pose(camera, frame, bearing) for camera in cameras]
    sizes = [(camera.width_px, camera.height_px) for camera in cameras]
    if kind == "plane" and not all(
        _faces_ahead(k, r, size) for (k, r), size in zip(poses, sizes, strict=True)
    ):
        raise ValueError(
            "rectilinear cannot show a ray 90 deg or more off its axis: "
            "use cylindrical or equirectangular"
        )

    scale = max(float(k[0, 0]) for k, _ in poses)
    if max_width is not None:
        native = cv2.PyRotationWarper(kind, scale)
        rois = [
            native.warpRoi(size, k, r)
            for size, (k, r) in zip(sizes, poses, strict=True)
        ]
        _, _, span, _ = cv2.detail.resultRoi(
            corners=[roi[:2] for roi in rois], sizes=[roi[2:] for roi in rois]
        )
        scale *= min(1.0, max_width / span)
    warper = cv2.PyRotationWarper(kind, scale)

    temps = [agc.kelvin(folder / camera.image) for camera in cameras]
    # One span for all, so a temperature is one grey on both sides of a seam.
    known = [t_k for t_k in temps if t_k is not None]
    low_k = min((float(t_k.min()) for t_k in known), default=0.0)
    high_k = max((float(t_k.max()) for t_k in known), default=0.0)
    warped = []
    for camera, (k_full, r), t_k in zip(cameras, poses, temps, strict=True):
        path = folder / camera.image
        image = (
            cv2.imread(str(path), cv2.IMREAD_COLOR)
            if t_k is None
            else cv2.cvtColor(agc.grey(t_k, low_k, high_k), cv2.COLOR_GRAY2BGR)
        )
        if image is None:
            raise ValueError(
                f"cannot read {path}: panorama takes the frames that "
                'outputs.format = "png" or "jpg" writes'
            )
        image, k = _shrink(image, k_full, scale / float(k_full[0, 0]))
        corner, pixels = warper.warp(image, k, r, cv2.INTER_LINEAR, cv2.BORDER_REFLECT)
        mask = np.full(image.shape[:2], 255, np.uint8)
        _, mask = warper.warp(mask, k, r, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)
        warped.append((corner, pixels, mask))

    roi = cv2.detail.resultRoi(
        corners=[c for c, _, _ in warped], sizes=[m.shape[1::-1] for _, _, m in warped]
    )
    blender = cv2.detail.MultiBandBlender()
    blender.prepare(roi)
    for corner, pixels, mask in warped:
        blender.feed(pixels.astype(np.int16), mask, corner)
    image, _ = blender.blend(np.empty(0, np.int16), np.empty(0, np.uint8))
    # The pyramid overshoots at edges; clip, where convertScaleAbs would fold it back.
    image = np.clip(image, 0, 255).astype(np.uint8)
    h, w = image.shape[:2]
    resize = 1.0
    if max_width is not None and w > max_width:
        # The warper rounds its extent outward, a pixel past the scale asked for.
        resize = max_width / w
        size = (max_width, round(h * resize))
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return image, Layout(kind, scale, bearing, roi[0], roi[1], resize)


def _faces_ahead(k: np.ndarray, r: np.ndarray, size: tuple[int, int]) -> bool:
    """Whether a whole frame lies less than 90 deg off the stitcher's +Z. Depth is
    linear across the frame, so the corners decide."""
    w, h = size
    corners = np.array([[0, 0, 1], [w - 1, 0, 1], [0, h - 1, 1], [w - 1, h - 1, 1]])
    _, _, z = r @ np.linalg.inv(k) @ corners.T
    return bool((z > 0).all())


def _shrink(
    image: np.ndarray, k: np.ndarray, factor: float
) -> tuple[np.ndarray, np.ndarray]:
    """Area-average a frame down before warping: the warper samples it bilinearly,
    which aliases anything finer than the output grid."""
    if factor >= 1.0:
        return image, k
    h, w = image.shape[:2]
    small = cv2.resize(
        image, (round(w * factor), round(h * factor)), interpolation=cv2.INTER_AREA
    )
    sx, sy = small.shape[1] / w, small.shape[0] / h
    # Pixel centres sit at integers, so the principal point scales about -0.5.
    k = np.array(
        [
            [k[0, 0] * sx, 0.0, (k[0, 2] + 0.5) * sx - 0.5],
            [0.0, k[1, 1] * sy, (k[1, 2] + 0.5) * sy - 0.5],
            [0.0, 0.0, 1.0],
        ],
        np.float32,
    )
    return small, k


def panoramas(
    folder: Path,
    projection: str,
    frame: str,
    max_width: int | None = None,
    ruler: bool = False,
) -> list[Path]:
    """One panorama per pod and band, written beside the frames."""
    groups: dict[tuple[str | None, str], list[CameraCalibration]] = {}
    for camera in Calibration.read(folder).cameras:
        groups.setdefault((camera.pod, camera.band), []).append(camera)

    written = []
    for (pod, band), cameras in groups.items():
        name = "_".join(part for part in (pod, band, projection, frame) if part)
        path = folder / f"panorama_{name}.png"
        try:
            image, layout = stitch(folder, cameras, projection, frame, max_width)
        except cv2.error as error:
            raise RuntimeError(f"{path.name}: {error.err}") from error
        if ruler:
            # World bearings are true bearings; every other frame's are relative.
            image = ruled(image, layout, true=frame == "world")
        cv2.imwrite(str(path), image)
        written.append(path)
    return written
