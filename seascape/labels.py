"""What each camera saw, in COCO's detection format, written beside its images.

Imports no Blender, so a consumer reads it without the bpy wheel. Pixel coordinates
follow COCO, where pixel i spans [i, i + 1); calibration.json puts pixel centres on
integers, half a pixel off.
"""

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

import cv2
import numpy as np
from pydantic import Field

from seascape.calibration import CameraCalibration
from seascape.model import Model

FILENAME = "labels.json"

# The horizon bows f dip (sec(hfov / 2) - 1) off its chord, and a segment's bow falls
# with the square of its length: 16 segments leave 1/256 of it.
HORIZON_POINTS = 17

# Luminance from linear BT.709 R, G, B: the Y row of sRGB's RGB to XYZ (IEC 61966-2-1).
BT709 = np.array([0.2126, 0.7152, 0.0722])
# A contrast's background is a ring this wide around the target, this far off it: the
# pixel filter spreads the target into the pixels its index does not reach.
RING_PX, RING_GAP_PX = 3, 1


class Target(NamedTuple):
    """A target as the object-index pass and the world see it."""

    pass_index: int
    name: str
    category_id: int
    category: str
    supercategory: str
    centre_m: tuple[float, float]  # world east, north
    waterline_m: np.ndarray  # (N, 2) world east, north along the hull's waterline
    heading_deg: float  # true, clockwise from north
    dims_m: tuple[float, float, float]  # length, beam, height above the waterline


class Image(Model):
    id: int
    file_name: str
    width: int
    height: int
    camera: str
    band: str
    hfov_deg: float
    time_s: float
    horizon_px: list[tuple[float, float]]


class Category(Model):
    id: int
    name: str
    supercategory: str


class Annotation(Model):
    id: int
    image_id: int
    category_id: int
    bbox: tuple[int, int, int, int]  # x, y, width, height
    area: int  # pixels the target covers
    iscrowd: int = 0
    name: str
    # From the camera, over the sea: to the hull's centre, and to its nearest
    # waterline, which is what a range from the horizon measures.
    range_m: float
    waterline_range_m: float
    bearing_deg: float  # true, clockwise from north
    heading_deg: float  # true, clockwise from north
    # length, beam, height above the waterline, as built
    dims_m: tuple[float, float, float]
    truncated: bool  # the box touches the frame's edge
    # Weber's, unsigned, in the frame as written; None with no background around it.
    contrast: float | None


class Labels(Model):
    info: dict[str, Any] = Field(default_factory=dict)
    images: list[Image] = Field(default_factory=list)
    annotations: list[Annotation] = Field(default_factory=list)
    categories: list[Category] = Field(default_factory=list)

    def add(
        self,
        camera: CameraCalibration,
        time_s: float,
        index: np.ndarray,
        frame: np.ndarray,
        targets: Sequence[Target],
        radius_m: float,
    ) -> None:
        """One frame: `index` is its object-index pass, (height, width), and `frame`
        the image as written, RGB or one channel, both top row first."""
        image = Image(
            id=len(self.images) + 1,
            file_name=camera.image,
            width=camera.width_px,
            height=camera.height_px,
            camera=camera.name,
            band=camera.band,
            hfov_deg=math.degrees(2 * math.atan(camera.width_px / 2 / camera.K[0][0])),
            time_s=time_s,
            horizon_px=horizon_px(camera, radius_m),
        )
        self.images.append(image)
        at = np.array(camera.extrinsics["world"])[:2, 3]
        lum = luminance(frame)
        rows, columns = np.nonzero(index)
        seen = index[rows, columns]
        for target in targets:
            mine = seen == target.pass_index
            ys, xs = rows[mine], columns[mine]
            if not len(xs):
                continue
            x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            east, north = np.subtract(target.centre_m, at)
            self.annotations.append(
                Annotation(
                    id=len(self.annotations) + 1,
                    image_id=image.id,
                    category_id=self._category(target),
                    bbox=(x0, y0, x1 - x0 + 1, y1 - y0 + 1),
                    area=len(xs),
                    name=target.name,
                    range_m=math.hypot(east, north),
                    waterline_range_m=float(
                        np.linalg.norm(target.waterline_m - at, axis=1).min()
                    ),
                    bearing_deg=math.degrees(math.atan2(east, north)),
                    heading_deg=target.heading_deg,
                    dims_m=target.dims_m,
                    truncated=x0 == 0
                    or y0 == 0
                    or x1 == image.width - 1
                    or y1 == image.height - 1,
                    contrast=contrast(index, lum, target.pass_index),
                )
            )

    def _category(self, target: Target) -> int:
        if all(c.id != target.category_id for c in self.categories):
            self.categories.append(
                Category(
                    id=target.category_id,
                    name=target.category,
                    supercategory=target.supercategory,
                )
            )
        return target.category_id

    def write(self, folder: Path) -> Path:
        path = folder / FILENAME
        partial = path.with_name(f"{path.name}.partial")
        partial.write_text(self.model_dump_json(indent=2) + "\n")
        return partial.replace(path)


def merge(root: Path, folders: Sequence[Path]) -> Labels:
    """The labels in `folders` as one set for `root`."""
    merged = Labels()
    scenarios: dict[str, Any] = {}
    for folder in folders:
        part = Labels.model_validate_json((folder / FILENAME).read_text())
        prefix = folder.relative_to(root).as_posix()
        merged.info = merged.info or part.info
        scenarios[prefix] = part.info["scenario"]
        ids: dict[int, int] = {}
        for image in part.images:
            ids[image.id] = len(merged.images) + 1
            merged.images.append(
                image.model_copy(
                    update={
                        "id": ids[image.id],
                        "file_name": f"{prefix}/{image.file_name}",
                    }
                )
            )
        for annotation in part.annotations:
            merged.annotations.append(
                annotation.model_copy(
                    update={
                        "id": len(merged.annotations) + 1,
                        "image_id": ids[annotation.image_id],
                    }
                )
            )
        for category in part.categories:
            known = next((c for c in merged.categories if c.id == category.id), None)
            if known is None:
                merged.categories.append(category)
            elif known != category:
                raise ValueError(f"{folder}: category {category.id} is {known.name}")
    merged.info = {k: v for k, v in merged.info.items() if k != "scenario"}
    merged.info["scenarios"] = scenarios
    return merged


def luminance(frame: np.ndarray) -> np.ndarray:
    """A frame's relative luminance: 8-bit RGB decoded from sRGB (IEC 61966-2-1),
    float RGB as linear, one channel as written."""
    if frame.ndim == 2:
        return frame.astype(float)
    if frame.dtype == np.uint8:
        c = frame / 255
        frame = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return frame @ BT709


def contrast(index: np.ndarray, lum: np.ndarray, pass_index: int) -> float | None:
    """The mean of |L - L_background| / L_background over the target's pixels, the
    background a ring of pixels no target covers. Unsigned, so a dark hull under a
    bright superstructure does not cancel to nothing."""
    mask = (index == pass_index).astype(np.uint8)
    near = cv2.dilate(mask, np.ones((2 * RING_GAP_PX + 1,) * 2, np.uint8))
    far = cv2.dilate(mask, np.ones((2 * (RING_GAP_PX + RING_PX) + 1,) * 2, np.uint8))
    ring = (far > near) & (index == 0)
    if not ring.any():
        return None
    background = lum[ring].mean()
    if background <= 0:
        return None
    return float(np.abs(lum[mask > 0] - background).mean() / background)


def horizon_px(camera: CameraCalibration, radius_m: float) -> list[tuple[float, float]]:
    """Where rays from the camera graze the sea, at evenly spaced columns.

    The sea is z = -(x^2 + y^2) / 2R. A ray C + t d meets it where a t^2 + b t + c = 0,
    with a = (dx^2 + dy^2) / 2R, b = dz + (cx dx + cy dy) / R and
    c = cz + (cx^2 + cy^2) / 2R, and grazes it where b^2 = 4 a c. Down a column d is
    linear in the row, so grazing is a quadratic in the row; its root with b < 0
    grazes in front of the camera.
    """
    pose = np.array(camera.extrinsics["world"])
    rotation, (cx, cy, cz) = pose[:3, :3], pose[:3, 3]
    k_inv = np.linalg.inv(camera.K)
    u = np.linspace(0.0, camera.width_px, HORIZON_POINTS)
    # The world ray through (u, v) is p + v q, in calibration.json's pixels.
    p = rotation @ k_inv @ np.stack([u - 0.5, np.zeros_like(u), np.ones_like(u)])
    q = rotation @ k_inv @ np.array([0.0, 1.0, 0.0])
    b0 = p[2] + (cx * p[0] + cy * p[1]) / radius_m
    b1 = q[2] + (cx * q[0] + cy * q[1]) / radius_m
    s = 2 * (cz + (cx * cx + cy * cy) / (2 * radius_m)) / radius_m
    qa = b1 * b1 - s * (q[0] ** 2 + q[1] ** 2)
    qb = 2 * b0 * b1 - 2 * s * (p[0] * q[0] + p[1] * q[1])
    qc = b0 * b0 - s * (p[0] ** 2 + p[1] ** 2)
    with np.errstate(invalid="ignore"):  # nan: the column never meets the sea
        root = np.sqrt(qb * qb - 4 * qa * qc)
    roots = (-qb + np.stack([root, -root])) / (2 * qa)
    v = np.where(b0 + roots[0] * b1 < 0, roots[0], roots[1]) + 0.5
    keep = (v >= 0) & (v <= camera.height_px)
    return [(float(x), float(y)) for x, y in zip(u[keep], v[keep], strict=True)]
