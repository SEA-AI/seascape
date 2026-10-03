"""What each camera saw, in COCO's detection format, written beside its images.

Imports no Blender, so a consumer reads it without the bpy wheel. Pixel coordinates
follow COCO, where pixel i spans [i, i + 1); calibration.json puts pixel centres on
integers, half a pixel off.
"""

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from pydantic import Field

from seascape.calibration import CameraCalibration
from seascape.model import Model

FILENAME = "labels.json"

# The horizon bows f dip (sec(hfov / 2) - 1) off its chord, and a segment's bow falls
# with the square of its length: 16 segments leave 1/256 of it.
HORIZON_POINTS = 17


class Target(NamedTuple):
    """A target as the object-index pass and the world see it."""

    pass_index: int
    name: str
    category: str
    centre_m: tuple[float, float]  # world east, north
    waterline_m: np.ndarray  # (N, 2) world east, north along the hull's waterline


class Image(Model):
    id: int
    file_name: str
    width: int
    height: int
    camera: str
    band: str
    time_s: float
    horizon_px: list[tuple[float, float]]


class Category(Model):
    id: int
    name: str


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
    truncated: bool  # the box touches the frame's edge


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
        targets: Sequence[Target],
        radius_m: float,
    ) -> None:
        """One frame: `index` is its object-index pass, (height, width), top row
        first."""
        image = Image(
            id=len(self.images) + 1,
            file_name=camera.image,
            width=camera.width_px,
            height=camera.height_px,
            camera=camera.name,
            band=camera.band,
            time_s=time_s,
            horizon_px=horizon_px(camera, radius_m),
        )
        self.images.append(image)
        at = np.array(camera.extrinsics["world"])[:2, 3]
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
                    category_id=self._category(target.category),
                    bbox=(x0, y0, x1 - x0 + 1, y1 - y0 + 1),
                    area=len(xs),
                    name=target.name,
                    range_m=math.hypot(east, north),
                    waterline_range_m=float(
                        np.linalg.norm(target.waterline_m - at, axis=1).min()
                    ),
                    bearing_deg=math.degrees(math.atan2(east, north)),
                    truncated=x0 == 0
                    or y0 == 0
                    or x1 == image.width - 1
                    or y1 == image.height - 1,
                )
            )

    def _category(self, name: str) -> int:
        found = next((c.id for c in self.categories if c.name == name), None)
        if found is None:
            found = len(self.categories) + 1
            self.categories.append(Category(id=found, name=name))
        return found

    def write(self, folder: Path) -> Path:
        path = folder / FILENAME
        partial = path.with_name(f"{path.name}.partial")
        partial.write_text(self.model_dump_json(indent=2) + "\n")
        return partial.replace(path)


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
