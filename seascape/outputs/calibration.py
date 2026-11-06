"""The geometry a render was made with, written beside its images.

Imports no Blender, so a consumer reads it without the bpy wheel.
"""

import math
from pathlib import Path

import numpy as np
from pydantic import ConfigDict, Field

from seascape import atomic
from seascape.config import Crop
from seascape.model import Model
from seascape.physics import fisheye

FILENAME = "calibration.json"

type Row3 = tuple[float, float, float]
type Row4 = tuple[float, float, float, float]
type Matrix3 = tuple[Row3, Row3, Row3]
type Matrix4 = tuple[Row4, Row4, Row4, Row4]

# A +X right, +Y ahead, +Z up frame's axes as OpenCV's: +X right, +Y down, +Z ahead.
TO_CV = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def write_json(model: Model, path: Path) -> Path:
    """`model` as JSON at `path`, whole or not at all: a reader never sees half."""
    with atomic.replacing(path) as partial:
        partial.write_text(model.model_dump_json(indent=2) + "\n")
    return path


class _Record(Model):
    # Another writer's file, or a newer one, carries fields this reader has no use for.
    model_config = ConfigDict(extra="ignore")


class CameraCalibration(_Record):
    """A pinhole, or with `distortion` a fisheye, in OpenCV's conventions.

    Camera frame: +X right, +Y down, +Z along the optical axis. Pixel centres sit at
    integer coordinates.
    """

    name: str
    band: str
    image: str = Field(description="relative to the calibration file's folder")
    rig: str
    model: str | None = None
    pitch_um: float | None = Field(
        default=None, gt=0.0, description="Pixel pitch, where the camera sets one."
    )
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
    K: Matrix3
    distortion: tuple[float, float, float, float] | None = Field(
        default=None,
        description="OpenCV's fisheye k1..k4, with K's fx the scale of the distorted "
        "angle. Unset, a pinhole.",
    )
    extrinsics: dict[str, Matrix4] = Field(
        min_length=1,
        description=(
            "camera to each named frame: world (+X east, +Y north, +Z up, at sea "
            "level), vessel (+X starboard, +Y bow, +Z up, moving with "
            "the hull) and rig (the product, +Y along its axis)"
        ),
    )

    @property
    def hfov_deg(self) -> float:
        """The field angle across the image width."""
        (fx, _, cx), _, _ = self.K
        # A crop can put the principal point off centre.
        reach = np.array([cx + 0.5, self.width_px - 0.5 - cx])
        if self.distortion is None:
            return math.degrees(np.arctan(reach / fx).sum())
        angles = fisheye.theta(reach, self.distortion, fx)
        return math.degrees(angles.sum())

    def cropped(self, crop: Crop) -> "CameraCalibration":
        """The camera that sees what `crop` leaves of this one's frame."""
        (fx, _, cx), (_, fy, cy), _ = self.K
        return self.model_copy(
            update={
                "width_px": self.width_px - crop.left_px - crop.right_px,
                "height_px": self.height_px - crop.top_px - crop.bottom_px,
                "K": (
                    (fx, 0.0, cx - crop.left_px),
                    (0.0, fy, cy - crop.top_px),
                    (0.0, 0.0, 1.0),
                ),
            }
        )


class Calibration(_Record):
    """Every camera of a render, as `FILENAME` in its folder."""

    cameras: list[CameraCalibration] = Field(min_length=1)
    raw: list[CameraCalibration] = Field(
        default_factory=list,
        description="The cameras a rig's videos were made from, as they rendered.",
    )

    def write(self, folder: Path) -> Path:
        """Write to `folder`; the file written."""
        return write_json(self, folder / FILENAME)

    @classmethod
    def read(cls, folder: Path) -> "Calibration":
        """The calibration in `folder`."""
        return cls.model_validate_json((folder / FILENAME).read_text())
