"""What each camera saw, in COCO's detection format, written beside its images.

Imports no Blender, so a consumer reads it without the bpy wheel. Pixel coordinates
follow COCO, where pixel i spans [i, i + 1); calibration.json puts pixel centres on
integers, half a pixel off.
"""

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, NamedTuple

import cv2
import numpy as np
from pydantic import Field

from seascape.config import (
    Agc,
    Camera,
    Lens,
    Microbolometer,
    Processing,
    Sensor,
)
from seascape.model import Model
from seascape.outputs.calibration import CameraCalibration

FILENAME = "labels.json"

# The horizon bows f dip (sec(hfov / 2) - 1) off its chord, and a segment's bow falls
# with the square of its length: 16 segments leave 1/256 of it.
HORIZON_POINTS = 17

# Judgement: the excess below which a viewer stops seeing a target.
MIN_EXCESS_DE = 1.0
# Judgement: the narrowest flank, enough of a row for its median to outvote a wave.
FLANK_PX = 8
# Judgement: a target's own glare, blur and jpg ringing reach about this far.
FLANK_GAP_PX = 4


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
    engine_off_min: float = 0.0  # 0 is running


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
    # The camera as drawn, None where the image did not go through it.
    sensor: Sensor | None = None
    lens: Lens | None = None
    processing: Processing | None = None
    microbolometer: Microbolometer | None = None


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
    engine_off_min: float = 0.0  # 0 is running


class Dropped(Model):
    image_id: int
    name: str
    range_m: float
    bearing_deg: float  # true, clockwise from north
    reason: Literal["no_pixels", "low_contrast"]
    # For "low_contrast" only; bbox and area as an annotation's.
    excess_de: float | None = None
    bbox: tuple[int, int, int, int] | None = None
    area: int | None = None


class Labels(Model):
    info: dict[str, Any] = Field(default_factory=dict)
    images: list[Image] = Field(default_factory=list)
    annotations: list[Annotation] = Field(default_factory=list)
    dropped: list[Dropped] = Field(default_factory=list)
    categories: list[Category] = Field(default_factory=list)

    def add(
        self,
        camera: CameraCalibration,
        time_s: float,
        index: np.ndarray,
        frame: np.ndarray,
        targets: Sequence[Target],
        radius_m: float,
        settings: Camera | None = None,
        *,
        faint: Mapping[str, float],
    ) -> None:
        """One frame: `index` is its object-index pass, (height, width), and `frame`
        the image as written, both top row first. `settings` is the scenario's camera,
        whose sensor, lens and processing an EO image carries, and lens and
        microbolometer an LWIR one. A target in `faint`, by name, is dropped with its
        excess ΔE*ab."""
        # Only an 8-bit EO frame went through the camera's chain, and only an integer
        # LWIR one (png, jpg) through the microbolometer.
        grade = (
            settings is not None and settings.band == "eo" and frame.dtype == np.uint8
        )
        thermal = (
            settings is not None
            and settings.band == "ir"
            and np.issubdtype(frame.dtype, np.integer)
        )
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
            sensor=settings.sensor if grade else None,
            lens=settings.lens if grade or thermal else None,
            processing=settings.processing if grade else None,
            microbolometer=settings.microbolometer if thermal else None,
        )
        self.images.append(image)
        at = np.array(camera.extrinsics["world"])[:2, 3]
        rows, columns = np.nonzero(index)
        seen = index[rows, columns]
        for target in targets:
            mine = seen == target.pass_index
            ys, xs = rows[mine], columns[mine]
            east, north = np.subtract(target.centre_m, at)
            where: dict[str, Any] = {
                "image_id": image.id,
                "name": target.name,
                "range_m": math.hypot(east, north),
                "bearing_deg": math.degrees(math.atan2(east, north)),
            }
            if not len(xs):
                self.dropped.append(Dropped(**where, reason="no_pixels"))
                continue
            x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            bbox = (x0, y0, x1 - x0 + 1, y1 - y0 + 1)
            if target.name in faint:
                self.dropped.append(
                    Dropped(
                        **where,
                        reason="low_contrast",
                        excess_de=faint[target.name],
                        bbox=bbox,
                        area=len(xs),
                    )
                )
                continue
            self.annotations.append(
                Annotation(
                    id=len(self.annotations) + 1,
                    image_id=image.id,
                    category_id=self._category(target),
                    bbox=bbox,
                    area=len(xs),
                    name=target.name,
                    range_m=where["range_m"],
                    waterline_range_m=float(
                        np.linalg.norm(target.waterline_m - at, axis=1).min()
                    ),
                    bearing_deg=where["bearing_deg"],
                    heading_deg=target.heading_deg,
                    dims_m=target.dims_m,
                    engine_off_min=target.engine_off_min,
                    truncated=x0 == 0
                    or y0 == 0
                    or x1 == image.width - 1
                    or y1 == image.height - 1,
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
        for gone in part.dropped:
            merged.dropped.append(
                gone.model_copy(update={"image_id": ids[gone.image_id]})
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


def present(index: np.ndarray, targets: Sequence[Target]) -> set[str]:
    return {t.name for t in targets if (index == t.pass_index).any()}


def agreed(
    faints: Sequence[Mapping[str, float]], presents: Sequence[set[str]]
) -> list[dict[str, float]]:
    """Each camera's faint targets, less those any camera of the same instant sees."""
    seen = set().union(
        *(names - low.keys() for low, names in zip(faints, presents, strict=True))
    )
    return [{n: e for n, e in low.items() if n not in seen} for low in faints]


def faint(
    index: np.ndarray, frame: np.ndarray, targets: Sequence[Target]
) -> dict[str, float]:
    """The excess ΔE*ab of each target in an 8-bit frame that falls short of
    MIN_EXCESS_DE. A deeper frame never went through a camera, so nothing is faint."""
    if frame.dtype != np.uint8:
        return {}
    low = {}
    for target in targets:
        ys, xs = np.nonzero(index == target.pass_index)
        if not len(xs):
            continue
        reach = FLANK_GAP_PX + flank_px(xs.max() - xs.min() + 1)
        window = np.s_[
            ys.min() : ys.max() + 1, max(xs.min() - reach, 0) : xs.max() + 1 + reach
        ]
        excess = excess_de(index[window], frame[window], target.pass_index)
        if excess is not None and excess < MIN_EXCESS_DE:
            low[target.name] = excess
    return low


def excess_de(index: np.ndarray, frame: np.ndarray, pass_index: int) -> float | None:
    """The target's mean ΔE*ab from its background, less its flanks' mean from the
    same background; None if no row of the target has a flank. A local target-to-
    clutter contrast (Schmieder & Weathersby 1983)."""
    lab = cielab(frame)
    mask = index == pass_index
    beside = flanks(index, pass_index)
    behind = background(lab, mask, beside)
    if behind is None:
        return None
    delta_e = np.linalg.norm(lab - behind, axis=2)
    return float(delta_e[mask].mean() - delta_e[beside].mean())


def cielab(frame: np.ndarray) -> np.ndarray:
    """An 8-bit RGB or grey frame in CIE 1976 L*a*b* (ISO/CIE 11664-4)."""
    grey = frame.ndim == 2
    if grey:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    # Float, as 8-bit Lab is rescaled to 0 to 255.
    lab = cv2.cvtColor(frame.astype(np.float32) / 255, cv2.COLOR_RGB2Lab)
    if grey:
        # OpenCV's D65 leaves grey a chroma of about 0.01 that bends with L*.
        lab[..., 1:] = 0
    return lab


def flank_px(span_px: int) -> int:
    """Each flank's width beside a row `span_px` wide: half the span, so flank and
    target pixels number about the same, and never under FLANK_PX."""
    return max(FLANK_PX, -(-span_px // 2))


def flanks(index: np.ndarray, pass_index: int) -> np.ndarray:
    """The pixels beside the target on each of its rows, past FLANK_GAP_PX of it in any
    direction, that no object, the ownship included, covers."""
    mask = index == pass_index
    beside = np.zeros_like(mask)
    for y in np.flatnonzero(mask.any(axis=1)):
        xs = np.flatnonzero(mask[y])
        reach = FLANK_GAP_PX + flank_px(xs[-1] - xs[0] + 1)
        beside[y, max(xs[0] - reach, 0) : xs[0]] = True
        beside[y, xs[-1] + 1 : xs[-1] + 1 + reach] = True
    gap = np.ones((2 * FLANK_GAP_PX + 1,) * 2, np.uint8)
    near = cv2.dilate(mask.astype(np.uint8), gap).astype(bool)
    return beside & (index == 0) & ~near


def background(
    lab: np.ndarray, mask: np.ndarray, beside: np.ndarray
) -> np.ndarray | None:
    """At the pixels of `mask` and its flanks, each row's `_line`, or the nearest
    row's that has a flank; NaN elsewhere. None if no row has a flank."""
    rows = np.flatnonzero(mask.any(axis=1))
    lines = {y: _line(lab[y], mask[y], beside[y]) for y in rows if beside[y].any()}
    if not lines:
        return None
    measured = np.array(list(lines))
    behind = np.full(lab.shape, np.nan, np.float32)
    for y in rows:
        points = lines[measured[np.abs(measured - y).argmin()]]
        where = np.flatnonzero(mask[y] | beside[y])
        if len(points) == 1:
            behind[y, where] = points[0][1]
            continue
        (xl, left), (xr, right) = points
        t = (where - xl) / (xr - xl)
        if y not in lines:
            # A borrowed line holds level past its flanks rather than run off them.
            t = t.clip(0, 1)
        behind[y, where] = left + np.outer(t, right - left)
    return behind


def _line(
    lab: np.ndarray, mask: np.ndarray, beside: np.ndarray
) -> list[tuple[float, np.ndarray]]:
    """Each flank of a row as (mean column, median L*a*b*)."""
    x0 = np.flatnonzero(mask)[0]
    xs = np.flatnonzero(beside)
    sides = [xs[xs < x0], xs[xs > x0]]
    return [(s.mean(), np.median(lab[s], axis=0)) for s in sides if len(s)]


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


def agcs(folder: Path) -> dict[str, Agc]:
    """Each camera's AGC, by name, from the scenario the labels in `folder` record,
    the default for a camera they do not, or for every camera in a folder without
    labels. Only the `agc` tables are read, so a scenario recorded by an older
    seascape still serves."""
    found: dict[str, Agc] = defaultdict(Agc)
    path = folder / FILENAME
    if not path.exists():
        return found
    info = Labels.model_validate_json(path.read_text()).info
    # A merged set records one scenario per seed, and seeds can draw different AGCs.
    recorded = info.get("scenarios") or {"": info.get("scenario") or {}}
    for scenario in recorded.values():
        for rig_name, rig in scenario.get("rigs", {}).items():
            for camera_name, camera in rig.get("cameras", {}).items():
                name = f"{rig_name}_{camera_name}"
                setting = Agc.model_validate(camera.get("agc", {}))
                if found.get(name, setting) != setting:
                    raise ValueError(
                        f"{name}: its seeds drew different agcs; tone each seed's "
                        "folder on its own"
                    )
                found[name] = setting
    return found
