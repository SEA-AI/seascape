"""Stitch a render's frames into panoramas, from its calibration.json.

The poses are known, so this is OpenCV's stitching pipeline with its estimation
stages skipped: warp each frame and its mask, blend, and draw on what the blender and
masks give back. No Blender.

Frames are stitched as they are, with no exposure compensation.
"""

import math
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

from seascape import angles
from seascape.config import BANDS
from seascape.outputs import agc, canvas
from seascape.outputs.calibration import Calibration, CameraCalibration, Frame
from seascape.outputs.labels import agcs

# CLI name to cv2.PyRotationWarper type.
PROJECTIONS = {
    "rectilinear": "plane",
    "cylindrical": "cylindrical",
    "equirectangular": "spherical",
}

TICK_DEG = 10
LABEL_DEG = 30
# Judgement: the band beneath stays readable through the one laid over it.
OPACITY = 0.6


class Layout(NamedTuple):
    """Where a stitched panorama sits in its projection."""

    kind: str
    scale: float
    axis_deg: float  # the bearing the panorama is turned to, in its frame
    left: int  # projection x of column 0


def _rotation(camera: CameraCalibration, frame: Frame) -> np.ndarray:
    return np.array(camera.extrinsics[frame])[:3, :3]


def _rays(camera: CameraCalibration, frame: Frame) -> np.ndarray:
    """(3, N) rays through a grid of a pinhole's pixel centres, edges included, in
    `frame`.
    """
    u, v = np.meshgrid(
        np.linspace(0, camera.width_px - 1, 32),
        np.linspace(0, camera.height_px - 1, 32),
    )
    pixels = np.stack([u.ravel(), v.ravel(), np.ones(u.size)])
    return _rotation(camera, frame) @ np.linalg.inv(camera.K) @ pixels


def axis_deg(cameras: list[CameraCalibration], frame: Frame) -> float:
    """Azimuth in `frame` halfway across what the cameras see: the middle of the arc
    their rays cover, which ends where the widest gap between them begins.
    """
    rays = np.hstack([_rays(c, frame) for c in cameras])
    azimuths_deg = np.sort(angles.bearing_deg(rays[0], rays[1]))
    gaps_deg = np.diff(azimuths_deg, append=azimuths_deg[0] + 360.0)
    widest = int(np.argmax(gaps_deg))
    start_deg = azimuths_deg[(widest + 1) % len(azimuths_deg)]
    return angles.wrapped_deg(start_deg + (360.0 - gaps_deg[widest]) / 2)


def pose(
    camera: CameraCalibration, frame: Frame, bearing_deg: float
) -> tuple[np.ndarray, np.ndarray]:
    """K and R as the stitcher takes them, turned so `bearing_deg` is straight ahead."""
    if camera.distortion is not None:
        raise ValueError(f"{camera.name}: a panorama stitches pinhole cameras")
    # The panorama's frame: +Y along the bearing.
    R_panorama_frame = angles.turned(bearing_deg, 0.0).T
    R = angles.TO_CV @ R_panorama_frame @ _rotation(camera, frame)
    return np.array(camera.K, np.float32), np.array(R, np.float32)


def column(layout: Layout, bearing_deg: float) -> float | None:
    """The column a ray at a bearing lands in, at any elevation, or None where the
    projection cannot show it.

    A camera with K = I looks along its rotation's third column, so a turn about the
    stitcher's vertical aims it at the bearing, and the warper projects it exactly
    as it projected the frames.

    Args:
        layout: Where the panorama sits in its projection.
        bearing_deg: Clockwise from the frame's +Y, north in the world frame.
    """
    off = math.radians(bearing_deg - layout.axis_deg)
    if layout.kind == "plane" and math.cos(off) <= 0.0:
        return None
    turn, _ = cv2.Rodrigues(np.array([0.0, off, 0.0]))
    warper = cv2.PyRotationWarper(layout.kind, layout.scale)
    u, _ = warper.warpPoint(
        (0.0, 0.0), np.eye(3, dtype=np.float32), turn.astype(np.float32)
    )
    return u - layout.left


def _ink(width_px: int) -> float:
    """Scale of lines and text on a panorama `width_px` wide: 1 to 2000 px, then in
    proportion.
    """
    return max(1.0, width_px / 2000)


def ruled(image: np.ndarray, layout: Layout, true: bool = False) -> np.ndarray:
    """A panorama over a strip of bearing ticks, in degrees of its frame.

    Args:
        image: The panorama, 8-bit BGR as `stitch` makes it.
        layout: Where it sits in its projection.
        true: Label 000-359, as a chart writes true bearings; else signed about the
            frame's axis.
    """
    w = image.shape[1]
    size = _ink(w)
    strip = np.full((round(36 * size), w, 3), canvas.MATTE, np.uint8)
    tick, line = round(8 * size), max(1, round(size))
    font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.5 * size
    for bearing in range(-180, 180, TICK_DEG):
        x = column(layout, bearing)
        if x is None or not 0 <= x < w:
            continue
        x, labelled = round(x), bearing % LABEL_DEG == 0
        cv2.line(strip, (x, 0), (x, tick * (2 if labelled else 1)), canvas.INK, line)
        if labelled:
            text = (
                f"{bearing % 360:03d}" if true else f"{bearing:+d}" if bearing else "0"
            )
            (tw, th), _ = cv2.getTextSize(text, font, scale, line)
            origin = (x - tw // 2, 2 * tick + th + round(4 * size))
            cv2.putText(strip, text, origin, font, scale, canvas.INK, line, cv2.LINE_AA)
    return np.vstack([image, strip])


def _frames(folder: Path, cameras: list[CameraCalibration]) -> Iterator[np.ndarray]:
    """Each camera's frame, 8-bit BGR, every thermal one through one transfer curve,
    so a temperature is one grey on both sides of a seam.
    """
    temps = {c.name: agc.kelvin(folder / c.image) for c in cameras}
    thermal = {name: t_k for name, t_k in temps.items() if t_k is not None}
    if thermal:
        every = agcs(folder)
        setting, *others = [every[name] for name in thermal]
        if any(other != setting for other in others):
            raise ValueError("a panorama's cameras need one agc, to share one curve")
        curve = agc.transfer(list(thermal.values()), setting)
    for camera in cameras:
        path = folder / camera.image
        image = (
            cv2.cvtColor(
                agc.grey(thermal[camera.name], curve, setting), cv2.COLOR_GRAY2BGR
            )
            if camera.name in thermal
            else cv2.imread(str(path), cv2.IMREAD_COLOR)
        )
        if image is None:
            raise ValueError(
                f"cannot read {path}: panorama takes the frames that "
                'outputs.format = "png" or "jpg" writes'
            )
        h, w = image.shape[:2]
        if (w, h) != (camera.width_px, camera.height_px):
            raise ValueError(
                f"{path} is {w}x{h} px; its calibration says "
                f"{camera.width_px}x{camera.height_px}"
            )
        yield image


def _shrunk(
    size: tuple[int, int], k: np.ndarray, factor: float
) -> tuple[tuple[int, int], np.ndarray]:
    """The size and K of a frame area-averaged down before warping: the warper
    samples it bilinearly, which aliases anything finer than the output grid.
    """
    if factor >= 1.0:
        return size, k
    w, h = size
    small = (max(1, round(w * factor)), max(1, round(h * factor)))
    sx, sy = small[0] / w, small[1] / h
    # Pixel centres sit at integers, so the principal point scales about -0.5.
    resize = np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]])
    return small, (resize @ k).astype(np.float32)


def _warped(
    folder: Path,
    cameras: list[CameraCalibration],
    projection: str,
    frame: Frame,
    max_width: int | None,
) -> tuple[Layout, Sequence[int], list[tuple]]:
    """Every camera's frame and mask warped onto one canvas, centred on what they see
    and scaled to `max_width`; the layout, the canvas's roi and the warps.
    """
    if max_width is not None and max_width < 1:
        raise ValueError(f"max width is {max_width}: it must be a pixel or more")
    if missing := [c.name for c in cameras if frame not in c.extrinsics]:
        raise ValueError(f"no {frame!r} extrinsics for {', '.join(missing)}")
    kind = PROJECTIONS[projection]
    bearing_deg = axis_deg(cameras, frame)
    poses = [pose(camera, frame, bearing_deg) for camera in cameras]
    if kind == "plane":
        ahead = angles.direction(bearing_deg, 0.0)
        if (ahead @ np.hstack([_rays(c, frame) for c in cameras]) <= 0).any():
            raise ValueError(
                "rectilinear cannot show a ray 90 deg or more off its axis: "
                "use cylindrical or equirectangular"
            )

    def canvas_of(scale: float) -> tuple[list, Sequence[int]]:
        warper = cv2.PyRotationWarper(kind, scale)
        shrunk = [
            (*_shrunk((c.width_px, c.height_px), k, scale / float(k[0, 0])), r)
            for c, (k, r) in zip(cameras, poses, strict=True)
        ]
        rois = [warper.warpRoi(size, k, r) for size, k, r in shrunk]
        roi = cv2.detail.resultRoi(
            corners=[roi[:2] for roi in rois], sizes=[roi[2:] for roi in rois]
        )
        return shrunk, roi

    # Native resolution is the largest fx: pixels per radian on axis, where every
    # projection here runs at one unit per radian.
    scale = max(float(k[0, 0]) for k, _ in poses)
    if max_width is not None:
        scale *= min(1.0, max_width / canvas_of(scale)[1][2])
    shrunk, roi = canvas_of(scale)
    warper = cv2.PyRotationWarper(kind, scale)
    warped = []
    for full, (size, k, r) in zip(_frames(folder, cameras), shrunk, strict=True):
        image = cv2.resize(full, size, interpolation=cv2.INTER_AREA)
        corner, pixels = warper.warp(image, k, r, cv2.INTER_LINEAR, cv2.BORDER_REFLECT)
        mask = np.full(image.shape[:2], 255, np.uint8)
        _, mask = warper.warp(mask, k, r, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)
        warped.append((corner, pixels, mask))
    return Layout(kind, scale, bearing_deg, roi[0]), roi, warped


def _blend(roi: Sequence[int], warped: list[tuple]) -> tuple[np.ndarray, np.ndarray]:
    """The warps blended on the canvas, 8-bit BGR, and where they cover it."""
    blender = cv2.detail.MultiBandBlender()
    blender.prepare(roi)
    for corner, pixels, mask in warped:
        blender.feed(pixels.astype(np.int16), mask, corner)
    image, mask = blender.blend(np.empty(0, np.int16), np.empty(0, np.uint8))
    # The pyramid overshoots at edges; clip, where convertScaleAbs would fold it back.
    return np.clip(image, 0, 255).astype(np.uint8), mask


def stitch(
    folder: Path,
    cameras: list[CameraCalibration],
    projection: str,
    frame: Frame,
    max_width: int | None = None,
) -> tuple[np.ndarray, Layout]:
    """One panorama of the cameras' frames, and where it sits in its projection.

    Args:
        folder: The render, its frames and its agc settings.
        cameras: The cameras to stitch; thermal ones need one agc.
        projection: A key of `PROJECTIONS`; rectilinear cannot show a ray 90 deg or
            more off its axis.
        frame: The calibration frame whose bearings lay the panorama out; world's
            are true.
        max_width: The widest the panorama may be, in pixels; None keeps native
            resolution.
    """
    layout, roi, warped = _warped(folder, cameras, projection, frame, max_width)
    image, _ = _blend(roi, warped)
    # The warper rounds its extent a pixel outward.
    return image[:, :max_width], layout


def overlaid(
    folder: Path,
    cameras: list[CameraCalibration],
    projection: str,
    frame: Frame,
    max_width: int | None = None,
) -> tuple[np.ndarray, Layout]:
    """Every band of `cameras` in one panorama, and where it sits in its projection.

    Each band is blended on one canvas and laid in `BANDS`' order: at `OPACITY`
    over what lies beneath, opaque where nothing does. Every frame is outlined in
    its band's colour and named, the first band's under their top edge and the
    rest's over their bottom, so coaxial frames' names stay apart.

    Args:
        folder: The render, its frames and its agc settings.
        cameras: The cameras to lay out; thermal ones need one agc.
        projection: As `stitch` takes it.
        frame: As `stitch` takes it.
        max_width: As `stitch` takes it, across every band.
    """
    layout, roi, warped = _warped(folder, cameras, projection, frame, max_width)
    left, top, w, h = roi
    image = np.full((h, w, 3), canvas.MATTE, np.uint8)
    covered = np.zeros((h, w), bool)
    bands = [b for b in BANDS if any(c.band == b for c in cameras)]
    for band in bands:
        layer, mask = _blend(
            roi, [x for c, x in zip(cameras, warped, strict=True) if c.band == band]
        )
        seen = mask > 0
        mixed = cv2.addWeighted(layer, OPACITY, image, 1 - OPACITY, 0)
        image[seen] = np.where(covered[seen, None], mixed[seen], layer[seen])
        covered |= seen

    size = _ink(w)
    line, gap = max(1, round(2 * size)), 10 * size
    font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.8 * size
    # The palette is RGB; cv2 draws BGR.
    colours = {
        b: c[::-1].tolist()
        for b, c in zip(BANDS, canvas.palette(len(BANDS)), strict=True)
    }
    for camera, ((x, y), _, mask) in zip(cameras, warped, strict=True):
        edge, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE, offset=(x - left, y - top)
        )
        # Judgement: a pixel's tolerance straightens the mask's staircase.
        edge = [cv2.approxPolyDP(e, 1.0, closed=True) for e in edge]
        cv2.drawContours(image, edge, -1, colours[camera.band], line, cv2.LINE_AA)
    # Every edge before any name, so no later edge strikes through one.
    warper = cv2.PyRotationWarper(layout.kind, layout.scale)
    for camera in cameras:
        first = camera.band == bands[0]
        edge = (camera.K[0][2], 0.0 if first else camera.height_px - 1.0)
        x, y = warper.warpPoint(edge, *pose(camera, frame, layout.axis_deg))
        (tw, th), _ = cv2.getTextSize(camera.name, font, scale, line)
        origin = (
            round(x - left - tw / 2),
            round(y - top + (th + gap if first else -gap)),
        )
        colour = colours[camera.band]
        cv2.putText(image, camera.name, origin, font, scale, colour, line, cv2.LINE_AA)
    return image[:, :max_width], layout


def panoramas(
    folder: Path,
    projection: str,
    frame: Frame,
    max_width: int | None = None,
    ruler: bool = False,
    *,
    overlay: bool = False,
) -> list[Path]:
    """One panorama per rig and band, or per rig with `overlay`, written beside the
    frames.
    """
    lay_out = overlaid if overlay else stitch
    groups: dict[tuple[str, str], list[CameraCalibration]] = {}
    for camera in Calibration.read(folder).cameras:
        # "overlay" is no band, so an overlay never takes a band panorama's file.
        key = (camera.rig, "overlay" if overlay else camera.band)
        groups.setdefault(key, []).append(camera)

    written = []
    for (rig, suffix), cameras in groups.items():
        names = [camera.name for camera in cameras]
        if len(set(names)) < len(names):
            raise ValueError(f"{rig} {suffix}: a sequence; a panorama takes stills")
        name = "_".join((rig, suffix, projection, frame))
        path = folder / f"panorama_{name}.png"
        try:
            image, layout = lay_out(folder, cameras, projection, frame, max_width)
        except cv2.error as error:
            raise RuntimeError(f"{path.name}: {error.err}") from error
        if ruler:
            # Bearings from world +Y read as true; every other frame's are relative.
            image = ruled(image, layout, true=frame == "world")
        if not cv2.imwrite(str(path), image):
            raise OSError(f"cannot write {path}")
        written.append(path)
    return written
