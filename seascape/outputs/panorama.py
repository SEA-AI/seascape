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

from seascape import angles
from seascape.config import BANDS, Agc
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


def _rotation(camera: CameraCalibration, frame: Frame) -> np.ndarray:
    return np.array(camera.extrinsics[frame])[:3, :3]


def axis_deg(cameras: list[CameraCalibration], frame: Frame) -> float:
    """Azimuth in `frame` halfway between the cameras' outermost rays, so the
    panorama is centred on what they see. Offsets are taken from the summed optical
    axes, so the span cannot wrap while the cameras cover less than 360 deg.
    """
    x, y, _ = np.sum([_rotation(camera, frame)[:, 2] for camera in cameras], axis=0)
    summed_deg = angles.bearing_deg(float(x), float(y))
    R_summed_frame = angles.turned(summed_deg, 0.0).T
    right, ahead, _ = R_summed_frame @ np.hstack([_rays(c, frame) for c in cameras])
    offsets_deg = angles.bearing_deg(right, ahead)
    middle_deg = (offsets_deg.min() + offsets_deg.max()) / 2
    return angles.wrapped_deg(summed_deg + middle_deg)


def _border(camera: CameraCalibration, n: int = 32) -> np.ndarray:
    """(4n, 2) pixels around a frame's edge, clockwise from its top-left corner."""
    w, h = camera.width_px - 1, camera.height_px - 1
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    zero, one = np.zeros(n), np.ones(n)
    return np.concatenate(
        [
            np.c_[t * w, zero],
            np.c_[one * w, t * h],
            np.c_[(1 - t) * w, one * h],
            np.c_[zero, (1 - t) * h],
        ]
    )


def _rays(camera: CameraCalibration, frame: Frame) -> np.ndarray:
    """(3, 4n) rays through a pinhole's `_border`, in `frame`."""
    border = _border(camera)
    pixels = np.c_[border, np.ones(len(border))].T
    return _rotation(camera, frame) @ np.linalg.inv(np.array(camera.K)) @ pixels


def pose(
    camera: CameraCalibration, frame: Frame, bearing_deg: float
) -> tuple[np.ndarray, np.ndarray]:
    """K and R as the stitcher takes them, turned so `bearing_deg` is straight ahead."""
    if camera.distortion is not None:
        raise ValueError(f"{camera.name}: a panorama stitches pinhole cameras")
    # The panorama's frame: +Y along the bearing.
    R_panorama_frame = angles.turned(bearing_deg, 0.0).T
    return (
        np.array(camera.K, np.float32),
        np.array(
            angles.TO_CV @ R_panorama_frame @ _rotation(camera, frame), np.float32
        ),
    )


class Layout(NamedTuple):
    """Where a stitched panorama sits in its projection."""

    kind: str
    scale: float
    axis_deg: float  # the bearing the panorama is turned to, in its frame
    left: int  # projection x of column 0
    top: int  # projection y of row 0
    resize: float  # the final shrink to max_width, 1.0 when none


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
    off_deg = bearing_deg - layout.axis_deg
    if layout.kind == "plane" and math.cos(math.radians(off_deg)) <= 0.0:
        return None
    # The same turn, in the stitcher's camera axes.
    turn = angles.TO_CV @ angles.turned(off_deg, 0.0) @ angles.TO_CV.T
    r = np.array(turn, np.float32)
    u, _ = _warper(layout.kind, layout.scale).warpPoint(
        (0.0, 0.0), np.eye(3, dtype=np.float32), r
    )
    # Pixel centres sit at integers, so the resize scales about -0.5.
    return (u - layout.left + 0.5) * layout.resize - 0.5


@cache
def _warper(kind: str, scale: float) -> cv2.PyRotationWarper:
    return cv2.PyRotationWarper(kind, scale)


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
    tick = round(8 * size)
    font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.5 * size
    line = max(1, round(size))
    for bearing in range(-180, 180, TICK_DEG):
        x = column(layout, bearing)
        if x is None or not 0 <= x < w:
            continue
        x = round(x)
        labelled = bearing % LABEL_DEG == 0
        cv2.line(strip, (x, 0), (x, tick * (2 if labelled else 1)), canvas.INK, line)
        if labelled:
            if true:
                text = f"{bearing % 360:03d}"
            else:
                text = f"{bearing:+d}" if bearing else "0"
            (tw, th), _ = cv2.getTextSize(text, font, scale, line)
            origin = (x - tw // 2, 2 * tick + th + round(4 * size))
            cv2.putText(strip, text, origin, font, scale, canvas.INK, line, cv2.LINE_AA)
    return np.vstack([image, strip])


def _curve(
    folder: Path, cameras: list[CameraCalibration], temps: list[np.ndarray | None]
) -> tuple[np.ndarray, Agc] | None:
    """One transfer curve for every thermal frame, so a temperature is one grey on both
    sides of a seam.
    """
    known = [t_k for t_k in temps if t_k is not None]
    if not known:
        return None
    every = agcs(folder)
    setting, *others = [every[camera.name] for camera in cameras]
    if any(other != setting for other in others):
        raise ValueError("a panorama's cameras need one agc, to share one curve")
    return agc.transfer(known, setting), setting


def stitch(
    folder: Path,
    cameras: list[CameraCalibration],
    projection: str,
    frame: Frame,
    max_width: int | None = None,
    *,
    among: list[CameraCalibration] | None = None,
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
            resolution, the largest fx: pixels per radian on axis, where every
            projection here runs at one unit per radian.
        among: The cameras whose axis and extent lay the panorama out, so panoramas
            of several groups of them come out on one canvas; None is `cameras`.
    """
    every = cameras if among is None else among
    if max_width is not None and max_width < 1:
        raise ValueError(f"max width is {max_width}: it must be a pixel or more")
    if missing := [c.name for c in every if frame not in c.extrinsics]:
        raise ValueError(f"no {frame!r} extrinsics for {', '.join(missing)}")
    kind = PROJECTIONS[projection]
    bearing_deg = axis_deg(every, frame)
    poses = [pose(camera, frame, bearing_deg) for camera in every]
    sizes = [(camera.width_px, camera.height_px) for camera in every]
    if kind == "plane" and not all(
        _faces_ahead(k, r, size) for (k, r), size in zip(poses, sizes, strict=True)
    ):
        raise ValueError(
            "rectilinear cannot show a ray 90 deg or more off its axis: "
            "use cylindrical or equirectangular"
        )

    scale = max(float(k[0, 0]) for k, _ in poses)  # every camera at native size
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
    # Each frame shrunk to the scale before warping; the warp covers what warpRoi
    # gives for that size and K, so `every`'s rois are the canvas.
    shrunk = {
        camera.name: _shrunk(size, k, scale / float(k[0, 0]))
        for camera, size, (k, _) in zip(every, sizes, poses, strict=True)
    }
    rois = [
        warper.warpRoi(*shrunk[camera.name], r)
        for camera, (_, r) in zip(every, poses, strict=True)
    ]
    roi = cv2.detail.resultRoi(
        corners=[roi[:2] for roi in rois], sizes=[roi[2:] for roi in rois]
    )

    temps = [agc.kelvin(folder / camera.image) for camera in cameras]
    shared = _curve(folder, cameras, temps)
    warped = []
    for camera, t_k in zip(cameras, temps, strict=True):
        path = folder / camera.image
        image = (
            cv2.imread(str(path), cv2.IMREAD_COLOR)
            if t_k is None or shared is None
            else cv2.cvtColor(agc.grey(t_k, *shared), cv2.COLOR_GRAY2BGR)
        )
        if image is None:
            raise ValueError(
                f"cannot read {path}: panorama takes the frames that "
                'outputs.format = "png" or "jpg" writes'
            )
        size, k = shrunk[camera.name]
        _, r = pose(camera, frame, bearing_deg)
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        corner, pixels = warper.warp(image, k, r, cv2.INTER_LINEAR, cv2.BORDER_REFLECT)
        mask = np.full(image.shape[:2], 255, np.uint8)
        _, mask = warper.warp(mask, k, r, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)
        warped.append((corner, pixels, mask))

    blender = cv2.detail.MultiBandBlender()
    blender.prepare(roi)
    for corner, pixels, mask in warped:
        blender.feed(pixels.astype(np.int16), mask, corner)
    image, _ = blender.blend(np.empty(0, np.int16), np.empty(0, np.uint8))
    # The pyramid overshoots at edges; clip, where convertScaleAbs would fold it back.
    image = np.clip(image, 0, 255).astype(np.uint8)
    image, resize = _fitted(image, max_width)
    return image, Layout(kind, scale, bearing_deg, roi[0], roi[1], resize)


def _fitted(image: np.ndarray, max_width: int | None) -> tuple[np.ndarray, float]:
    """`image` shrunk to `max_width`, and the factor. The warp is scaled to
    `max_width` already, but the warper rounds its extent a pixel outward.
    """
    h, w = image.shape[:2]
    if max_width is None or w <= max_width:
        return image, 1.0
    resize = max_width / w
    size = (max_width, round(h * resize))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA), resize


def overlaid(
    folder: Path,
    cameras: list[CameraCalibration],
    projection: str,
    frame: Frame,
    max_width: int | None = None,
) -> tuple[np.ndarray, Layout]:
    """Every band of `cameras` in one panorama, and where it sits in its projection.

    Each band is stitched on one canvas and laid in `BANDS`' order: at `OPACITY`
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
    groups = [g for band in BANDS if (g := [c for c in cameras if c.band == band])]
    stitched = [
        stitch(folder, group, projection, frame, max_width, among=cameras)
        for group in groups
    ]
    # One canvas: every band's layout is the first's.
    layers = [pixels for pixels, _ in stitched]
    _, layout = stitched[0]
    warper = _warper(layout.kind, layout.scale)

    def landed(camera: CameraCalibration, pixels: np.ndarray) -> np.ndarray:
        """(N, 2) pixels of `camera`'s frame, where they land on the overlay."""
        k, r = pose(camera, frame, layout.axis_deg)
        uv = np.array([warper.warpPoint(p, k, r) for p in pixels])
        # Pixel centres sit at integers, so the resize scales about -0.5.
        return (uv - (layout.left, layout.top) + 0.5) * layout.resize - 0.5

    image = np.full_like(layers[0], canvas.MATTE)
    covered = np.zeros(image.shape[:2], bool)
    outlines = []
    for group, layer in zip(groups, layers, strict=True):
        polygons = [landed(c, _border(c)).round().astype(np.int32) for c in group]
        footprint = np.zeros(image.shape[:2], np.uint8)
        for polygon in polygons:
            # One at a time: fillPoly leaves where two polygons overlap unfilled.
            cv2.fillPoly(footprint, [polygon], 1)
        seen = footprint > 0
        mixed = cv2.addWeighted(layer, OPACITY, image, 1 - OPACITY, 0)
        image[seen] = np.where(covered[seen, None], mixed[seen], layer[seen])
        covered |= seen
        outlines.append(polygons)

    size = _ink(image.shape[1])
    line, gap = max(1, round(2 * size)), 10 * size
    font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.8 * size
    # The palette is RGB; cv2 draws BGR.
    colours = [canvas.palette(len(BANDS))[BANDS.index(g[0].band)][::-1] for g in groups]
    # Every edge before any name, so no later edge strikes through one.
    for polygons, colour in zip(outlines, colours, strict=True):
        cv2.polylines(image, polygons, True, colour.tolist(), line, cv2.LINE_AA)
    for i, (group, colour) in enumerate(zip(groups, colours, strict=True)):
        for camera in group:
            edge = (camera.K[0][2], camera.height_px - 1.0 if i else 0.0)
            ((x, y),) = landed(camera, np.array([edge]))
            (tw, th), _ = cv2.getTextSize(camera.name, font, scale, line)
            y += -gap if i else th + gap
            origin = (round(x - tw / 2), round(y))
            cv2.putText(
                image,
                camera.name,
                origin,
                font,
                scale,
                colour.tolist(),
                line,
                cv2.LINE_AA,
            )
    return image, layout


def _faces_ahead(k: np.ndarray, r: np.ndarray, size: tuple[int, int]) -> bool:
    """Whether a whole frame lies less than 90 deg off the stitcher's +Z. Depth is
    linear across the frame, so the corners decide.
    """
    w, h = size
    corners = np.array([[0, 0, 1], [w - 1, 0, 1], [0, h - 1, 1], [w - 1, h - 1, 1]])
    _, _, z = r @ np.linalg.inv(k) @ corners.T
    return bool((z > 0).all())


def _shrunk(
    size: tuple[int, int], k: np.ndarray, factor: float
) -> tuple[tuple[int, int], np.ndarray]:
    """The size and K of a frame area-averaged down before warping: the warper
    samples it bilinearly, which aliases anything finer than the output grid.
    """
    if factor >= 1.0:
        return size, k
    w, h = size
    small = (round(w * factor), round(h * factor))
    sx, sy = small[0] / w, small[1] / h
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
