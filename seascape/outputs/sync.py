"""How far a recording's videos lag their IMU rows and each other, read off an ArUco
marker in each video.

The marker is static, so the rays to its corners turn exactly as the camera does.
Each video's turn over _BASELINE_FRAMES is matched to its IMU rows' at each trial lag
under one fitted orthogonal map, so no mounting, axis or sign convention enters.
"""

import math
from collections import Counter
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from pydantic import BaseModel, ValidationError

from seascape.config import ARUCO_DICTIONARY

# Judgement: over one frame, corner noise swamps the turn.
_BASELINE_FRAMES = 8
# Judgement: fewer pairs of turns fit noise.
_MIN_PAIRS = 3 * _BASELINE_FRAMES
# Judgements.
_MIN_PEAK = 0.8
_MIN_FOUND = 0.95
# Judgement: OpenCV's fps of one recording's videos differ by its rounding at most.
_FPS_REL_TOL = 1e-6


@dataclass(frozen=True, slots=True, kw_only=True)
class Lag:
    """How far one motion trails another, and how well they match, 0 to 1."""

    frames: float
    ms: float
    peak: float
    at_edge: bool  # the whole-frame search ended on its window's edge


@dataclass(frozen=True, slots=True, kw_only=True)
class VideoSync:
    """One video against its own IMU rows."""

    name: str
    frame_count: int
    row_count: int
    found_share: float  # of frames that find the marker
    lag: Lag | None  # positive: the rows log the motion late; None on too few frames


@dataclass(frozen=True, slots=True, kw_only=True)
class PairSync:
    """Two videos against each other."""

    first: str
    second: str
    lag: Lag | None  # positive: the second video shows the motion late


@dataclass(frozen=True, slots=True, kw_only=True)
class Report:
    """Every video of a recording, and every pair of them."""

    videos: tuple[VideoSync, ...]
    pairs: tuple[PairSync, ...]

    def failures(self, max_lag_ms: float) -> list[str]:
        """What breaks the limits, one line each."""
        broken = []
        for video in self.videos:
            if video.frame_count != video.row_count:
                broken.append(
                    f"{video.name}: {video.frame_count} frames, {video.row_count} rows"
                )
            if video.found_share < _MIN_FOUND:
                broken.append(
                    f"{video.name}: marker in {video.found_share:.0%} of frames"
                )
        lags = [(v.name, v.lag) for v in self.videos]
        lags += [(f"{p.first}~{p.second}", p.lag) for p in self.pairs]
        for name, lag in lags:
            if lag is None:
                broken.append(f"{name}: too few frames to match the motions")
            elif lag.peak < _MIN_PEAK:
                broken.append(f"{name}: motions match poorly, peak {lag.peak:.2f}")
            elif lag.at_edge:
                broken.append(
                    f"{name}: lag {lag.ms:+.1f} ms, on the search window's edge"
                )
            elif abs(lag.ms) > max_lag_ms:
                broken.append(f"{name}: lag {lag.ms:+.1f} ms")
        return broken


class _Camera(BaseModel):
    """A camera of `serialized_parameters.json`, the keys the check reads."""

    name: str
    id: int
    focal_length_x: float
    focal_length_y: float
    center_x: float
    center_y: float
    distortion_model: Literal["pinhole", "fisheye"]
    fisheye_distortion: tuple[float, float, float, float]


class _Parameters(BaseModel):
    cameras: list[_Camera]


@dataclass(frozen=True, slots=True, kw_only=True)
class _Video:
    name: str
    marker_id: int
    K: np.ndarray
    fisheye: np.ndarray | None  # OpenCV's k1..k4; None for a pinhole
    mp4: Path
    rows: Path


def _one(folder: Path, pattern: str) -> Path:
    found = sorted(folder.glob(pattern))
    if not found:
        raise FileNotFoundError(f"{folder}: no {pattern}")
    if len(found) > 1:
        raise ValueError(f"{folder}: {len(found)} files match {pattern}, want one")
    return found[0]


def _videos(folder: Path) -> list[_Video]:
    """Every video `serialized_parameters.json` lists, each watching the marker of
    its camera's `id`.
    """
    path = folder / "serialized_parameters.json"
    try:
        cameras = _Parameters.model_validate_json(path.read_text()).cameras
    except ValidationError as error:
        raise ValueError(f"{path}: {error}") from error
    if repeated := [n for n, k in Counter(c.name for c in cameras).items() if k > 1]:
        raise ValueError(f"{path}: cameras named {repeated} more than once")
    return [
        _Video(
            name=camera.name,
            marker_id=camera.id,
            K=np.array(
                [
                    [camera.focal_length_x, 0.0, camera.center_x],
                    [0.0, camera.focal_length_y, camera.center_y],
                    [0.0, 0.0, 1.0],
                ]
            ),
            fisheye=(
                np.array(camera.fisheye_distortion)
                if camera.distortion_model == "fisheye"
                else None
            ),
            mp4=_one(folder, f"*_{camera.name}_record_*.mp4"),
            rows=_one(folder, f"*_{camera.name}_metadata_*.csv"),
        )
        for camera in cameras
    ]


@contextmanager
def _opened(mp4: Path) -> Generator[cv2.VideoCapture]:
    capture = cv2.VideoCapture(str(mp4))
    try:
        if not capture.isOpened() or capture.get(cv2.CAP_PROP_FPS) <= 0:
            raise ValueError(f"{mp4}: OpenCV cannot read it")
        yield capture
    finally:
        capture.release()


def detector() -> cv2.aruco.ArucoDetector:
    """ARUCO_DICTIONARY, with corners refined to subpixel."""
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARY)
    return cv2.aruco.ArucoDetector(dictionary, parameters)


def corners(
    frame: np.ndarray, marker_id: int, find: cv2.aruco.ArucoDetector
) -> np.ndarray | None:
    """The marker's corners in pixels, (4, 2), clockwise from its top left; None if
    the frame does not show it.
    """
    grey = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    quads, ids, _ = find.detectMarkers(grey)
    if ids is None:
        return None
    for quad, found_id in zip(quads, ids.ravel(), strict=True):
        if found_id == marker_id:
            return quad.reshape(4, 2).astype(float)
    return None


def _rays(
    px: np.ndarray, intrinsics: np.ndarray, fisheye: np.ndarray | None
) -> np.ndarray:
    """Unit rays in the camera's frame, (n, 3), through pixels, (n, 2)."""
    points = px.reshape(-1, 1, 2)
    if fisheye is None:
        xy = cv2.undistortPoints(points, intrinsics, None)
    else:
        xy = cv2.fisheye.undistortPoints(points, intrinsics, fisheye)
    xyz = np.c_[xy.reshape(-1, 2), np.ones(len(px))]
    return xyz / np.linalg.norm(xyz, axis=1, keepdims=True)


def _kabsch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The rotation taking a's rows nearest b's, (n, 3) each."""
    u, _, vt = np.linalg.svd(b.T @ a)
    return u @ np.diag([1.0, 1.0, np.linalg.det(u @ vt)]) @ vt


def imu_rotation(roll_rad: float, pitch_rad: float, yaw_rad: float) -> np.ndarray:
    """Core-Backend's `rotation_matrix(..., roll_axis="x")`: Rz @ Ry @ Rx."""
    about_x, about_y, about_z = (
        cv2.Rodrigues(np.array(axis, float))[0]
        for axis in ([roll_rad, 0, 0], [0, pitch_rad, 0], [0, 0, yaw_rad])
    )
    return about_z @ about_y @ about_x


def _marker_rotations(
    video: _Video, capture: cv2.VideoCapture
) -> list[np.ndarray | None]:
    """Per frame, the camera's attitude in the axes of the first frame that found the
    marker; None where it is not found.
    """
    find = detector()
    seen: list[np.ndarray | None] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        px = corners(frame, video.marker_id, find)
        seen.append(None if px is None else _rays(px, video.K, video.fisheye))
    first = next((r for r in seen if r is not None), None)
    if first is None:
        return seen
    # A static marker's rays turn against the camera: from frame r back to first.
    return [None if r is None else _kabsch(r, first) for r in seen]


def _turns(rotations: Sequence[np.ndarray | None]) -> np.ndarray:
    """Rotation vectors from each frame to the one _BASELINE_FRAMES later; NaN where
    either frame is missing.
    """
    out = np.full((max(0, len(rotations) - _BASELINE_FRAMES), 3), np.nan)
    later = rotations[_BASELINE_FRAMES:]
    for i, (a, b) in enumerate(zip(rotations, later, strict=False)):
        if a is not None and b is not None:
            out[i] = cv2.Rodrigues(a.T @ b)[0].ravel()
    return out


def _correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    """How well one orthogonal map takes the rows of b, (n, 3), onto those of a. A
    reflection is allowed, so a flipped axis still scores 1.
    """
    both = ~(np.isnan(a).any(axis=1) | np.isnan(b).any(axis=1))
    if both.sum() < _MIN_PAIRS:
        return None
    a, b = a[both] - a[both].mean(axis=0), b[both] - b[both].mean(axis=0)
    singular = np.linalg.svd(a.T @ b, compute_uv=False)
    return float(singular.sum() / math.sqrt((a * a).sum() * (b * b).sum()))


def _later(turns: np.ndarray, frames: float, n: int) -> np.ndarray:
    """`n` rows of `turns`, read `frames` later, linear between rows; NaN past its
    ends.
    """
    index = np.arange(len(turns))
    at = np.arange(n) + frames
    return np.stack(
        [np.interp(at, index, axis, left=np.nan, right=np.nan) for axis in turns.T],
        axis=1,
    )


def _lag(a: np.ndarray, b: np.ndarray, window_frames: int, fps: float) -> Lag | None:
    """How far b's turns trail a's: whole frames within +-window_frames, then to a
    fraction of a frame by the parabola through the best and its two neighbours.
    None where no trial lag overlaps enough turns.
    """

    def score(frames: float) -> float:
        peak = _correlation(a, _later(b, frames, len(a)))
        return -1.0 if peak is None else peak

    if min(len(a), len(b)) < _MIN_PAIRS:
        return None
    coarse = max(range(-window_frames, window_frames + 1), key=score)
    left, mid, right = (score(coarse + step) for step in (-1, 0, 1))
    curve = left - 2 * mid + right
    best = coarse + (0.5 * (left - right) / curve if curve < 0 else 0.0)
    peak = _correlation(a, _later(b, best, len(a)))
    if peak is None:
        return None
    return Lag(
        frames=best,
        ms=1000.0 * best / fps,
        peak=peak,
        at_edge=abs(coarse) == window_frames,
    )


def _rows_deg(path: Path) -> np.ndarray:
    """Roll, pitch and yaw per row, (n, 3)."""
    rows = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))
    return np.column_stack([rows[c] for c in ("roll", "pitch", "yaw")])


def check(folder: Path, window_s: float) -> Report:
    """Each video's lag behind its IMU rows, and each pair's.

    Args:
        folder: A recording as a device writes it: `serialized_parameters.json`, and
            an mp4 and a metadata CSV per video.
        window_s: The largest lag searched, either way.
    """
    videos, motions = [], {}
    for video in _videos(folder):
        with _opened(video.mp4) as capture:
            fps = capture.get(cv2.CAP_PROP_FPS)
            rotations = _marker_rotations(video, capture)
        rows = _rows_deg(video.rows)
        logged = _turns([imu_rotation(*np.radians(row)) for row in rows])
        motions[video.name] = (_turns(rotations), fps)
        videos.append(
            VideoSync(
                name=video.name,
                frame_count=len(rotations),
                row_count=len(rows),
                found_share=sum(r is not None for r in rotations)
                / max(len(rotations), 1),
                lag=_lag(motions[video.name][0], logged, round(window_s * fps), fps),
            )
        )
    pairs = []
    for (a, (turns_a, fps)), (b, (turns_b, fps_b)) in combinations(motions.items(), 2):
        if not math.isclose(fps, fps_b, rel_tol=_FPS_REL_TOL):
            raise ValueError(f"{a} runs at {fps} fps and {b} at {fps_b}: no pair lag")
        # Limitation: frame i of one is frame i of the other; align by `timestp`
        # once a recording's videos can start apart.
        lag = _lag(turns_a, turns_b, round(window_s * fps), fps)
        pairs.append(PairSync(first=a, second=b, lag=lag))
    return Report(videos=tuple(videos), pairs=tuple(pairs))
