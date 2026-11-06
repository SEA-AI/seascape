import csv
import json
import math
import shutil
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest
from click.testing import CliRunner

from seascape import cli
from seascape.config import ARUCO_DICTIONARY
from seascape.outputs import sync
from seascape.outputs.calibration import TO_CV
from seascape.outputs.recording import COLUMNS, attitude_deg

FPS, FRAMES = 30, 300
WIDTH, HEIGHT, HFOV_DEG = 320, 256, 20.0
MARKER_M, RANGE_M, EYE_M = 2.0, 30.0, 2.5
# Incommensurate sines, within the camera's view.
PITCH = [(0.8, 5.3), (0.5, 2.3), (0.3, 0.97)]
ROLL = [(1.5, 7.1), (0.8, 3.1), (0.4, 1.31)]
# A projected marker's corners carry a fraction of a pixel of noise.
FRAMES_OFF = 0.1


def _deg(waves: list[tuple[float, float]], t_s: np.ndarray) -> np.ndarray:
    return sum(a * np.sin(2 * np.pi * t_s / p) for a, p in waves)


def _body_to_world(t_s: float) -> np.ndarray:
    """Body to world, x starboard, y bow, z up: pitch bow up about x after roll
    starboard down about y, as `vessels.add_ownship` turns the ownship.
    """
    p, r = (math.radians(float(_deg(w, np.array(t_s)))) for w in (PITCH, ROLL))
    about_x = np.array(
        [[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]]
    )
    about_y = np.array(
        [[math.cos(r), 0, math.sin(r)], [0, 1, 0], [-math.sin(r), 0, math.cos(r)]]
    )
    return about_x @ about_y


F_PX = (WIDTH / 2) / math.tan(math.radians(HFOV_DEG) / 2)
K = np.array([[F_PX, 0, (WIDTH - 1) / 2], [0, F_PX, (HEIGHT - 1) / 2], [0, 0, 1]])


def _frame(t_s: float, tile: np.ndarray) -> np.ndarray:
    """The marker and a one-module white margin, at the camera's height ahead."""
    body = _body_to_world(t_s)
    eye = body @ np.array([0.0, 0.0, EYE_M])
    half = MARKER_M / 2 * 8 / 6  # the margin's corners
    world = np.array(
        [
            [x, RANGE_M, EYE_M + z]
            for x, z in ((-half, half), (half, half), (half, -half), (-half, -half))
        ]
    )
    camera = (body @ TO_CV.T).T @ (world - eye).T
    px = (K @ (camera / camera[2])).T[:, :2]
    size = tile.shape[0]
    square = np.array([[0, 0], [size, 0], [size, size], [0, size]], np.float32)
    warp = cv2.getPerspectiveTransform(square, px.astype(np.float32))
    frame = cv2.warpPerspective(tile, warp, (WIDTH, HEIGHT), borderValue=90)
    return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)


def write_recording(
    folder: Path, *, late_rows: int = 0, late_frames: int = 0, hidden: int = 0
) -> Path:
    """Two videos, A and B, of one motion. B's frames show it `late_frames` late, A's
    rows log it `late_rows` late, and A's first `hidden` frames miss the marker.
    """
    folder.mkdir()
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARY)
    times = np.arange(FRAMES) / FPS
    cameras = []
    for i, name in enumerate("AB"):
        marker = cv2.aruco.generateImageMarker(dictionary, i, 60)
        tile = cv2.copyMakeBorder(
            marker, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255
        )
        late = late_frames if name == "B" else 0
        video = cv2.VideoWriter(
            str(folder / f"Test_{name}_record_x.mp4"),
            cv2.VideoWriter.fourcc(*"mp4v"),
            FPS,
            (WIDTH, HEIGHT),
        )
        for f, t_s in enumerate(times.tolist()):
            frame = _frame(t_s - late / FPS, tile)
            if name == "A" and f < hidden:
                frame[:] = 90
            video.write(frame)
        video.release()
        shift = late_rows if name == "A" else 0
        with (folder / f"Test_{name}_metadata_x.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for counter, t_s in enumerate((times - shift / FPS).tolist()):
                pose = np.eye(4)
                pose[:3, :3] = _body_to_world(t_s)
                roll, pitch, yaw = attitude_deg(pose)
                writer.writerow(
                    dict.fromkeys(COLUMNS, 0)
                    | {"counter": counter, "roll": roll, "pitch": -pitch, "yaw": yaw}
                )
        cameras.append(
            {
                "name": name,
                "id": i,
                "focal_length_x": F_PX,
                "focal_length_y": F_PX,
                "center_x": K[0, 2],
                "center_y": K[1, 2],
                "distortion_model": "pinhole",
                "fisheye_distortion": [0.0] * 4,
            }
        )
    (folder / "serialized_parameters.json").write_text(json.dumps({"cameras": cameras}))
    return folder


@pytest.fixture(scope="module")
def in_step_folder(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_recording(tmp_path_factory.mktemp("s") / "r")


@pytest.fixture(scope="module")
def in_step(in_step_folder: Path) -> sync.Report:
    return sync.check(in_step_folder, 1.0)


def _video(report: sync.Report, name: str) -> sync.VideoSync:
    return next(v for v in report.videos if v.name == name)


def _lag(read: sync.VideoSync | sync.PairSync) -> sync.Lag:
    assert read.lag is not None
    return read.lag


def test_a_recording_in_step_reads_no_lag(in_step: sync.Report) -> None:
    lags = [_lag(v).frames for v in in_step.videos] + [
        _lag(p).frames for p in in_step.pairs
    ]
    assert lags == pytest.approx([0.0] * 3, abs=FRAMES_OFF)
    assert in_step.failures(max_lag_ms=33.0) == []


@pytest.mark.parametrize("late", [4, -9])
def test_rows_logged_late_read_as_that_lag(tmp_path: Path, late: int) -> None:
    report = sync.check(write_recording(tmp_path / "r", late_rows=late), 1.0)
    a = _lag(_video(report, "A"))
    assert a.frames == pytest.approx(late, abs=FRAMES_OFF)
    assert _lag(_video(report, "B")).frames == pytest.approx(0.0, abs=FRAMES_OFF)
    assert report.failures(max_lag_ms=33.0) == [f"A: lag {a.ms:+.1f} ms"]


def test_a_video_late_against_another_reads_their_lag(tmp_path: Path) -> None:
    report = sync.check(write_recording(tmp_path / "r", late_frames=3), 1.0)
    # B's pixels are late against its own rows, which are on time.
    assert _lag(_video(report, "B")).frames == pytest.approx(-3.0, abs=FRAMES_OFF)
    (pair,) = report.pairs
    assert (pair.first, pair.second) == ("A", "B")
    assert _lag(pair).frames == pytest.approx(3.0, abs=FRAMES_OFF)


def test_a_marker_out_of_sight_fails_the_check(tmp_path: Path) -> None:
    report = sync.check(write_recording(tmp_path / "r", hidden=30), 1.0)
    assert _video(report, "A").found_share == pytest.approx(0.9)
    assert "A: marker in 90% of frames" in report.failures(33.0)


def test_frames_without_rows_fail_the_check(in_step: sync.Report) -> None:
    a = replace(_video(in_step, "A"), row_count=FRAMES - 3)
    report = replace(in_step, videos=(a, *in_step.videos[1:]))
    assert report.failures(33.0) == [f"A: {FRAMES} frames, {FRAMES - 3} rows"]


def test_the_command_prints_the_report_and_fails_past_the_limits(
    in_step_folder: Path,
) -> None:
    run = CliRunner().invoke(cli.main, ["sync", str(in_step_folder)])
    assert run.exit_code == 0, run.output
    assert [v["name"] for v in json.loads(run.stdout)["videos"]] == ["A", "B"]
    strict = CliRunner().invoke(
        cli.main, ["sync", str(in_step_folder), "--max-lag-ms", "0"]
    )
    assert strict.exit_code == 1
    assert json.loads(strict.stdout)["failures"]


def test_a_lag_on_the_window_s_edge_fails_the_check(tmp_path: Path) -> None:
    report = sync.check(write_recording(tmp_path / "r", late_rows=4), 2 / FPS)
    a = _lag(_video(report, "A"))
    assert a.at_edge
    edge = f"A: lag {a.ms:+.1f} ms, on the search window's edge"
    assert edge in report.failures(1000.0)


def test_too_few_turns_read_no_lag_and_fail_the_check(in_step: sync.Report) -> None:
    turns = np.zeros((5, 3))
    assert sync._lag(turns, turns, 3, FPS) is None
    a = replace(_video(in_step, "A"), lag=None)
    report = replace(in_step, videos=(a, *in_step.videos[1:]))
    assert report.failures(33.0) == ["A: too few frames to match the motions"]


def test_motions_that_match_poorly_fail_the_check(in_step: sync.Report) -> None:
    a = _video(in_step, "A")
    a = replace(a, lag=replace(_lag(a), peak=0.5))
    report = replace(in_step, videos=(a, *in_step.videos[1:]))
    assert report.failures(33.0) == ["A: motions match poorly, peak 0.50"]


@pytest.mark.parametrize(
    "edit",
    [
        lambda cameras: cameras[0].pop("id"),
        lambda cameras: cameras[1].update(name=cameras[0]["name"]),
    ],
    ids=["missing-key", "repeated-name"],
)
def test_a_bad_parameters_file_is_named(
    in_step_folder: Path, tmp_path: Path, edit: Callable[[list[dict]], object]
) -> None:
    folder = tmp_path / "r"
    shutil.copytree(in_step_folder, folder)
    path = folder / "serialized_parameters.json"
    parameters = json.loads(path.read_text())
    edit(parameters["cameras"])
    path.write_text(json.dumps(parameters))
    with pytest.raises(ValueError, match=r"serialized_parameters\.json"):
        sync.check(folder, 1.0)


def test_rows_without_an_attitude_column_fail(
    in_step_folder: Path, tmp_path: Path
) -> None:
    folder = tmp_path / "r"
    shutil.copytree(in_step_folder, folder)
    (rows,) = folder.glob("*_A_metadata_*.csv")
    rows.write_text("counter,roll,pitch\n0,0,0\n")
    with pytest.raises(ValueError, match="no field of name yaw"):
        sync.check(folder, 1.0)


def test_fisheye_pixels_unproject_to_their_rays() -> None:
    rays = np.array([[0.0, 0.0, 1.0], [0.3, -0.2, 1.0], [-0.6, 0.4, 1.0]])
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    distortion = np.array([0.05, -0.01, 0.002, 0.0])
    px, _ = cv2.fisheye.projectPoints(
        rays.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, distortion
    )
    assert sync._rays(px.reshape(-1, 2), K, distortion) == pytest.approx(rays, abs=1e-6)
