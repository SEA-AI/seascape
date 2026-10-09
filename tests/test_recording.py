"""A render laid out as a pod recording, from synthetic calibration and labels."""

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

from seascape import recording, waves
from seascape.calibration import Calibration, CameraCalibration
from seascape.labels import Image, Labels, horizon_px


def rotation(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """Body to frame: yaw clockwise about +Z, then pitch bow up, then roll starboard
    down."""
    r, p, y = np.radians([roll_deg, pitch_deg, -yaw_deg])
    about_z = np.array(
        [[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]]
    )
    about_x = np.array(
        [[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]]
    )
    about_y = np.array(
        [[np.cos(r), 0, np.sin(r)], [0, 1, 0], [-np.sin(r), 0, np.cos(r)]]
    )
    return about_z @ about_x @ about_y


def pose(r: np.ndarray, t: tuple[float, float, float] = (0, 0, 0)) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = r, t
    return m


def rows(m: np.ndarray) -> tuple:
    return tuple(tuple(float(v) for v in row) for row in m)


# roll, pitch, yaw per frame; a yaw a hair west of north must not wrap to 2 pi
HULL = [(2.0, 0.5, 0.0), (-1.5, 1.0, -1e-7)]
POD_YAW_DEG, CAMERA_YAW_DEG, CAMERA_PITCH_DEG = -60.0, 40.0, -10.0
WIDTH, HEIGHT, F = 640, 480, 1000.0
RADIUS_M = waves.earth_radius_m(0.13)


def make_run(into: Path, camera_yaw_deg: float) -> Path:
    """Stand-ins for a render: one camera in a pod on a rolling hull."""
    body_to_cv = pose(recording._BODY_TO_CV)
    camera_to_pod = pose(
        rotation(0, CAMERA_PITCH_DEG, camera_yaw_deg), (0.1, 0, 0)
    ) @ np.linalg.inv(body_to_cv)
    pod_to_vessel = pose(rotation(0, 0, POD_YAW_DEG), (-20.0, -60.0, 50.0))
    cameras, images = [], []
    for i, (roll, pitch, yaw) in enumerate(HULL):
        vessel_to_world = pose(rotation(roll, pitch, yaw), (0, 0, 0.2 * i))
        camera_to_vessel = pod_to_vessel @ camera_to_pod
        name = f"EO/{i:04d}.jpg"
        cameras.append(
            CameraCalibration(
                name="EO",
                band="eo",
                image=name,
                rig="port",
                model="Pod",
                width_px=WIDTH,
                height_px=HEIGHT,
                K=(
                    (F, 0.0, (WIDTH - 1) / 2),
                    (0.0, F, (HEIGHT - 1) / 2),
                    (0.0, 0.0, 1.0),
                ),
                extrinsics={
                    "world": rows(vessel_to_world @ camera_to_vessel),
                    "vessel": rows(camera_to_vessel),
                    "rig": rows(camera_to_pod),
                },
            )
        )
        images.append(
            Image(
                id=i + 1,
                file_name=name,
                width=WIDTH,
                height=HEIGHT,
                camera="EO",
                band="eo",
                time_s=i / 10,
                horizon_px=horizon_px(cameras[-1], RADIUS_M),
            )
        )
    Calibration(cameras=cameras[::-1]).write(into)
    scenario = {
        "rigs": {"port": {"height_m": 51.8}},
        "sky": {"t_air_k": 283.15},
        "sea": {"wind_speed_mps": 7.0, "wind_from_deg": -30.0},
    }
    info = {"date_created": "2026-09-29T03:55:21+00:00", "scenario": scenario}
    Labels(info=info, images=images).write(into)
    (into / "EO.mp4").write_bytes(b"not a video")
    return into


@pytest.fixture
def run(tmp_path: Path) -> Path:
    return make_run(tmp_path, CAMERA_YAW_DEG)


def test_a_rig_becomes_one_recording_folder_named_for_its_model(run: Path) -> None:
    into = run / "recordings/port/Pod_recordings_2026_09_29_03_55_21"
    assert sorted(recording.export(run)) == sorted(
        [
            into / "Pod_EO_record_2026_09_29_03_55_21.mp4",
            into / "Pod_EO_metadata_2026_09_29_03_55_21.csv",
            into / "user_settings.json",
            into / "POWER_Const/camera_parameters.json",
            into / "POWER_Const/pod_sensors.json",
        ]
    )
    assert (into / "Pod_EO_record_2026_09_29_03_55_21.mp4").samefile(run / "EO.mp4")


def test_each_row_is_its_frame_in_time_order(run: Path) -> None:
    recording.export(run)
    (table,) = run.glob("recordings/port/*/*.csv")
    with table.open() as f:
        table_rows = list(csv.DictReader(f))
    assert list(table_rows[0]) == recording.COLUMNS
    for i, (row, (roll, pitch, yaw)) in enumerate(zip(table_rows, HULL, strict=True)):
        assert int(row["counter"]) == i
        hull = [float(row[k]) for k in ("bb_roll", "bb_pitch", "bb_yaw")]
        assert hull == pytest.approx(np.radians([roll, pitch, yaw]), abs=1e-6)
        assert float(row["heading"]) == pytest.approx(math.radians(yaw), abs=1e-6)
        assert not any(value.startswith("-0.000000") for value in row.values())
        assert float(row["pan"]) == float(row["tilt"]) == 0.0
        assert float(row["yaw"]) == pytest.approx(POD_YAW_DEG, abs=0.05)
        assert float(row["temp"]) == pytest.approx(10.0)
        assert float(row["wind_angle"]) == pytest.approx(math.radians(330.0))


def test_pod_sensors_hold_the_camera_in_its_pod(run: Path) -> None:
    recording.export(run)
    (folder,) = run.glob("recordings/port/*/POWER_Const")
    sensor = json.loads((folder / "pod_sensors.json").read_text())["cameras"]["EO"]
    angles = [sensor[k]["value"] for k in ("roll", "pitch", "yaw")]
    # The recording's camera frame: pitch positive down, yaw positive to port.
    assert angles == pytest.approx([0.0, -CAMERA_PITCH_DEG, -CAMERA_YAW_DEG], abs=1e-9)
    assert sensor["position"]["value"] == pytest.approx([100.0, 0.0, 0.0])
    assert sensor["camera_type"] == "Blender EO"
    assert sensor["focal_length"]["value"] == pytest.approx(
        sensor["focal_length_x"]["value"] * sensor["pixel_size"]["value"]
    )
    parameters = json.loads((folder / "camera_parameters.json").read_text())["EO"]
    assert parameters == pytest.approx(
        {
            "roll": 0.0,
            "pitch": -CAMERA_PITCH_DEG,
            "yaw": -CAMERA_YAW_DEG,
            "barrel": 0.0,
        },
        abs=1e-9,
    )


def test_a_missing_video_writes_nothing(run: Path) -> None:
    (run / "EO.mp4").unlink()
    with pytest.raises(FileNotFoundError, match="seascape video"):
        recording.export(run)
    assert not (run / "recordings").exists()


def test_a_rig_without_a_model_writes_nothing(run: Path) -> None:
    calibration = Calibration.read(run)
    unmodelled = [c.model_copy(update={"model": None}) for c in calibration.cameras]
    Calibration(cameras=unmodelled).write(run)
    with pytest.raises(ValueError, match=r"\['port'\] have no model"):
        recording.export(run)
    assert not (run / "recordings").exists()


def _core_rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Core-Backend's `rotation_matrix(..., roll_axis="x")`: Rz @ Ry @ Rx, radians."""
    c, s = np.cos([roll, pitch, yaw]), np.sin([roll, pitch, yaw])
    about_x = np.array([[1, 0, 0], [0, c[0], -s[0]], [0, s[0], c[0]]])
    about_y = np.array([[c[1], 0, s[1]], [0, 1, 0], [-s[1], 0, c[1]]])
    about_z = np.array([[c[2], -s[2], 0], [s[2], c[2], 0], [0, 0, 1]])
    return about_z @ about_y @ about_x


def _core_imu_horizon(
    row: dict, sensor: dict, x: np.ndarray, dip: float = 0.0
) -> np.ndarray:
    """Core-Backend's IMU horizon, ported: the CSV's roll and pitch with its yaw
    zeroed, turned by the camera's rpy, and projected. Core leaves out the horizon's
    `dip`, which reads as the camera pitched up by it."""
    imu = _core_rotation(
        math.radians(float(row["roll"])), math.radians(float(row["pitch"])), 0.0
    )
    camera = _core_rotation(
        *np.radians([sensor[k]["value"] for k in ("roll", "pitch", "yaw")])
    )
    m = camera @ imu
    psi = math.atan2(m[2, 1], m[2, 2])
    theta = math.atan2(-m[2, 0], math.hypot(m[0, 0], m[1, 0])) - dip
    fy, cx, cy = (
        sensor[k]["value"] for k in ("focal_length_y", "center_x", "center_y")
    )
    return cy - math.tan(psi) * (x - cx) - fy * math.tan(theta) / math.cos(psi)


def test_core_draws_the_imu_horizon_on_the_rendered_one(tmp_path: Path) -> None:
    """Boresight only: Core turns the IMU by the camera in the order that holds for a
    small angle, and a camera yawed in its pod is not one."""
    run = make_run(tmp_path, camera_yaw_deg=0.0)
    recording.export(run)
    (folder,) = run.glob("recordings/port/*")
    sensor = json.loads((folder / "POWER_Const/pod_sensors.json").read_text())
    with next(folder.glob("*.csv")).open() as f:
        table_rows = list(csv.DictReader(f))
    images = sorted(
        Labels.model_validate_json((run / "labels.json").read_text()).images,
        key=lambda image: image.time_s,
    )
    cameras = sorted(Calibration.read(run).cameras, key=lambda c: c.image)
    for row, image, camera in zip(table_rows, images, cameras, strict=True):
        # labels.json puts pixel centres at +0.5, calibration.json at integers.
        x, y = (np.array(image.horizon_px) - 0.5).T
        height_m = camera.extrinsics["world"][2][3]
        dip = math.acos(RADIUS_M / (RADIUS_M + height_m))
        drawn = _core_imu_horizon(row, sensor["cameras"]["EO"], x, dip)
        # Within a pixel: Core draws a chord the horizon bows off, and turns the IMU by
        # the camera in the order right for small angles. A wrong sign tilts the line
        # or moves it by twice a pitch.
        assert drawn == pytest.approx(y, abs=1.0)


def test_a_frame_labels_json_does_not_list_writes_nothing(run: Path) -> None:
    truth = Labels.model_validate_json((run / "labels.json").read_text())
    Labels(info=truth.info, images=truth.images[:1]).write(run)
    with pytest.raises(ValueError, match=r"EO/0001\.jpg"):
        recording.export(run)
    assert not (run / "recordings").exists()
