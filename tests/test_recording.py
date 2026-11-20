"""A render laid out as a pod recording, from synthetic calibration and labels."""

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

from seascape import angles
from seascape.config import Crop, Ownship, Stitch, Undistortion
from seascape.outputs import recording
from seascape.outputs.calibration import Calibration, CameraCalibration
from seascape.outputs.labels import Image, Labels, horizon_px
from seascape.physics import earth


def rotation(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """Body to frame: yaw clockwise about +Z, then pitch bow up, then roll starboard
    down.
    """
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
PTU_TILT_DEG, SCAN_DEG_S, TOP_DEG_S = 4.0, 6.0, 50.0
# A head's mount, under its pan and tilt, and the hull's heading while it scans.
MOUNT_YAW_DEG, MOUNT_PITCH_DEG, SCANNING_HULL_YAW_DEG = -20.0, -2.0, 15.0
WIDTH, HEIGHT, F = 640, 480, 1000.0
RADIUS_M = earth.earth_radius_m(0.13)
GOLDEN = Path(__file__).with_name("recording_golden.json")


def make_run(
    into: Path,
    camera_yaw_deg: float,
    *,
    ptu: bool = False,
    scan_deg_s: float | None = SCAN_DEG_S,
    turned: str | None = "port",
) -> Path:
    """Stand-ins for a render: one camera in a pod on a rolling hull; with `ptu`,
    panned, tilted and in a turn of the rig `turned`, else of the ownship, at
    `scan_deg_s`.
    """
    body_to_cv = pose(angles.TO_CV)
    camera_to_pod = pose(
        rotation(0, CAMERA_PITCH_DEG, camera_yaw_deg), (0.1, 0, 0)
    ) @ np.linalg.inv(body_to_cv)
    pod_pitch_deg = PTU_TILT_DEG if ptu else 0.0
    pod_to_vessel = pose(rotation(0, pod_pitch_deg, POD_YAW_DEG), (-20.0, -60.0, 50.0))
    cameras, images = [], []
    for i, (roll, pitch, yaw) in enumerate(HULL):
        heading = yaw + (SCANNING_HULL_YAW_DEG if ptu else 0.0)
        vessel_to_world = pose(rotation(roll, pitch, heading), (0, 0, 0.2 * i))
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
                    "ownship": rows(camera_to_vessel),
                    "head": rows(camera_to_pod),
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
                hfov_deg=45.0,
                time_s=i / 10,
                horizon_px=horizon_px(cameras[-1], RADIUS_M),
            )
        )
    Calibration(cameras=cameras[::-1]).write(into)
    scenario = {
        "ownship": {},
        "rigs": {"port": {"height_m": 51.8, "videos": {}}},
        "sky": {"t_air_k": 283.15},
        "sea": {"wind_speed_mps": 7.0, "wind_from_deg": -30.0},
    }
    if ptu:
        scenario["rigs"]["port"] |= {
            "ptu": {},
            "yaw_deg": MOUNT_YAW_DEG,
            "pitch_deg": MOUNT_PITCH_DEG,
        }
        scenario["turn"] = {
            "ptu": turned,
            "top_rate_deg_s": TOP_DEG_S,
            "moves": [{"to_deg": 60.0, "rate_deg_s": scan_deg_s, "hold_s": 0.0}],
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
            into / "serialized_parameters.json",
        ]
    )
    assert (into / "Pod_EO_record_2026_09_29_03_55_21.mp4").samefile(run / "EO.mp4")


def test_a_head_on_a_pan_tilt_unit_records_no_scan_while_the_ownship_turns(
    tmp_path: Path,
) -> None:
    recording.export(make_run(tmp_path, CAMERA_YAW_DEG, ptu=True, turned=None))
    (table,) = tmp_path.glob("recordings/port/*/*.csv")
    with table.open() as f:
        table_rows = list(csv.DictReader(f))

    for row in table_rows:
        # Oceanus's manual mode, at no scan speed.
        assert float(row["proc_ptu_mode"]) == 1
        assert float(row["proc_colav_speed"]) == 0.0


def test_a_head_scanning_at_no_set_rate_records_the_turn_s_top_rate(
    tmp_path: Path,
) -> None:
    recording.export(make_run(tmp_path, CAMERA_YAW_DEG, ptu=True, scan_deg_s=None))
    (table,) = tmp_path.glob("recordings/port/*/*.csv")
    with table.open() as f:
        speeds = [float(row["proc_colav_speed"]) for row in csv.DictReader(f)]

    # Revolutions per second.
    assert speeds == pytest.approx([TOP_DEG_S / 360.0] * len(HULL), abs=1e-6)


def test_a_camera_s_own_pitch_sizes_its_pixels(run: Path) -> None:
    eo = Calibration.read(run).cameras[0]
    ir = eo.model_copy(update={"name": "IR", "band": "ir", "pitch_um": 17.0})
    sensor = recording._sensor(ir)
    assert sensor["pixel_size"]["value"] == pytest.approx(0.017)
    assert sensor["focal_length"]["value"] == pytest.approx(
        sensor["focal_length_x"]["value"] * 0.017
    )


def test_a_camera_with_an_unrendered_video_is_recorded_as_it_is(run: Path) -> None:
    """As in an exr, which renders no video."""
    truth = Labels.model_validate_json((run / "labels.json").read_text())
    video = {"kind": "crop", "camera": "EO", "left_px": 40}
    truth.info["scenario"]["rigs"]["port"]["videos"] = {"RGB": video}
    truth.write(run)

    recording.export(run)

    (folder,) = run.glob("recordings/port/*")
    serialized = json.loads((folder / "serialized_parameters.json").read_text())
    (entry,) = serialized["cameras"]
    assert entry["name"] == "EO"
    assert entry["transformation_type"] == "identity"
    assert entry["transformation_parameters"] is None
    (source,) = entry["camera_sources"]
    assert (source["name"], source["device_name"]) == ("EO", "EO")


def test_an_undistorted_video_is_recorded_with_its_camera_as_rendered(
    run: Path,
) -> None:
    video = Undistortion(
        kind="undistortion", camera="EO", hfov_deg=130.0, vfov_deg=65.5, downscale=2
    )
    calibration = Calibration.read(run)
    # A wider pinhole than the camera it dewarps.
    recorded = [
        camera.model_copy(
            update={
                "name": "port_RGB",
                "K": tuple(map(tuple, camera.K @ np.diag([0.5, 0.5, 1.0]))),
            }
        )
        for camera in calibration.cameras
    ]
    Calibration(cameras=recorded, raw=calibration.cameras).write(run)
    truth = Labels.model_validate_json((run / "labels.json").read_text())
    truth.info["scenario"]["rigs"]["port"]["videos"] = {"RGB": video.model_dump()}
    truth.write(run)
    (run / "EO.mp4").rename(run / "port_RGB.mp4")

    recording.export(run)

    (folder,) = run.glob("recordings/port/*")
    assert (folder / "Pod_RGB_record_2026_09_29_03_55_21.mp4").exists()
    sensors = json.loads((folder / "POWER_Const/pod_sensors.json").read_text())
    (sensor,) = sensors["cameras"].values()
    assert sensor["resolution"]["value"] == [WIDTH, HEIGHT]
    serialized = json.loads((folder / "serialized_parameters.json").read_text())
    (entry,) = serialized["cameras"]
    assert entry["resolution"] == [WIDTH, HEIGHT]
    assert entry["center_x"] == pytest.approx((WIDTH - 1) / 2)
    assert entry["transformation_type"] == "undistortion"
    assert entry["transformation_parameters"] == {
        "downscale": 2,
        "out-fov": {"value": [130.0, 65.5], "unit": "degrees"},
    }
    assert entry["focal_length"] == sensor["focal_length"]["value"]
    (source,) = entry["camera_sources"]
    assert (source["name"], source["device_name"]) == ("EO", "RGB")
    assert source["center_x"] == sensor["center_x"]["value"]


def test_a_stitch_is_recorded_with_both_its_cameras(run: Path) -> None:
    left = Calibration.read(run).cameras
    right = [camera.model_copy(update={"name": "IR"}) for camera in left]
    stitched = [camera.model_copy(update={"name": "port_T1"}) for camera in left]
    Calibration(cameras=stitched, raw=left + right).write(run)
    truth = Labels.model_validate_json((run / "labels.json").read_text())
    video = Stitch(kind="horizontal_stitching", cameras=("EO", "IR"))
    truth.info["scenario"]["rigs"]["port"]["videos"] = {"T1": video.model_dump()}
    truth.write(run)
    (run / "EO.mp4").rename(run / "port_T1.mp4")

    recording.export(run)

    (folder,) = run.glob("recordings/port/*")
    sensors = json.loads((folder / "POWER_Const/pod_sensors.json").read_text())
    assert set(sensors["cameras"]) == {"EO", "IR"}
    serialized = json.loads((folder / "serialized_parameters.json").read_text())
    (entry,) = serialized["cameras"]
    assert entry["transformation_type"] == "horizontal_stitching"
    devices = [(s["name"], s["device_name"]) for s in entry["camera_sources"]]
    assert devices == [("EO", "EO"), ("IR", "IR")]


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
    """Oceanus's rotation order, Rz @ Ry @ Rx, in radians."""
    c, s = np.cos([roll, pitch, yaw]), np.sin([roll, pitch, yaw])
    about_x = np.array([[1, 0, 0], [0, c[0], -s[0]], [0, s[0], c[0]]])
    about_y = np.array([[c[1], 0, s[1]], [0, 1, 0], [-s[1], 0, c[1]]])
    about_z = np.array([[c[2], -s[2], 0], [s[2], c[2], 0], [0, 0, 1]])
    return about_z @ about_y @ about_x


def _core_imu_horizon(
    row: dict, sensor: dict, x: np.ndarray, dip: float = 0.0
) -> np.ndarray:
    """Oceanus's horizon from the IMU, ported: the CSV's roll and pitch with its yaw
    zeroed, turned by the camera's rpy, and projected. Oceanus leaves out the horizon's
    `dip`, which reads as the camera pitched up by it.
    """
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
    small angle, and a camera yawed in its pod is not one.
    """
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


def test_a_lighter_hull_raises_its_mounting_height(run: Path) -> None:
    truth = Labels.model_validate_json((run / "labels.json").read_text())
    ship = Ownship(asset="cargo_ship", draught_m=5.24)
    truth.info["scenario"]["ownship"] = ship.model_dump()
    truth.write(run)
    recording.export(run)
    (settings,) = run.glob("recordings/port/*/user_settings.json")
    height_m = json.loads(settings.read_text())["boat_setup"]["mounting_height"]
    assert height_m == pytest.approx(51.8 + ship.rise_m)


def test_a_re_export_leaves_only_what_it_wrote(run: Path) -> None:
    truth = Labels.model_validate_json((run / "labels.json").read_text())
    truth.info["scenario"]["rigs"]["port"]["hardware_version"] = "Oscar_V7_Gen_3_1"
    truth.write(run)
    recording.export(run)
    truth.info["scenario"]["rigs"]["port"]["hardware_version"] = None
    truth.write(run)
    written = recording.export(run)
    (into,) = run.glob("recordings/port/*")
    assert sorted(p for p in into.rglob("*") if p.is_file()) == sorted(written)


def _cropped_for_hardware(run: Path) -> Path:
    """The rig as a cropped video on a named hardware version: the branches a bare
    camera leaves out of `serialized_parameters.json` and `module_info.json`.
    """
    video = Crop(kind="crop", camera="EO", left_px=40, right_px=40)
    calibration = Calibration.read(run)
    cropped = [
        camera.model_copy(update={"name": "port_RGB"}).cropped(video)
        for camera in calibration.cameras
    ]
    Calibration(cameras=cropped, raw=calibration.cameras).write(run)
    truth = Labels.model_validate_json((run / "labels.json").read_text())
    rig = truth.info["scenario"]["rigs"]["port"]
    rig |= {
        "videos": {"RGB": video.model_dump()},
        "hardware_version": "Oscar_V7_Gen_3_1",
    }
    truth.write(run)
    (run / "EO.mp4").rename(run / "port_RGB.mp4")
    return run


def _written(run: Path) -> dict[str, object]:
    """Each table and setting, with JSON floats to 9 decimals: libm's last digits
    differ across machines.
    """
    (into,) = {p.parent for p in recording.export(run) if p.suffix == ".csv"}
    files = {}
    for p in sorted(into.rglob("*")):
        if p.suffix == ".csv":
            files[p.relative_to(into).as_posix()] = p.read_text(newline="").split("\n")
        elif p.suffix == ".json":
            text = p.read_text()
            files[p.relative_to(into).as_posix()] = json.loads(
                text, parse_float=lambda f: round(float(f), 9) + 0.0
            )
    return files


def test_a_recording_stays_as_written(tmp_path: Path) -> None:
    """Every table and setting a fixed rig, and a cropped PTU rig, write, as the
    text of `recording_golden.json`, so an int turned float shows. To change the
    recording on purpose, delete the file and rerun.
    """
    written = {}
    for key in ("fixed", "ptu"):
        (tmp_path / key).mkdir()
        run = make_run(tmp_path / key, CAMERA_YAW_DEG, ptu=key == "ptu")
        written[key] = _written(_cropped_for_hardware(run) if key == "ptu" else run)
    text = json.dumps(written, indent=1) + "\n"
    if not GOLDEN.exists():
        GOLDEN.write_text(text)
        pytest.fail(f"wrote {GOLDEN.name}; rerun")
    assert text == GOLDEN.read_text()
