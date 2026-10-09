"""A render's videos laid out as a rig model's recording, for tools that replay them.

Imports no Blender: everything comes from the files a render and `seascape video`
wrote.
"""

import csv
import json
import math
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from seascape import labels
from seascape.calibration import Calibration, CameraCalibration

COLUMNS = [
    "counter",
    "roll",
    "pitch",
    "yaw",
    "pan",
    "tilt",
    "lati",
    "longi",
    "timestp",
    "heading",
    "rateofturn",
    "cog",
    "sog2",
    "sow",
    "sog",
    "temp",
    "humidity",
    "pressure",
    "wind_speed",
    "wind_angle",
    "bb_roll",
    "bb_pitch",
    "bb_yaw",
    "proc_ptu_mode",
    "proc_nav_mode",
    "proc_colav_limit",
    "proc_colav_speed",
    "proc_surveil_speed",
]

# ponytail: nominal pitches, not a datasheet's; seascape models no sensor.
PIXEL_SIZE_MM = {"eo": 0.00345, "ir": 0.012}

# The recording's camera frame, +X right, +Y forward, +Z up, as OpenCV's axes.
_BODY_TO_CV = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def attitude_deg(body_to_frame: np.ndarray) -> tuple[float, float, float]:
    """Roll, pitch, yaw of a +X starboard, +Y bow, +Z up body, yaw then pitch then
    roll: roll positive starboard down, pitch bow up, yaw clockwise in [-180, 180)."""
    x, y, z = body_to_frame[:3, :3].T
    roll = math.atan2(-x[2], z[2])
    pitch = math.asin(max(-1.0, min(1.0, y[2])))
    yaw = math.atan2(y[0], y[1])
    # Rounded to the CSV's precision first, so float32 noise at 180 cannot wrap.
    return (
        math.degrees(roll),
        math.degrees(pitch),
        (round(math.degrees(yaw), 6) + 180.0) % 360.0 - 180.0,
    )


def _to(camera: CameraCalibration, frame: str) -> np.ndarray:
    """The named frame's own axes in world: camera-to-world after frame-to-camera."""
    world = np.array(camera.extrinsics["world"])
    return world @ np.linalg.inv(np.array(camera.extrinsics[frame]))


def _row(camera: CameraCalibration, counter: int, scenario: dict[str, Any]) -> dict:
    roll, pitch, yaw = attitude_deg(_to(camera, "rig"))
    # NMEA 2000 attitude and heading, in radians: roll positive starboard down, pitch
    # bow up.
    bb_roll, bb_pitch, bb_yaw = np.radians(attitude_deg(_to(camera, "vessel")))
    # Rounded first, as in attitude_deg.
    heading = round(float(bb_yaw), 6) % math.tau
    # pan and tilt stay 0: a rig is bolted on, with no pan-tilt unit.
    row = dict.fromkeys(COLUMNS, 0.0)
    row |= {
        "counter": counter,
        # The recording's IMU: degrees, pitch positive nose down.
        "roll": roll,
        "pitch": -pitch,
        "yaw": yaw,
        "heading": heading,
        "cog": heading,
        "temp": scenario["sky"]["t_air_k"] - 273.15,
        "wind_speed": scenario["sea"]["wind_speed_mps"],
        # NMEA 2000: radians in [0, 2 pi). Relative to the bow, as the scenario's.
        "wind_angle": math.radians(scenario["sea"]["wind_from_deg"]) % math.tau,
        "bb_roll": float(bb_roll),
        "bb_pitch": float(bb_pitch),
        "bb_yaw": float(bb_yaw),
    }
    # + 0.0 turns the -0.0 that rounding leaves of noise into 0.0.
    return {
        k: v if isinstance(v, int) else f"{round(v, 6) + 0.0:.6f}"
        for k, v in row.items()
    }


def _unit(value: Any, unit: str) -> dict[str, Any]:
    return {"value": value, "unit": unit}


def _sensor(camera: CameraCalibration) -> dict[str, Any]:
    rig = np.array(camera.extrinsics["rig"])
    roll, pitch, yaw = attitude_deg(rig[:3, :3] @ _BODY_TO_CV)
    # The recording's camera rpy turn +X forward, +Y left, +Z up: yaw positive to
    # port, pitch positive down.
    pitch, yaw = -pitch, -yaw
    (fx, _, cx), (_, fy, cy), _ = camera.K
    pixel_mm = PIXEL_SIZE_MM[camera.band]
    return {
        # Never a hardware model's name: a live pipeline picks its driver by it.
        "camera_type": f"Blender {camera.band.upper()}",
        "position": _unit([float(v) * 1000.0 for v in rig[:3, 3]], "mm"),
        "focal_length": _unit(fx * pixel_mm, "mm"),
        "barrel_distortion": _unit(None, "1"),
        "distortion_model": "pinhole",
        "roll": _unit(roll, "deg"),
        "pitch": _unit(pitch, "deg"),
        "yaw": _unit(yaw, "deg"),
        "pixel_size": _unit(pixel_mm, "mm"),
        "resolution": _unit([camera.width_px, camera.height_px], "1"),
        "camera_orientation": _unit(0, "deg"),
        "camera_serial_number": camera.name,
        "border_cut": _unit([0, 0], "px"),
        "border_shift": _unit([0, 0], "px"),
        "center_x": _unit(cx, "1"),
        "center_y": _unit(cy, "1"),
        "focal_length_x": _unit(fx, "1"),
        "focal_length_y": _unit(fy, "1"),
        "fisheye_distortion": _unit([0.0] * 4, "1"),
    }


def _pod_sensors(cameras: list[CameraCalibration], created: datetime) -> dict:
    """The simulated IMU sits exactly on the rig's axes, so its corrections are 0."""
    zero = {"value": 0.0, "unit": "deg"}
    return {
        "calibration_info": {
            "setup_date": created.strftime("%Y-%m-%d %H:%M:%S.%f"),
            "json_version": 3,
            "calibrated_by": "seascape",
            "camera_calibration_version": 2,
            "imu_calibration_version": 1,
        },
        "imu": {
            "bno055": {
                "offset_xyz": _unit([0.0] * 3, "g"),
                "gain_xyz": _unit([1.0] * 3, "1"),
                "yaw": zero,
                "pitch": zero,
                "roll": zero,
                "t_mean": _unit(0.0, "°C"),
                "gyro_offset": _unit([0.0] * 3, "rad/s"),
            },
            "mti2": dict.fromkeys(
                ("yaw", "pitch", "roll", "yaw_err", "pitch_err", "roll_err"), zero
            ),
        },
        "cameras": {camera.name: _sensor(camera) for camera in cameras},
    }


def _camera_parameters(cameras: list[CameraCalibration]) -> dict:
    sensors = {camera.name: _sensor(camera) for camera in cameras}
    return {
        name: {k: sensor[k]["value"] for k in ("roll", "pitch", "yaw")}
        | {"barrel": 0.0}
        for name, sensor in sensors.items()
    }


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=4) + "\n")


def export(run: Path) -> list[Path]:
    """One recording folder per rig, under `run/recordings/<rig>/`, named for its
    model."""
    truth = labels.Labels.model_validate_json((run / labels.FILENAME).read_text())
    if "scenario" not in truth.info or "date_created" not in truth.info:
        raise ValueError(f"{run / labels.FILENAME}: no scenario or date in its info")
    # Read, not validated: a dump carries every default, and a validator that rejects
    # a field set alongside another rejects the dump.
    scenario = truth.info["scenario"]
    created = datetime.fromisoformat(truth.info["date_created"])
    ts = created.strftime("%Y_%m_%d_%H_%M_%S")
    time_s = {image.file_name: image.time_s for image in truth.images}

    frames: dict[str, dict[str, list[CameraCalibration]]] = {}
    models: dict[str, str | None] = {}
    for camera in Calibration.read(run).cameras:
        frames.setdefault(camera.rig, {}).setdefault(camera.name, []).append(camera)
        models[camera.rig] = camera.model

    # Checked before anything is written, so a bad run leaves no half folder.
    if unnamed := sorted(rig for rig, model in models.items() if model is None):
        raise ValueError(
            f"rigs {unnamed} have no model, whose format a recording takes"
        )
    shots = [s for cameras in frames.values() for c in cameras.values() for s in c]
    if unlabelled := [shot.image for shot in shots if shot.image not in time_s]:
        raise ValueError(f"{run / labels.FILENAME} has no frame {unlabelled}")
    names = [name for cameras in frames.values() for name in cameras]
    if missing := [name for name in names if not (run / f"{name}.mp4").exists()]:
        raise FileNotFoundError(f"no video for {missing}: run `seascape video`")

    written = []
    for rig, cameras in frames.items():
        model = models[rig]
        into = run / "recordings" / rig / f"{model}_recordings_{ts}"
        (into / "POWER_Const").mkdir(parents=True, exist_ok=True)
        for name, shots in cameras.items():
            shots.sort(key=lambda shot: time_s[shot.image])
            video = into / f"{model}_{name}_record_{ts}.mp4"
            video.unlink(missing_ok=True)
            # A hard link keeps the render's own layout valid without a copy; a
            # filesystem without them gets the copy.
            try:
                os.link(run / f"{name}.mp4", video)
            except OSError:
                shutil.copy2(run / f"{name}.mp4", video)
            table = into / f"{model}_{name}_metadata_{ts}.csv"
            with table.open("w", newline="") as f:
                writer = csv.DictWriter(f, COLUMNS, lineterminator="\n")
                writer.writeheader()
                writer.writerows(_row(s, i, scenario) for i, s in enumerate(shots))
            written += [video, table]
        firsts = [shots[0] for shots in cameras.values()]
        settings = into / "user_settings.json"
        _write_json(
            settings,
            {
                "boat_setup": {
                    "mounting_height": scenario["rigs"][rig]["height_m"],
                    "mast_to_bow_length": 0.0,
                }
            },
        )
        parameters = into / "POWER_Const" / "camera_parameters.json"
        _write_json(parameters, _camera_parameters(firsts))
        sensors = into / "POWER_Const" / "pod_sensors.json"
        _write_json(sensors, _pod_sensors(firsts, created))
        written += [settings, parameters, sensors]
    return written
