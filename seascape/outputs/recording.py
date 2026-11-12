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
from pydantic import TypeAdapter

from seascape.angles import TO_CV, attitude_deg, wrapped_deg
from seascape.config import Crop, Turn, Undistortion, Video, sources
from seascape.outputs import labels
from seascape.outputs.calibration import Calibration, CameraCalibration

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

# Nominal pitches, not a datasheet's, for a camera with no pitch set.
PIXEL_SIZE_MM = {"eo": 0.00345, "ir": 0.012}
_VIDEO = TypeAdapter(Video)
# Oceanus's `proc_ptu_mode`: collision avoidance while a ptu
# turns, else manual, as a replay reads a recording without the column. It has no 0.
MANUAL_MODE, COLAV_MODE = 1, 3
# Roll, pitch and yaw signs in FLU (ROS REP 103) against `attitude_deg`'s: pitch
# counts nose down, yaw to port.
FLU = (1.0, -1.0, -1.0)
# The recording's IMU pitch counts nose down, its roll and yaw as `attitude_deg`'s;
# no change of basis gives that mix.
IMU_PITCH_SIGN = -1.0


def _in_world(camera: CameraCalibration, frame: str) -> np.ndarray:
    """T_w_f: the named frame's pose in world."""
    T_w_c = np.array(camera.extrinsics["world"])
    return T_w_c @ np.linalg.inv(np.array(camera.extrinsics[frame]))


def _attitude_deg(body_to_frame: np.ndarray) -> tuple[float, float, float]:
    """`angles.attitude_deg`, its yaw rounded to the CSV's precision first, so float32
    noise at 180 cannot wrap.
    """
    roll, pitch, yaw = attitude_deg(body_to_frame)
    return roll, pitch, wrapped_deg(round(yaw, 6))


def _flu_deg(body_to_frame: np.ndarray) -> tuple[float, float, float]:
    """`_attitude_deg` in FLU."""
    roll, pitch, yaw = (
        s * a for s, a in zip(FLU, _attitude_deg(body_to_frame), strict=True)
    )
    return roll, pitch, yaw


def _row(
    camera: CameraCalibration,
    counter: int,
    scenario: dict[str, Any],
    rig: dict[str, Any],
    scan_deg_s: float | None,
) -> dict:
    roll, pitch, yaw = _attitude_deg(_in_world(camera, "rig"))
    # NMEA 2000 attitude and heading, in radians: roll positive starboard down, pitch
    # bow up.
    bb_roll, bb_pitch, bb_yaw = np.radians(_attitude_deg(_in_world(camera, "vessel")))
    # Rounded first, as in _attitude_deg.
    heading = round(float(bb_yaw), 6) % math.tau
    row = dict.fromkeys(COLUMNS, 0.0)
    row["proc_ptu_mode"] = MANUAL_MODE
    # A run from before ptus has no key.
    if rig.get("ptu") is not None:
        # The head on the deck, less its mount: Rig.pose adds pan to yaw and tilt to
        # pitch, so subtracting undoes them exactly.
        T_v_r = np.linalg.inv(_in_world(camera, "vessel")) @ _in_world(camera, "rig")
        _, tilt, pan = _attitude_deg(T_v_r)
        row["pan"] = FLU[2] * wrapped_deg(pan - rig["yaw_deg"])
        row["tilt"] = FLU[1] * (tilt - rig["pitch_deg"])
        if scan_deg_s is not None:
            row["proc_ptu_mode"] = COLAV_MODE
            # Revolutions per second, as the worker's setting.
            row["proc_colav_speed"] = scan_deg_s / 360.0
    row |= {
        "counter": counter,
        # The recording's IMU, in degrees.
        "roll": roll,
        "pitch": IMU_PITCH_SIGN * pitch,
        "yaw": yaw,
        "heading": heading,
        "cog": heading,
        "temp": scenario["sky"]["t_air_k"] - 273.15,
        "wind_speed": scenario["sea"]["wind_speed_mps"],
        # NMEA 2000: radians in [0, 2 pi) from the current bow; the scenario's is
        # from the bow at the start.
        "wind_angle": (math.radians(scenario["sea"]["wind_from_deg"]) - heading)
        % math.tau,
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
    T_r_c = np.array(camera.extrinsics["rig"])
    # The camera's OpenCV axes as a body's: right, ahead, up.
    rpy = zip(("roll", "pitch", "yaw"), _flu_deg(T_r_c[:3, :3] @ TO_CV), strict=True)
    (fx, _, cx), (_, fy, cy), _ = camera.K
    pixel_mm = camera.pitch_um / 1000 if camera.pitch_um else PIXEL_SIZE_MM[camera.band]
    return {
        # Never a hardware model's name: a live pipeline picks its driver by it.
        "camera_type": f"Blender {camera.band.upper()}",
        "position": _unit([float(v) * 1000.0 for v in T_r_c[:3, 3]], "mm"),
        "focal_length": _unit(fx * pixel_mm, "mm"),
        "barrel_distortion": _unit(None, "1"),
        "distortion_model": "pinhole" if camera.distortion is None else "fisheye",
        **{k: _unit(v, "deg") for k, v in rpy},
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
        "fisheye_distortion": _unit(list(camera.distortion or [0.0] * 4), "1"),
    }


def _sensors(cameras: dict[str, CameraCalibration], created: datetime) -> dict:
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
        "cameras": {name: _sensor(camera) for name, camera in cameras.items()},
    }


def _camera_parameters(cameras: dict[str, CameraCalibration]) -> dict:
    sensors = {name: _sensor(camera) for name, camera in cameras.items()}
    return {
        name: {k: sensor[k]["value"] for k in ("roll", "pitch", "yaw")}
        | {"barrel": 0.0}
        for name, sensor in sensors.items()
    }


def _parameters(
    videos: dict[str, CameraCalibration],
    raw: dict[str, CameraCalibration],
    settings: dict[str, Any],
) -> list[dict[str, Any]]:
    """`serialized_parameters.json`'s cameras: each video and its cameras.

    Args:
        videos: Each video's first frame, by its name in the recording.
        raw: Each rendered camera's first frame, by its name in the recording.
        settings: The rig's videos as the scenario set them, by name.
    """

    def values(camera: CameraCalibration) -> dict[str, Any]:
        return {
            k: v["value"] if isinstance(v, dict) else v
            for k, v in _sensor(camera).items()
        }

    entries = []
    for name, camera in videos.items():
        video = _VIDEO.validate_python(settings[name])
        transform = None
        if isinstance(video, Crop):
            # The recorder's order.
            sides = [video.left_px, video.right_px, video.top_px, video.bottom_px]
            transform = {"pixel_sacrifice": sides}
        if isinstance(video, Undistortion):
            fov = {"value": [video.hfov_deg, video.vfov_deg], "unit": "degrees"}
            transform = {"downscale": video.downscale, "out-fov": fov}
        names = sources(video)
        # One camera takes its video's name; a stitch's keep their own.
        devices = [name] if len(names) == 1 else names
        cameras = [values(raw[c]) for c in names]
        # The recorder gives a video its camera's lens and pixel, not its own.
        lens = {k: cameras[0][k] for k in ("focal_length", "pixel_size")}
        entries.append(
            {"name": name}
            | values(camera)
            | lens
            | {
                "transformation_type": video.kind,
                "transformation_parameters": transform,
                "camera_sources": [
                    {"name": c, "device_name": device} | fields
                    for c, device, fields in zip(names, devices, cameras, strict=True)
                ],
            }
        )
    return entries


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=4) + "\n")


def _write_settings(
    into: Path,
    model: str,
    rig: dict[str, Any],
    firsts: dict[str, CameraCalibration],
    raw: dict[str, CameraCalibration] | None,
    created: datetime,
) -> list[Path]:
    """A recording's settings and calibration, beside its videos.

    Args:
        into: The recording's folder.
        model: The product, which names its sensors file.
        rig: The rig as the scenario set it. Its hardware version, when set, goes to
            `module_info.json`, by which a replay picks its profile.
        firsts: Each video's first frame, by its name in the recording.
        raw: Each camera the videos were made from, by its name in the recording;
            None when the rig has no videos.
        created: When the render was made, the calibration's date.
    """
    settings = into / "user_settings.json"
    _write_json(
        settings,
        {"boat_setup": {"mounting_height": rig["height_m"], "mast_to_bow_length": 0.0}},
    )
    parameters = into / "POWER_Const" / "camera_parameters.json"
    _write_json(parameters, _camera_parameters(firsts))
    videos = rig["videos"]
    if raw is None:
        raw = firsts
        videos = {name: {"kind": "identity", "camera": name} for name in firsts}
    sensors = into / "POWER_Const" / f"{model.lower()}_sensors.json"
    _write_json(sensors, _sensors(raw, created))
    serialized = into / "serialized_parameters.json"
    entries = _parameters(firsts, raw, videos)
    _write_json(serialized, {"product_name": model, "cameras": entries})
    written = [settings, parameters, sensors, serialized]
    if rig.get("hardware_version"):
        module = into / "POWER_Const" / "module_info.json"
        _write_json(
            module,
            {
                "hardware_version": rig["hardware_version"],
                "setup_date": created.strftime("%Y-%m-%d %H:%M:%S.%f"),
                "serial_number": "seascape",
                "jetson_serial_number": 0,
                "json_version": 1,
            },
        )
        written.append(module)
    return written


def export(run: Path) -> list[Path]:
    """One recording folder per rig, under `run/recordings/<rig>/`, named for its
    model.
    """
    truth = labels.Labels.model_validate_json((run / labels.FILENAME).read_text())
    if "scenario" not in truth.info or "date_created" not in truth.info:
        raise ValueError(f"{run / labels.FILENAME}: no scenario or date in its info")
    # Read, not validated: a dump carries every default, and a validator that rejects
    # a field set alongside another rejects the dump.
    scenario = truth.info["scenario"]
    turn = (
        None if scenario.get("turn") is None else Turn.model_validate(scenario["turn"])
    )
    created = datetime.fromisoformat(truth.info["date_created"])
    ts = created.strftime("%Y_%m_%d_%H_%M_%S")
    time_s = {image.file_name: image.time_s for image in truth.images}

    calibration = Calibration.read(run)
    frames: dict[str, dict[str, list[CameraCalibration]]] = {}
    models: dict[str, str | None] = {}
    for camera in calibration.cameras:
        frames.setdefault(camera.rig, {}).setdefault(camera.name, []).append(camera)
        models[camera.rig] = camera.model
    raw: dict[str, dict[str, CameraCalibration]] = {}
    for camera in calibration.raw:
        raw.setdefault(camera.rig, {}).setdefault(_short(camera), camera)

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
        assert model is not None  # refused above
        into = run / "recordings" / rig / f"{model}_recordings_{ts}"
        _empty(into / "POWER_Const")
        for name, shots in cameras.items():
            shots.sort(key=lambda shot: time_s[shot.image])
            short = _short(shots[0])
            video = into / f"{model}_{short}_record_{ts}.mp4"
            # A hard link keeps the render's own layout valid without a copy; a
            # filesystem without them gets the copy.
            try:
                os.link(run / f"{name}.mp4", video)
            except OSError:
                shutil.copy2(run / f"{name}.mp4", video)
            table = into / f"{model}_{short}_metadata_{ts}.csv"
            with table.open("w", newline="") as f:
                writer = csv.DictWriter(f, COLUMNS, lineterminator="\n")
                writer.writeheader()
                head = scenario["rigs"][rig]
                scan = (
                    turn.scan_rate_deg_s
                    if turn is not None and turn.pans(rig)
                    else None
                )
                writer.writerows(
                    _row(s, i, scenario, head, scan) for i, s in enumerate(shots)
                )
            written += [video, table]
        firsts = {_short(shots[0]): shots[0] for shots in cameras.values()}
        written += _write_settings(
            into, model, scenario["rigs"][rig], firsts, raw.get(rig), created
        )
    return written


def _empty(folder: Path) -> None:
    """`folder` made afresh under an empty parent, so nothing an earlier export wrote
    outlives a re-export.
    """
    if folder.parent.exists():
        shutil.rmtree(folder.parent)
    folder.mkdir(parents=True)


def _short(camera: CameraCalibration) -> str:
    """A camera's name in its rig's recording, which is the rig's alone."""
    return camera.name.removeprefix(f"{camera.rig}_")
