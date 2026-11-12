"""Rigs and their cameras, and where each camera points."""

import math
from collections.abc import Callable, Sequence
from typing import NamedTuple

import bpy
import numpy as np
from mathutils import Matrix, Vector

from seascape.blender.blend import (
    BLENDER_TO_CV,
    animate,
    empty,
    yaw,
)
from seascape.config import (
    Band,
    Mount,
    Ptu,
    Rig,
    Scenario,
)
from seascape.motion import panned_deg
from seascape.outputs.calibration import in_rig
from seascape.physics import fisheye, ptu


class _RigObjects(NamedTuple):
    root: bpy.types.Object
    rigs: dict[str, bpy.types.Object]
    cameras: dict[str, bpy.types.Object]


def add_rigs(scenario: Scenario) -> _RigObjects:
    """The scenario's rigs and cameras under one root."""
    root = empty("rigs")

    rigs: dict[str, bpy.types.Object] = {}
    for name, rig in scenario.rigs.items():
        rig_root = empty(
            f"rig_{name}", root, (rig.offset_x_m, rig.offset_y_m, rig.height_m)
        )
        # XYZ euler is Rz @ Ry @ Rx: yaw, then pitch about the rig's own transverse
        # axis, so a pitched rig rolls the horizon of its off-axis cameras.
        rig_root.rotation_euler = _head(rig.pitch_deg, rig.yaw_deg)
        rigs[name] = rig_root

    cameras: dict[str, bpy.types.Object] = {}
    for mount in scenario.mounts:
        data = bpy.data.cameras.new(mount.name)
        # AUTO fits the field of view to whichever image dimension is larger, so a
        # portrait sensor would silently reinterpret hfov as a vertical angle.
        data.sensor_fit = "HORIZONTAL"
        width, height = mount.camera.width_px, mount.camera.height_px
        if (k := mount.camera.lens.fisheye) is None:
            data.angle_x = math.radians(mount.camera.hfov_deg)
        else:
            fx = mount.camera.fisheye_focal_px
            data.type, data.panorama_type = "PANO", "FISHEYE_LENS_POLYNOMIAL"
            # The sensor fit is horizontal, so the width spans the frame.
            pitch_mm = data.sensor_width / width
            for i, c in enumerate(fisheye.polynomial(k, fx, width, height, pitch_mm)):
                setattr(data, f"fisheye_polynomial_k{i}", c)
            # What the sea's shader and the cull read as a pixel: the one at the
            # centre, which sees the least. Past it a fisheye's pixel sees more.
            data.angle_x = width / fx
        camera = bpy.data.objects.new(data.name, data)
        bpy.context.collection.objects.link(camera)
        camera.parent = rigs[mount.rig_name]
        camera.rotation_mode = "XYZ"
        # A Blender camera looks down its local -Z, OpenCV's down +Z.
        camera.matrix_basis = Matrix(in_rig(mount.camera)) @ BLENDER_TO_CV
        cameras[mount.name] = camera
    return _RigObjects(root, rigs, cameras)


def stabilise(
    rigs: dict[str, bpy.types.Object],
    scenario: Scenario,
    attitude_deg: Callable[[float], tuple[float, float]],
) -> None:
    """Key each rig on a pan-tilt unit: its pan to a turn by the ptu, and its tilt to
    the unit's loop holding its pitch against the base under it.

    Args:
        rigs: The rig empties `add_rigs` made, by name.
        scenario: Its rigs, its turn and the times keyed.
        attitude_deg: The ownship's pitch, bow up, and roll, starboard down, at a time.
    """
    turn, times = scenario.turn, scenario.outputs.times_s
    for name, rig in scenario.rigs.items():
        if rig.ptu is not None:
            panned = panned_deg(turn, name)
            _hold(rigs[name], rig, rig.ptu, panned, attitude_deg, times)


def _head(pitch_deg: float, yaw_deg: float) -> tuple[float, float, float]:
    """A rig empty's XYZ euler for its pitch up and its bearing clockwise."""
    return (math.radians(pitch_deg), 0.0, yaw(yaw_deg))


def _hold(
    head: bpy.types.Object,
    rig: Rig,
    unit: Ptu,
    panned: Callable[[float], float],
    attitude_deg: Callable[[float], tuple[float, float]],
    times_s: Sequence[float],
) -> None:
    def pan_deg(t_s: float) -> float:
        return rig.yaw_deg + panned(t_s)

    tilt = ptu.panned_tilt_deg(
        attitude_deg, pan_deg, times_s, unit.inclination_deg, rig.pitch_deg
    )

    def euler(t_s: float) -> tuple[float, float, float]:
        return _head(rig.pitch_deg + float(np.interp(t_s, times_s, tilt)), pan_deg(t_s))

    animate(head, "rotation_euler", times_s, lambda t: euler(t)[0], 0)
    animate(head, "rotation_euler", times_s, lambda t: euler(t)[2], 2)


class View(NamedTuple):
    """A camera's pose at one instant, and its pixel at the render's scale."""

    mount: Mount
    t_s: float
    matrix: Matrix
    angle_x_rad: float
    pixel_rad: float


def views(
    scenario: Scenario, band: Band, cameras: dict[str, bpy.types.Object]
) -> list[View]:
    """Every frame's pose of each of `band`'s cameras, the scene left at frame 0."""
    sc = bpy.context.scene
    # As the shader's pixel does, so a cull matches what it draws.
    scale = sc.render.resolution_percentage / 100
    found = []
    for frame, t_s in enumerate(scenario.outputs.times_s):
        sc.frame_set(frame)
        for mount in scenario.band_mounts(band):
            data = cameras[mount.name].data
            found.append(
                View(
                    mount,
                    t_s,
                    cameras[mount.name].matrix_world.copy(),
                    data.angle_x,
                    data.angle_x / (mount.camera.width_px * scale),
                )
            )
    sc.frame_set(0)
    return found


def boresight_deg(camera: bpy.types.Object) -> tuple[float, float]:
    """Bearing and elevation a built camera points at, in degrees, in the world.
    `matrix_world` is stale until the depsgraph runs, so build first.
    """
    forward = camera.matrix_world.to_3x3() @ Vector((0.0, 0.0, -1.0))
    forward.normalize()
    return (
        math.degrees(math.atan2(forward.x, forward.y)),
        math.degrees(math.asin(min(1.0, max(-1.0, forward.z)))),
    )
