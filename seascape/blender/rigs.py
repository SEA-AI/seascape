"""Rigs and their cameras, and where each camera points."""

import math
from collections.abc import Callable, Sequence
from typing import NamedTuple

import bpy
import numpy as np
from mathutils import Euler, Matrix

from seascape import angles
from seascape.blender.blend import (
    BLENDER_TO_CV,
    animate,
    empty,
)
from seascape.config import (
    Band,
    Mount,
    Rig,
    Scenario,
)
from seascape.motion import Tilt, held_tilt_deg, panned_deg
from seascape.physics import fisheye


class _RigObjects(NamedTuple):
    root: bpy.types.Object
    heads: dict[str, bpy.types.Object]
    cameras: dict[str, bpy.types.Object]


def add_rigs(scenario: Scenario) -> _RigObjects:
    """The scenario's rigs and cameras under one root."""
    root = empty("rigs")

    heads: dict[str, bpy.types.Object] = {}
    for name, rig in scenario.rigs.items():
        head = empty(f"head_{name}", root)
        head.matrix_basis = Matrix(rig.pose().tolist())
        heads[name] = head

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
        camera.parent = heads[mount.rig_name]
        camera.rotation_mode = "XYZ"
        # A Blender camera looks down its local -Z, OpenCV's down +Z.
        camera.matrix_basis = Matrix(mount.camera.pose().tolist()) @ BLENDER_TO_CV
        cameras[mount.name] = camera
    return _RigObjects(root, heads, cameras)


def stabilise(
    heads: dict[str, bpy.types.Object],
    scenario: Scenario,
    tilt: Tilt,
) -> None:
    """Key each rig on a pan-tilt unit: its pan to a turn by the ptu, and its tilt to
    the unit's loop holding its pitch against the base under it.

    Args:
        heads: The head empties `add_rigs` made, by name.
        scenario: Its rigs, its turn and the times keyed.
        tilt: The ownship's `R_level_ownship` at a time, heading aside.
    """
    times = scenario.outputs.times_s
    for name, rig in scenario.rigs.items():
        if rig.ptu is not None:
            held = held_tilt_deg(scenario, name, tilt)
            _hold(heads[name], rig, panned_deg(scenario.turn, name), held, times)


def _hold(
    head: bpy.types.Object,
    rig: Rig,
    pan_deg: Callable[[float], float],
    tilt_deg: np.ndarray,
    times_s: Sequence[float],
) -> None:
    # Each frame's euler nearest the last, so a pan past the stern keys no spin back.
    eulers: dict[float, Euler] = {}
    last: Euler | None = None
    for t_s, tilt in zip(times_s, tilt_deg, strict=True):
        turn = Matrix(rig.pose(pan_deg(t_s), float(tilt))[:3, :3].tolist())
        last = turn.to_euler("XYZ") if last is None else turn.to_euler("XYZ", last)
        eulers[t_s] = last
    animate(head, "rotation_euler", times_s, lambda t: tuple(eulers[t]))


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
    # OpenCV's camera looks along +Z.
    forward = (camera.matrix_world @ BLENDER_TO_CV).to_3x3().col[2].normalized()
    return angles.bearing_deg(forward.x, forward.y), float(
        angles.elevation_deg(forward.z)
    )
