"""Rigs and their cameras, and where each camera points."""

import math
from typing import NamedTuple

import bpy
from mathutils import Vector

from seascape.blender.blend import (
    place,
    yaw,
)
from seascape.config import (
    Scenario,
)


class _RigObjects(NamedTuple):
    root: bpy.types.Object
    rigs: dict[str, bpy.types.Object]
    cameras: dict[str, bpy.types.Object]


def add_rigs(scenario: Scenario, far_m: float) -> _RigObjects:
    root = bpy.data.objects.new("rigs", None)
    bpy.context.collection.objects.link(root)
    place(root, 0.0, 0.0, 0.0)

    rigs: dict[str, bpy.types.Object] = {}
    for name, rig in scenario.rigs.items():
        # Blender takes an inverted frustum without complaint and renders nothing.
        if rig.near_clip_m >= far_m:
            raise ValueError(
                f"{name}: near clip {rig.near_clip_m} m is past the far plane at "
                f"{far_m:.0f} m"
            )
        empty = bpy.data.objects.new(f"rig_{name}", None)
        bpy.context.collection.objects.link(empty)
        empty.parent = root
        place(empty, rig.offset_x_m, rig.offset_y_m, rig.height_m)
        # XYZ euler is Rz @ Ry @ Rx: yaw, then pitch about the rig's own transverse
        # axis, so a pitched rig rolls the horizon of its off-axis cameras.
        empty.rotation_euler = (math.radians(rig.pitch_deg), 0.0, yaw(rig.yaw_deg))
        rigs[name] = empty

    cameras: dict[str, bpy.types.Object] = {}
    for mount in scenario.mounts:
        data = bpy.data.cameras.new(mount.name)
        # AUTO fits the field of view to whichever image dimension is larger, so a
        # portrait sensor would silently reinterpret hfov as a vertical angle.
        data.sensor_fit = "HORIZONTAL"
        data.angle_x = math.radians(mount.camera.hfov_deg)
        data.clip_start = mount.rig.near_clip_m
        # The default 1000 m renders a target past it as sky, and the clip boundary
        # reads as the horizon. Nothing warns.
        data.clip_end = far_m
        camera = bpy.data.objects.new(data.name, data)
        bpy.context.collection.objects.link(camera)
        camera.parent = rigs[mount.rig_name]
        camera.rotation_mode = "XYZ"
        # A camera looks down its local -Z; +90 deg about X aims it at the horizon.
        camera.rotation_euler = (
            math.radians(90.0 + mount.camera.pitch_deg),
            0.0,
            yaw(mount.camera.yaw_deg),
        )
        cameras[mount.name] = camera
    return _RigObjects(root, rigs, cameras)


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
