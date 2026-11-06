"""Rigs and their cameras, and where each camera points."""

import math
from typing import NamedTuple

import bpy
from mathutils import Matrix, Vector

from seascape.blender.blend import (
    empty,
    yaw,
)
from seascape.config import (
    Band,
    Mount,
    Scenario,
)
from seascape.physics import fisheye


class _RigObjects(NamedTuple):
    root: bpy.types.Object
    rigs: dict[str, bpy.types.Object]
    cameras: dict[str, bpy.types.Object]


def add_rigs(scenario: Scenario, far_m: float) -> _RigObjects:
    """The scenario's rigs and cameras under one root, each seeing to `far_m`."""
    root = empty("rigs")

    rigs: dict[str, bpy.types.Object] = {}
    for name, rig in scenario.rigs.items():
        # Blender takes an inverted frustum without complaint and renders nothing.
        if rig.near_clip_m >= far_m:
            raise ValueError(
                f"{name}: near clip {rig.near_clip_m} m is past the far plane at "
                f"{far_m:.0f} m"
            )
        rig_root = empty(
            f"rig_{name}", root, (rig.offset_x_m, rig.offset_y_m, rig.height_m)
        )
        # XYZ euler is Rz @ Ry @ Rx: yaw, then pitch about the rig's own transverse
        # axis, so a pitched rig rolls the horizon of its off-axis cameras.
        rig_root.rotation_euler = (math.radians(rig.pitch_deg), 0.0, yaw(rig.yaw_deg))
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
