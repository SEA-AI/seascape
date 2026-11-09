"""Workbench renders of objects, framed whole from a bearing and elevation."""

from pathlib import Path
from typing import NamedTuple

import bpy
import numpy as np
from mathutils import Vector

from seascape.blender.meshes import box_corners
from seascape.physics import earth

VIEW_PX = (1600, 1200)


class Aim(NamedTuple):
    """Where a view looks from."""

    name: str
    # Bearing from the asset's centre, bow 0, starboard 90.
    bearing_deg: float
    elevation_deg: float


def setup() -> None:
    """Workbench at `VIEW_PX` on a transparent film, its colours as drawn."""
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_WORKBENCH"
    sc.render.resolution_x, sc.render.resolution_y = VIEW_PX
    sc.render.resolution_percentage = 100
    sc.render.film_transparent = True
    sc.render.dither_intensity = 0.0
    view = sc.view_settings
    view.view_transform, view.look = "Standard", "None"
    view.exposure, view.gamma = 0.0, 1.0
    shading = sc.display.shading
    shading.show_cavity = shading.show_shadows = False
    shading.show_specular_highlight = False
    shading.single_color = (0.8, 0.8, 0.8)


def aim(
    objects: list[bpy.types.Object], bearing_deg: float, elevation_deg: float
) -> None:
    """Point the scene's orthographic camera at `objects`, framed whole."""
    corners = np.array([tuple(c) for c in box_corners(objects)])
    centre = Vector(tuple((corners.min(axis=0) + corners.max(axis=0)) / 2))
    toward = Vector(earth.direction(bearing_deg, elevation_deg))
    camera = bpy.context.scene.camera
    camera.rotation_mode = "QUATERNION"
    camera.rotation_quaternion = (-toward).to_track_quat("-Z", "Y")
    # Not camera_fit_coords: its ortho scale is wrong on a non-square frame unless the
    # height binds.
    rotation = camera.rotation_quaternion.to_matrix()
    right, up = rotation.col[0], rotation.col[1]
    offsets = [Vector(tuple(c)) - centre for c in corners]
    half_w = max(abs(o.dot(right)) for o in offsets)
    half_h = max(abs(o.dot(up)) for o in offsets)
    aspect = VIEW_PX[0] / VIEW_PX[1]
    radius = max(o.length for o in offsets)
    camera.data.ortho_scale = 2.1 * max(half_w, half_h * aspect)
    camera.data.clip_end = 4 * radius
    camera.location = centre + 2 * radius * toward


def shoot(path: Path, color_type: str, light: str, aa: str, exr: bool) -> None:
    """Render the scene's camera to `path` in Workbench's `color_type` and `light`."""
    sc = bpy.context.scene
    sc.display.shading.color_type = color_type
    sc.display.shading.light = light
    sc.display.render_aa = aa
    settings = sc.render.image_settings
    settings.file_format = "OPEN_EXR" if exr else "PNG"
    settings.color_depth = "32" if exr else "8"
    settings.color_mode = "RGBA"
    sc.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
