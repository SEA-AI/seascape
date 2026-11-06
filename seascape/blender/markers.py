"""An ArUco marker on a hull's roof, facing the ownship."""

import math

import bpy
import cv2
import numpy as np

from seascape.blender.heat import thermal_skin
from seascape.config import ARUCO_DICTIONARY, Sky

# Judgement: as large as a container's roof carries.
SIZE_M = 2.0
_ARUCO = cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARY)
BITS = _ARUCO.markerSize
# OpenCV finds a marker by its dark border against a light margin.
QUIET = 1
# Judgement: clear of the roof's edge, so the margin reads against the sky.
STAND_M = 0.3
# Judgement.
LIGHT, DARK = 0.85, 0.03
# Judgement: the light modules this far over the air and the dark ones this far under,
# so the marker's edges survive a lossy video.
CONTRAST_K = 30.0


def _paint(name: str, grey: float) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    bsdf = material.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (grey, grey, grey, 1.0)
    return material


def add_marker(
    anchor: bpy.types.Object,
    aruco_id: int,
    toward_deg: float,
    sky: Sky | None,
) -> bpy.types.Object:
    """An upright panel over the middle of the hull, riding its attitude. Call before
    `sky.haze` runs.

    Args:
        anchor: The hull, over its attitude empty.
        aruco_id: Which of `ARUCO_DICTIONARY`'s markers.
        toward_deg: Where the panel faces, clockwise from the hull's bow.
        sky: The air the LWIR modules straddle; None for EO paint.
    """
    module_m = SIZE_M / (BITS + 2)
    image = np.pad(
        cv2.aruco.generateImageMarker(_ARUCO, aruco_id, BITS + 2),
        QUIET,
        constant_values=255,
    )
    side = len(image)
    toward = np.array(
        [math.sin(math.radians(toward_deg)), math.cos(math.radians(toward_deg)), 0.0]
    )
    up = np.array([0.0, 0.0, 1.0])
    # As seen from where it faces, so a face wound right then up faces it.
    right = np.cross(-toward, up)
    centre = up * (anchor["height_m"] + STAND_M + side / 2 * module_m)

    def at(row: int, column: int) -> tuple[float, float, float]:
        """The corner left of `column` and over `row` of the image, whose row 0 is
        the top.
        """
        x, y = column - side / 2, side / 2 - row
        return tuple(centre + module_m * (x * right + y * up))

    verts = [at(row, column) for row in range(side + 1) for column in range(side + 1)]

    def corner(row: int, column: int) -> int:
        return row * (side + 1) + column

    # One quad a module, wound right then up.
    cells = [(row, column) for row in range(side) for column in range(side)]
    faces = [
        (
            corner(row + 1, column),
            corner(row + 1, column + 1),
            corner(row, column + 1),
            corner(row, column),
        )
        for row, column in cells
    ]

    name = f"{anchor.name}_aruco_{aruco_id}"
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    for shade, grey, over_air_k in (
        ("light", LIGHT, CONTRAST_K),
        ("dark", DARK, -CONTRAST_K),
    ):
        mesh.materials.append(
            _paint(f"{name}_{shade}", grey)
            if sky is None
            else thermal_skin(f"{name}_{shade}", sky.t_air_k + over_air_k, sky)
        )
    for polygon, (row, column) in zip(mesh.polygons, cells, strict=True):
        polygon.material_index = int(image[row, column] < 128)
    panel = bpy.data.objects.new(name, mesh)
    panel["aruco_id"] = aruco_id
    bpy.context.collection.objects.link(panel)
    (attitude,) = anchor.children
    panel.parent = attitude
    return panel
