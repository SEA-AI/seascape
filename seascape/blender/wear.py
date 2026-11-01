"""Where an asset wears, written on its meshes at import for the wear masks to read.

Per face, each layer's weight for the face's zone. Per vertex, its place in the
fitted hull frame: metres, bow to +Y, the waterline at z = 0. Per face corner, its
depth below the edge above it, where runoff starts.
"""

from typing import get_args

import bpy
import numpy as np

from seascape.assets import Material
from seascape.blender import faces
from seascape.physics.thermal import ZoneName

LAYERS = ("grime", "rust", "stain", "soot", "fade")

# Judgement: how much of each layer a zone collects, in the order of LAYERS. A stain
# is the waterline's, so only what the sea wets takes one.
PROFILE: dict[ZoneName, tuple[float, float, float, float, float]] = {
    "paint_hull": (1.0, 1.0, 1.0, 0.0, 0.5),
    "paint_superstructure": (0.8, 0.7, 0.0, 0.0, 0.5),
    "deck": (1.0, 0.6, 0.0, 0.0, 0.3),
    "glass": (0.35, 0.0, 0.0, 0.0, 0.0),
    "bare_metal": (0.6, 0.6, 0.0, 0.0, 0.0),
    "rubber": (0.6, 0.0, 0.6, 0.0, 0.7),
    "fabric": (0.8, 0.0, 0.0, 0.0, 0.7),
    "funnel": (0.5, 0.5, 0.0, 1.0, 0.3),
    "exhaust": (0.4, 0.0, 0.0, 1.0, 0.0),
    "engine_cowl": (0.5, 0.0, 0.0, 0.0, 0.7),
    "wood_composite": (0.8, 0.0, 1.0, 0.0, 0.6),
}
_VOCABULARY: list[ZoneName] = list(get_args(ZoneName.__value__))


def weights(material: Material) -> np.ndarray:
    """Each zone's layer weights, (zones, layers), in the zone vocabulary's order."""
    table = np.array([PROFILE[z] for z in _VOCABULARY], dtype=np.float32)
    if material != "steel":
        table[:, LAYERS.index("rust")] = 0.0  # fittings on other hulls are stainless
    return table


def below_edge_m(position_m: np.ndarray, zone: np.ndarray) -> np.ndarray:
    """Each point's depth below the highest point of its zone over the same patch of
    plan: for a hull side the sheer, for a house wall its roof edge.

    Patches are 1/200 of the asset's length, no less than 10 cm, and each reaches its
    eight neighbours, so a side that flares out by a patch still finds its sheer.
    """
    # shortcut: a plan grid, not a walk up the surface; a tier stepped back by less
    # than a patch reads its upper tier's edge.
    length_m = float(np.ptp(position_m[:, :2], axis=0).max())
    patch = np.floor(position_m[:, :2] / max(length_m / 200.0, 0.1)).astype(np.int64)
    patch -= patch.min(axis=0) - 1  # a border of empty patches round the plan
    rows, cols = patch.max(axis=0) + 2
    cell = (zone * rows + patch[:, 0]) * cols + patch[:, 1]
    order = np.argsort(cell, kind="stable")
    first = np.flatnonzero(np.diff(cell[order], prepend=-1))
    top = np.full((zone.max() + 1) * rows * cols, -np.inf)
    top[cell[order][first]] = np.maximum.reduceat(position_m[order, 2], first)
    top = top.reshape(-1, rows, cols)
    reach = top.copy()
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            np.maximum(reach, np.roll(top, (di, dj), axis=(1, 2)), out=reach)
    return reach.ravel()[cell] - position_m[:, 2]


def tag(
    objects: list[bpy.types.Object], zones: list[np.ndarray], material: Material
) -> None:
    """Write the wear attributes on fitted `objects`, given each face's zone.

    Objects whose world is the hull frame: at import, after the fit. A skinned mesh is
    read as posed, where it renders.
    """
    bpy.context.view_layer.update()  # the fit leaves matrix_world stale
    depsgraph = bpy.context.evaluated_depsgraph_get()
    table = weights(material)
    corners_m, corner_zones = [], []
    for obj, zone in zip(objects, zones, strict=True):
        mesh = obj.data
        posed = obj.evaluated_get(depsgraph).to_mesh() if obj.modifiers else mesh
        if len(posed.vertices) != len(mesh.vertices):
            raise ValueError(f"{obj.name}: its modifiers change its vertex count")
        world = np.array(obj.matrix_world)
        position_m = (
            faces.floats(posed.vertices, "co", 3 * len(mesh.vertices)).reshape(-1, 3)
            @ world[:3, :3].T
            + world[:3, 3]
        )
        normal = faces.floats(posed.polygons, "normal", 3 * len(zone)).reshape(-1, 3)
        if posed is not mesh:
            obj.evaluated_get(depsgraph).to_mesh_clear()
        # A normal takes the inverse transpose.
        normal = normal @ np.linalg.inv(world[:3, :3])
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
        _write(mesh, "wear_position_m", "POINT", position_m)
        _write(mesh, "wear_normal", "FACE", normal)
        for i, layer in enumerate(LAYERS):
            _write(mesh, f"wear_{layer}", "FACE", table[zone, i])
        n_loops = len(mesh.loops)
        totals = faces.ints(mesh.polygons, "loop_total", len(zone))
        corners_m.append(position_m[faces.ints(mesh.loops, "vertex_index", n_loops)])
        corner_zones.append(np.repeat(zone, totals))
    below_m = below_edge_m(np.concatenate(corners_m), np.concatenate(corner_zones))
    for obj, at in zip(objects, np.cumsum([len(c) for c in corners_m]), strict=True):
        n = len(obj.data.loops)
        _write(obj.data, "wear_below_m", "CORNER", below_m[at - n : at])


def _write(mesh: bpy.types.Mesh, name: str, domain: str, values: np.ndarray) -> None:
    vector = values.ndim == 2
    kind = "FLOAT_VECTOR" if vector else "FLOAT"
    attribute = mesh.attributes.get(name) or mesh.attributes.new(name, kind, domain)
    attribute.data.foreach_set(
        "vector" if vector else "value", values.astype(np.float32).ravel()
    )
