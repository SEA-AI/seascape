"""Wear attributes written at import, in the fitted hull frame: metres, bow +Y,
waterline z = 0."""

from typing import NamedTuple

import bpy
import numpy as np

from seascape.assets import Material
from seascape.blender import faces
from seascape.physics.thermal import ZONE_NAMES, ZoneName


class Weights(NamedTuple):
    """How much of each layer a zone collects, and whether runoff streaks it."""

    grime: float
    rust: float
    stain: float
    soot: float
    fade: float
    runoff: float


# The layers the masks cover; `runoff` only gates streaks.
LAYERS = Weights._fields[:-1]

# Judgement, all of it. A stain is the waterline's, so only what the sea wets takes
# one; cloth sags and flexes, so water runs off it without leaving a streak.
PROFILE: dict[ZoneName, Weights] = {
    "paint_hull": Weights(1.0, 1.0, 1.0, 0.0, 0.5, 1.0),
    "paint_superstructure": Weights(0.8, 0.7, 0.0, 0.0, 0.5, 1.0),
    "deck": Weights(1.0, 0.6, 0.0, 0.0, 0.3, 1.0),
    "glass": Weights(0.35, 0.0, 0.0, 0.0, 0.0, 1.0),
    "bare_metal": Weights(0.6, 0.6, 0.0, 0.0, 0.0, 1.0),
    "rubber": Weights(0.6, 0.0, 0.6, 0.0, 0.7, 1.0),
    "fabric": Weights(0.8, 0.0, 0.0, 0.0, 0.7, 0.0),
    "funnel": Weights(0.5, 0.5, 0.0, 1.0, 0.3, 1.0),
    "exhaust": Weights(0.4, 0.0, 0.0, 1.0, 0.0, 1.0),
    "engine_cowl": Weights(0.5, 0.0, 0.0, 0.0, 0.7, 1.0),
    "wood_composite": Weights(0.8, 0.0, 1.0, 0.0, 0.6, 1.0),
}


def weights(material: Material) -> np.ndarray:
    """(zones, weights), zones in vocabulary order."""
    table = np.array([PROFILE[z] for z in ZONE_NAMES], dtype=np.float32)
    if material != "steel":
        # Judgement: other hulls' fittings are stainless.
        table[:, Weights._fields.index("rust")] = 0.0
    return table


def below_edge_m(position_m: np.ndarray, zone: np.ndarray) -> np.ndarray:
    """Each point's depth below the highest point of its zone over the same patch of
    plan: for a hull side the sheer, for a house wall its roof edge."""
    # shortcut: a plan grid, not a walk up the surface; a tier stepped back by less
    # than a patch reads its upper tier's edge.
    length_m = float(np.ptp(position_m[:, :2], axis=0).max())
    # Judgement: patches 1/200 of the length, no less than 10 cm, each reaching its
    # eight neighbours, so a side flared out by a patch still finds its sheer.
    patch = np.floor(position_m[:, :2] / max(length_m / 200.0, 0.1)).astype(np.int64)
    # An empty border, so np.roll never wraps a patch into the plan.
    patch -= patch.min(axis=0) - 1
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
    """`objects` already fitted; a skinned mesh is read as posed."""
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
        # A normal takes the inverse transpose; a mirrored part's winding turns it in.
        linear = world[:3, :3]
        normal = normal @ np.linalg.inv(linear) * np.sign(np.linalg.det(linear))
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
        _write(mesh, "wear_position_m", "POINT", position_m)
        _write(mesh, "wear_normal", "FACE", normal)
        for i, name in enumerate(Weights._fields):
            _write(mesh, f"wear_{name}", "FACE", table[zone, i])
        n_loops = len(mesh.loops)
        totals = faces.ints(mesh.polygons, "loop_total", len(zone))
        corners_m.append(position_m[faces.ints(mesh.loops, "vertex_index", n_loops)])
        corner_zones.append(np.repeat(zone, totals))
    corner_m = np.concatenate(corners_m)
    below_m = below_edge_m(corner_m, np.concatenate(corner_zones))
    # Per asset, on every face, so they survive the LWIR import's join.
    length_m = float(np.ptp(corner_m[:, 1]))
    steel = float(material == "steel")
    for obj, at in zip(objects, np.cumsum([len(c) for c in corners_m]), strict=True):
        mesh = obj.data
        n, faces_n = len(mesh.loops), len(mesh.polygons)
        _write(mesh, "wear_below_m", "CORNER", below_m[at - n : at])
        _write(mesh, "wear_length_m", "FACE", np.full(faces_n, length_m))
        _write(mesh, "wear_steel", "FACE", np.full(faces_n, steel))


def _write(mesh: bpy.types.Mesh, name: str, domain: str, values: np.ndarray) -> None:
    vector = values.ndim == 2
    kind = "FLOAT_VECTOR" if vector else "FLOAT"
    if old := mesh.attributes.get(name):
        mesh.attributes.remove(old)
    attribute = mesh.attributes.new(name, kind, domain)
    attribute.data.foreach_set(
        "vector" if vector else "value", values.astype(np.float32).ravel()
    )
