"""An imported asset's parts: each face's material and island, and the zone they take.

A part is a material slot, or an island: the faces of one material in one object that
touch at a vertex, and also at a UV where the material is textured.
"""

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, NamedTuple, get_args

import bpy
import numpy as np

from seascape.assets import Zones
from seascape.blender import blend
from seascape.physics.thermal import ZoneName


@dataclass(frozen=True)
class Island:
    id: int
    object: str
    material: str
    first_face: int
    faces: int


@dataclass
class Parts:
    """An imported asset's faces, each with its material and island."""

    objects: list[bpy.types.Object]  # sorted by name
    materials: list[str]  # sorted by name
    face_material: list[np.ndarray]  # per object, an index into `materials`
    face_island: list[np.ndarray]  # per object, an index into `islands`
    islands: list[Island]

    @property
    def island_digest(self) -> str:
        """A fingerprint of the islands' numbering, which their count alone misses."""
        body = [(i.object, i.first_face, i.faces) for i in self.islands]
        return hashlib.sha256(json.dumps(body).encode()).hexdigest()[:16]


def ints(collection: Any, attribute: str, n: int) -> np.ndarray:
    out = np.empty(n, dtype=np.int64)
    collection.foreach_get(attribute, out)
    return out


def floats(collection: Any, attribute: str, n: int) -> np.ndarray:
    out = np.empty(n, dtype=np.float64)
    collection.foreach_get(attribute, out)
    return out


def uv(mesh: bpy.types.Mesh) -> np.ndarray:
    """Each loop's UV, (loops, 2), from the layer an image node with no Vector link
    reads: the one marked for render, not the one active for editing."""
    layer = next(u for u in mesh.uv_layers if u.active_render)
    return floats(layer.data, "uv", 2 * len(mesh.loops)).reshape(-1, 2)


def _components(n: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Each node's connected component, as the smallest-numbered node reached in it."""
    label = np.arange(n)
    while True:
        least = np.minimum(label[a], label[b])
        new = label.copy()
        np.minimum.at(new, a, least)
        np.minimum.at(new, b, least)
        new = new[new]
        if np.array_equal(new, label):
            return label
        label = new


def _face_components(
    mesh: bpy.types.Mesh, material: np.ndarray, textured: np.ndarray
) -> np.ndarray:
    """Per face, a label shared by the faces of its material that touch it: at a
    vertex, and where the material is `textured`, at a UV too, so that an atlas splits
    where its texture does.

    By position, not vertex index: glTF splits a vertex at every UV and normal seam, so
    Blender's own linked and loose parts stop at each one.
    """
    n_faces, n_loops = len(mesh.polygons), len(mesh.loops)
    co = floats(mesh.vertices, "co", 3 * len(mesh.vertices)).reshape(-1, 3)
    low, high = co.min(axis=0), co.max(axis=0)
    span = max(float(np.linalg.norm(high - low)), 1e-12)
    start = ints(mesh.polygons, "loop_start", n_faces)
    total = ints(mesh.polygons, "loop_total", n_faces)
    face = np.repeat(np.arange(n_faces), total)
    # Positions to a millionth of the mesh's diagonal, UVs to a millionth of the
    # texture: coincident to any modeller.
    key = np.zeros((n_loops, 6), dtype=np.int64)
    key[:, :3] = np.round((co - low) / span * 1e6)[
        ints(mesh.loops, "vertex_index", n_loops)
    ]
    if mesh.uv_layers:
        key[:, 3:5] = np.round(uv(mesh) * 1e6) * textured[material[face], None]
    # A vertex per material, so slots that meet stay apart.
    key[:, 5] = material[face]
    # One bytes value per row groups as `axis=0` does, in another order: islands are
    # numbered by their first face, so only the grouping is read.
    rows = key.view(np.dtype((np.void, key.itemsize * key.shape[1]))).ravel()
    node = np.unique(rows, return_inverse=True)[1]
    following = np.arange(n_loops) + 1
    following[start + total - 1] = start
    label = _components(int(node.max()) + 1, node, node[following])
    return label[node[start]]


def found(meshes: list[bpy.types.Object]) -> Parts:
    """An import's mesh objects, their faces each given a material and an island.

    Objects and materials are named as a fresh import names them, which is what zones
    key on: a build importing several assets renames on collision. Run it on one
    import before any join or subdivision, which renumber faces.
    """
    objects = sorted(meshes, key=lambda o: o.name)
    for obj in objects:
        # A shared mesh would give two objects' faces one colour.
        if obj.data.users > 1:
            obj.data = obj.data.copy()
    slot_names = [
        [slot.material.name if slot.material else "" for slot in obj.material_slots]
        or [""]
        for obj in objects
    ]
    materials = sorted({n for names in slot_names for n in names})
    textured = np.array([texture(n) is not None for n in materials])
    face_material, face_island, islands = [], [], []
    for obj, names in zip(objects, slot_names, strict=True):
        mesh = obj.data
        index = ints(mesh.polygons, "material_index", len(mesh.polygons))
        lookup = np.array([materials.index(n) for n in names])
        material = lookup[np.minimum(index, len(names) - 1)]
        label = _face_components(mesh, material, textured)
        _, first, inverse = np.unique(label, return_index=True, return_inverse=True)
        order = np.argsort(first)
        rank = np.empty_like(order)
        rank[order] = np.arange(len(order))
        face_island.append(len(islands) + rank[inverse])
        counts = np.bincount(inverse, minlength=len(first))
        islands += [
            Island(
                len(islands) + r,
                obj.name,
                materials[material[first[k]]],
                int(first[k]),
                int(counts[k]),
            )
            for r, k in enumerate(order)
        ]
        face_material.append(material)
    return Parts(objects, materials, face_material, face_island, islands)


def face_zones(parts: Parts, zones: Zones) -> list[np.ndarray]:
    """Per object, each face's index into the zone vocabulary."""
    vocabulary = list(get_args(ZoneName.__value__))
    by_material = np.array(
        [vocabulary.index(zones.materials[m]) for m in parts.materials]
    )
    by_island = np.array(
        [by_material[parts.materials.index(i.material)] for i in parts.islands]
    )
    means: dict[str, np.ndarray] = {}
    material = np.array([i.material for i in parts.islands])
    # Applied last to first, so the first rule an island matches wins.
    for rule in reversed(zones.rules):
        hit = material == rule.material
        for channel, (low, high) in rule.ranges.items():
            if channel not in means:
                means[channel] = island_means(parts, CHANNELS[channel])
            hit &= (means[channel] >= low) & (means[channel] <= high)
        by_island[hit] = vocabulary.index(rule.zone)
    for i, zone in zones.islands.items():
        by_island[i] = vocabulary.index(zone)
    return [by_island[island] for island in parts.face_island]


# A rule's channels, by the Principled BSDF input each reads.
CHANNELS = {"metallic": "Metallic", "roughness": "Roughness", "alpha": "Alpha"}


def island_means(parts: Parts, socket: str) -> np.ndarray:
    """Each island's mean of a Principled BSDF input over its faces, read at each
    face's mean UV."""
    total = np.zeros(len(parts.islands))
    count = np.zeros(len(parts.islands))
    images: dict[str, np.ndarray] = {}  # objects share an atlas: read it once
    for obj, material, island in zip(
        parts.objects, parts.face_material, parts.face_island, strict=True
    ):
        value = np.zeros(len(island))
        for m, name in enumerate(parts.materials):
            on = np.flatnonzero(material == m)
            if len(on):
                value[on] = _input(obj.data, name, socket, on, images)
        total += np.bincount(island, value, len(parts.islands))
        count += np.bincount(island, minlength=len(parts.islands))
    return total / np.maximum(count, 1)


def _input(
    mesh: bpy.types.Mesh,
    name: str,
    socket: str,
    faces: np.ndarray,
    images: dict[str, np.ndarray],
) -> np.ndarray:
    """`name`'s `socket` at each of `faces`: the nearest texel to the face's mean UV,
    red where the link is colour, times the factors on its way; or its flat value."""
    shader = bsdf(bpy.data.materials.get(name)) if name else None
    if shader is None:
        return np.zeros(len(faces))
    found = source(shader.inputs[socket])
    if found is None or not mesh.uv_layers:
        return np.full(len(faces), float(shader.inputs[socket].default_value))
    if found.image.name not in images:
        images[found.image.name] = blend.pixels(found.image)
    pixels = images[found.image.name]
    height, width = pixels.shape[:2]
    n_faces = len(mesh.polygons)
    start = ints(mesh.polygons, "loop_start", n_faces)[faces]
    total = ints(mesh.polygons, "loop_total", n_faces)[faces]
    first = np.cumsum(total) - total
    loops = np.repeat(start - first, total) + np.arange(total.sum())
    centre = np.add.reduceat(uv(mesh)[loops], first) / total[:, None]
    x = np.clip((centre[:, 0] % 1.0 * width).astype(int), 0, width - 1)
    y = np.clip((centre[:, 1] % 1.0 * height).astype(int), 0, height - 1)
    return pixels[y, x, found.channel or 0] * found.scale


def bsdf(material: bpy.types.Material) -> Any:
    if material is None or material.node_tree is None:
        return None
    return next(
        (n for n in material.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None
    )


class Source(NamedTuple):
    image: bpy.types.Image
    channel: int | None  # None for colour
    node: Any
    scale: float  # what the shader multiplies the texel by


def source(socket: Any) -> Source | None:
    """The first image upstream of `socket`, following each node's first linked input
    but not into groups, and the factors a Math node multiplies it by on the way: glTF
    imports a metallic or roughness factor as one."""
    if not socket.is_linked:
        return None
    link = socket.links[0]
    node, out = link.from_node, link.from_socket.name
    if node.type == "TEX_IMAGE":
        if node.image is None:
            return None
        # Every reader samples the render UV layer as is.
        if node.inputs["Vector"].is_linked:
            raise ValueError(
                f"{node.name}: its UVs come through a node no reader follows"
            )
        return Source(node.image, 3 if out == "Alpha" else None, node, 1.0)
    if node.type == "MATH" and node.operation != "MULTIPLY":
        raise ValueError(f"{node.name}: a Math {node.operation} on a texture's way")
    channel = {"Red": 0, "Green": 1, "Blue": 2}.get(out)
    for socket_in in node.inputs:
        upstream = source(socket_in)
        if upstream is not None:
            if node.type == "MATH":
                factors = [i.default_value for i in node.inputs[:2] if not i.is_linked]
                upstream = upstream._replace(scale=upstream.scale * math.prod(factors))
            if channel is not None:
                upstream = upstream._replace(channel=channel)
            return upstream
    return None


def texture(name: str) -> bpy.types.Image | None:
    """The image a material's base colour reads, if any."""
    shader = bsdf(bpy.data.materials.get(name)) if name else None
    found = source(shader.inputs["Base Color"]) if shader is not None else None
    return found.image if found is not None else None


def check(name: str, found: Parts, zones: Zones) -> None:
    """Refuse zones that name a material the import lacks, leave one of its materials
    out, or number its islands differently."""
    if missing := sorted(set(zones.materials) - set(found.materials)):
        raise ValueError(f"{name}: no material {missing} in the import")
    if unzoned := sorted(set(found.materials) - set(zones.materials)):
        raise ValueError(f"{name}: no zone for {unzoned}")
    if zones.island_count is None:
        return
    got = (len(found.islands), found.island_digest)
    if got != (zones.island_count, zones.island_digest):
        raise ValueError(
            f"{name}: {got[0]} islands, digest {got[1]}; assets.toml has "
            f"{zones.island_count}, {zones.island_digest}: the import's faces moved"
        )
