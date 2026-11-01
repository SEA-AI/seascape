"""An asset's meshes: imported, joined, fitted to its size, and measured."""

import contextlib
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import get_args

import bpy
import numpy as np
from mathutils import Matrix, Vector

from seascape.assets import Asset, Turbine, Zones, fetch, manifest
from seascape.blender import faces, wear
from seascape.blender.blend import (
    yaw,
)
from seascape.config import (
    Band,
)
from seascape.physics import thermal


def _joined(
    parts: list[bpy.types.Object], frame: Matrix | None = None
) -> bpy.types.Object:
    """Every mesh under `parts` as one object at `frame`, its vertices in metres about
    it; the default, the fitted origin, puts the waterline at z = 0."""
    everything = [o for part in parts for o in [part, *part.children_recursive]]
    meshes = [o for o in everything if o.type == "MESH"]
    others = [o for o in everything if o.type != "MESH"]
    bpy.context.view_layer.update()
    # A skinned mesh takes its pose from an armature modifier, which the join keeps
    # pointing at an armature removed below: bake the pose in first.
    depsgraph = bpy.context.evaluated_depsgraph_get()
    posed = {
        obj: bpy.data.meshes.new_from_object(
            obj.evaluated_get(depsgraph),
            preserve_all_data_layers=True,
            depsgraph=depsgraph,
        )
        for obj in meshes
        if obj.modifiers
    }
    for obj, mesh in posed.items():
        rest = obj.data
        obj.modifiers.clear()
        obj.data = mesh
        if not rest.users:
            bpy.data.meshes.remove(rest)
    with bpy.context.temp_override(
        active_object=meshes[0], selected_editable_objects=meshes
    ):
        bpy.ops.object.join()
    hull = meshes[0]
    frame = Matrix.Identity(4) if frame is None else frame
    baked = frame.inverted() @ hull.matrix_world
    hull.parent = None
    hull.matrix_parent_inverse.identity()
    hull.data.transform(baked)
    if baked.determinant() < 0:  # a mirrored part turns inside out
        hull.data.flip_normals()
    hull.matrix_basis = frame
    for obj in others:
        bpy.data.objects.remove(obj)
    return hull


def _face_zones(name: str, found: faces.Parts, zones: Zones | None) -> list[np.ndarray]:
    """Per object, each face's index into the zone vocabulary. Unlabelled, every face
    is `paint_hull`."""
    if zones is None:
        paint = list(get_args(thermal.ZoneName.__value__)).index("paint_hull")
        return [np.full(len(obj.data.polygons), paint) for obj in found.objects]
    # A name another object in the file already took renumbers the islands.
    faces.check(name, found, zones)
    return faces.face_zones(found, zones)


def _zone_slots(found: faces.Parts, per_face: list[np.ndarray]) -> None:
    """Give each face its zone as a material slot, one placeholder per zone, which the
    join merges.

    The asset's own materials go: they are albedo, which says nothing about 8-14 um,
    and a later import sharing their names would be renamed past its zone labels.
    """
    vocabulary = list(get_args(thermal.ZoneName.__value__))
    originals = {s.material for obj in found.objects for s in obj.material_slots}
    for obj, zone in zip(found.objects, per_face, strict=True):
        present, index = np.unique(zone, return_inverse=True)
        obj.data.materials.clear()
        for z in present:
            name = f"zone_{vocabulary[z]}"
            placeholder = bpy.data.materials.get(name) or bpy.data.materials.new(name)
            obj.data.materials.append(placeholder)
        obj.data.polygons.foreach_set("material_index", index.astype(np.int32))
    for material in originals - {None}:
        bpy.data.materials.remove(material)


def _unslotted(joined: list[bpy.types.Object]) -> None:
    """Record each joined object's zone per slot in its `["zones"]` and empty the
    slots; each copy fills them with its own skin."""
    placeholders = set()
    for obj in joined:
        mesh = obj.data
        obj["zones"] = [m.name.removeprefix("zone_") for m in mesh.materials]
        placeholders.update(mesh.materials)
        for i in range(len(mesh.materials)):
            mesh.materials[i] = None
    for material in placeholders:
        bpy.data.materials.remove(material)


def box_corners(
    objects: Iterable[bpy.types.Object], into: Matrix | None = None
) -> list[Vector]:
    """Corners of the axis-aligned box around the vertices of the meshes in
    `objects`, in the frame `into` maps world space to (default: world).

    Vertices, not `bound_box`: a part turned off its parent's axes carries its own box
    turned with it, which can be far wider than the part.
    """
    depsgraph = bpy.context.evaluated_depsgraph_get()
    points = []
    for obj in objects:
        if obj.type != "MESH":
            continue
        # Evaluated, so an armature's pose and the modifiers move the vertices.
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        co = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", co)
        evaluated.to_mesh_clear()
        if not co.size:
            continue
        to = np.array((into or Matrix()) @ obj.matrix_world)
        moved = co.reshape(-1, 3) @ to[:3, :3].T + to[:3, 3]
        points += [moved.min(axis=0), moved.max(axis=0)]
    if not points:
        return []
    low, high = np.min(points, axis=0), np.max(points, axis=0)
    return [
        Vector((x, y, z))
        for x in (low[0], high[0])
        for y in (low[1], high[1])
        for z in (low[2], high[2])
    ]


def _fit(corners: Iterable[Vector], asset: Asset) -> Matrix:
    """Bow to +Y, scaled to the manifest size, centred, keel at the draught.

    Turned first so a hull's length is measured bow to stern whatever the authored axis.
    """
    turn = Matrix.Rotation(yaw(-asset.bow_deg), 4, "Z")
    axes = list(zip(*(turn @ c for c in corners), strict=True))
    low, high = Vector([min(a) for a in axes]), Vector([max(a) for a in axes])
    scale = asset.scale(tuple(high - low))
    low, high = low * scale, high * scale
    east, north = asset.centre(tuple(low), tuple(high))
    return (
        Matrix.Translation((-east, -north, -low.z - asset.draught_m))
        @ Matrix.Scale(scale, 4)
        @ turn
    )


def mesh_objects(parts: Iterable[bpy.types.Object]) -> list[bpy.types.Object]:
    return [
        obj
        for part in parts
        for obj in [part, *part.children_recursive]
        if obj.type == "MESH"
    ]


_CACHE_FORMAT = 2  # raise to orphan every cache already written


def _load(path: Path) -> set[bpy.types.Object]:
    """Import a mesh file; returns what it added. Caches the parse as a .blend beside
    the file and appends from it on later calls."""
    stat = path.stat()
    version = ".".join(map(str, bpy.app.version))
    imported = path.with_name(
        f"{path.name}.{stat.st_size}-{stat.st_mtime_ns}.{version}.{_CACHE_FORMAT}.blend"
    )
    if imported.exists():
        try:
            with bpy.data.libraries.load(str(imported)) as (found, into):
                into.objects = found.objects
            if None in into.objects:  # Blender leaves None for what it cannot read
                raise RuntimeError(f"{imported} loaded partly")
        except (OSError, RuntimeError):
            with contextlib.suppress(OSError):
                imported.unlink(missing_ok=True)
        else:
            for obj in into.objects:
                bpy.context.scene.collection.objects.link(obj)
            # The fit reads matrix_world, which an append leaves stale.
            bpy.context.view_layer.update()
            return set(into.objects)
    # The cache and the zones key on the mesh file's own names: the .blend's datablocks
    # step aside for the parse, then take their names back from the import.
    held: dict[bpy.types.ID, str] = {}
    try:
        for datablock in bpy.data.user_map():
            name = datablock.name
            # A linked name is its library's; the built-in font's cannot be set.
            with contextlib.suppress(AttributeError):
                if datablock.library is None:
                    datablock.name = uuid.uuid4().hex
                    held[datablock] = name
        objects = _parse(path)
        # A killed write must not leave a truncated file under the cache name.
        part = imported.with_name(f"{imported.name}.{uuid.uuid4().hex}.part")
        try:
            bpy.data.libraries.write(
                str(part), objects, path_remap="ABSOLUTE", compress=False
            )
            part.replace(imported)
        except (OSError, RuntimeError):  # a read-only or full cache is only slower
            part.unlink(missing_ok=True)
    finally:
        for datablock, name in held.items():
            datablock.rename(name, mode="ALWAYS")
    return objects


def _parse(path: Path) -> set[bpy.types.Object]:
    before = set(bpy.data.objects)
    gltf = bpy.ops.import_scene.gltf
    importer = {".fbx": bpy.ops.import_scene.fbx, ".glb": gltf, ".gltf": gltf}
    importer[path.suffix](filepath=str(path))
    imported = set(bpy.data.objects) - before
    # A Poly glTF brings its viewer's camera and lights, which would light the scene.
    # glTF nodes outside the file's scene land in an excluded collection: matrix_world
    # never updates there, and copies would render.
    shown = bpy.context.view_layer.objects
    # A skinned glTF's bones are drawn with a sphere the importer adds to the scene,
    # where it renders and enters the fit.
    shapes = {
        bone.custom_shape
        for o in imported
        if o.type == "ARMATURE"
        for bone in o.pose.bones
    }
    for obj in [
        o
        for o in imported
        if o.type in {"CAMERA", "LIGHT"} or o.name not in shown or o in shapes
    ]:
        imported.remove(obj)
        bpy.data.objects.remove(obj)
    return imported


def import_asset(name: str, band: Band, worn: bool = False) -> list[bpy.types.Object]:
    """Import an asset and fit it; returns the root parts. `worn` tags it with where
    it wears.

    An asset arrives in its author's units, off-origin, in many parts.
    """
    imported = _load(fetch(name))
    asset = manifest()[name]
    # Measure everything, move the roots. An FBX keeps meshes under empties, and
    # measuring only the roots would leave them out of the fit.
    parts = [o for o in imported if o.parent is None]

    corners = box_corners(imported)
    fit = _fit(corners, asset)
    for part in parts:
        part.matrix_world = fit @ part.matrix_world
    rotor = None
    if isinstance(asset, Turbine):
        named = [o for o in imported if o.name == asset.rotor.node]
        if len(named) != 1:
            raise ValueError(f"{name}: {len(named)} objects named {asset.rotor.node!r}")
        (rotor,) = named
    # Fitted bow to +Y, so the beam is the width along x and the length along y.
    fitted = [fit @ c for c in corners]
    if band == "ir" or worn:
        found = faces.found(mesh_objects(parts))
        per_face = _face_zones(name, found, asset.zones)
        if worn:
            wear.tag(found.objects, per_face, asset.material)
    if band == "ir":
        _zone_slots(found, per_face)
        if rotor is not None:
            # Joined apart from the rest, so it still turns about its hub.
            bpy.context.view_layer.update()
            hub = rotor.matrix_world.copy()
            rotor.parent = None
            rotor.matrix_world = hub
            # Without the fit's scale, so the join leaves its vertices in metres.
            rotor = _joined([rotor], hub.normalized())
        parts = [_joined(parts)]
        # Frees the import's object name, so the next asset's objects keep theirs,
        # which number its islands.
        parts[0].name = parts[0].data.name = f"{name}_hull"
        if rotor is not None:
            rotor.name = rotor.data.name = f"{name}_rotor"
            parts.append(rotor)
        _unslotted(parts)
    if rotor is not None:
        rotor["rotor"] = True
        # The rest angle, read before a copy is animated; YXZ euler is Rz @ Rx @ Ry,
        # the spin about the shaft innermost.
        rotor.rotation_mode = "YXZ"
        rotor["rest_rad"] = rotor.rotation_euler.y
    parts[0]["beam_m"] = max(c.x for c in fitted) - min(c.x for c in fitted)
    parts[0]["length_m"] = max(c.y for c in fitted) - min(c.y for c in fitted)
    # Above the waterline, which the fit puts at z = 0.
    parts[0]["height_m"] = max(c.z for c in fitted)
    return parts


def free_names(name: str, parts: list[bpy.types.Object]) -> None:
    """Prefix an EO import's objects and materials with its asset's name, so the next
    import keeps the names its zones key on."""
    everything = [o for part in parts for o in [part, *part.children_recursive]]
    for material in {s.material for o in everything for s in o.material_slots}:
        if material is not None:
            material.name = f"{name}/{material.name}"
    for obj in everything:
        obj.name = f"{name}/{obj.name}"


def _images(tree: bpy.types.NodeTree) -> set[bpy.types.Image]:
    found = set()
    for node in tree.nodes:
        if node.type == "TEX_IMAGE" and node.image:
            found.add(node.image)
        elif node.type == "GROUP" and node.node_tree:
            found |= _images(node.node_tree)
    return found


def measure(path: Path) -> tuple[int, tuple[int, ...]]:
    """A mesh file's triangles and the longest side of each image, as imported."""
    return measured(str(path), [obj for obj in _load(path) if obj.type == "MESH"])


def measured(name: str, meshes: list[bpy.types.Object]) -> tuple[int, tuple[int, ...]]:
    """`measure` of `name`'s meshes, already imported."""
    for mesh in meshes:
        mesh.data.calc_loop_triangles()
    images = set().union(
        *(
            _images(material.node_tree)
            for mesh in meshes
            for material in mesh.data.materials
            if material and material.node_tree
        )
    )
    if missing := sorted(image.name for image in images if 0 in image.size):
        raise ValueError(f"{name}: no pixels in {missing}")
    return (
        sum(len(mesh.data.loop_triangles) for mesh in meshes),
        tuple(sorted((max(image.size) for image in images), reverse=True)),
    )
