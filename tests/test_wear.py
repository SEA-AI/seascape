from typing import get_args

import bpy
import numpy as np
import pytest

from seascape.blender import wear
from seascape.physics.thermal import ZoneName

ZONES = list(get_args(ZoneName.__value__))


def _read(mesh: bpy.types.Mesh, name: str) -> np.ndarray:
    attribute = mesh.attributes[name]
    vector = attribute.data_type == "FLOAT_VECTOR"
    out = np.empty(len(attribute.data) * (3 if vector else 1), dtype=np.float32)
    attribute.data.foreach_get("vector" if vector else "value", out)
    return out.reshape(-1, 3) if vector else out


@pytest.fixture
def box() -> tuple[bpy.types.Object, np.ndarray]:
    """A 4 x 10 x 3 m box, its top a deck at 2 m, its sides hull from 2 m down to
    1 m below the waterline; authored at the origin, placed 10 m east and turned a
    quarter, as the fit leaves an import."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    mesh = bpy.data.meshes.new("box")
    corners = [(x, y, z) for x in (-2, 2) for y in (-5, 5) for z in (-1, 2)]
    sides = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4)]
    mesh.from_pydata(corners, [], [*sides, (1, 5, 7, 3)])
    obj = bpy.data.objects.new("box", mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = (10.0, 0.0, 0.0)
    obj.rotation_euler.z = np.pi / 2
    normals = np.empty(3 * len(mesh.polygons))
    mesh.polygons.foreach_get("normal", normals)
    up = normals.reshape(-1, 3)[:, 2] > 0.5
    zone = np.where(up, ZONES.index("deck"), ZONES.index("paint_hull"))
    return obj, zone


def test_a_point_is_tagged_where_it_sits_in_the_hull_frame(box) -> None:
    obj, zone = box
    wear.tag([obj], [zone], "steel")
    position_m = _read(obj.data, "wear_position_m")
    corner = np.array(obj.matrix_world) @ np.append(obj.data.vertices[0].co, 1.0)
    np.testing.assert_allclose(position_m[0], corner[:3], atol=1e-5)
    # Turned a quarter: the box's 10 m now runs east, from 5 m to 15 m.
    np.testing.assert_allclose(position_m.min(axis=0), (5.0, -2.0, -1.0), atol=1e-5)
    np.testing.assert_allclose(position_m.max(axis=0), (15.0, 2.0, 2.0), atol=1e-5)


def test_a_face_normal_turns_with_the_fit(box) -> None:
    obj, zone = box
    wear.tag([obj], [zone], "steel")
    normal = _read(obj.data, "wear_normal")
    # The faces facing +y as authored face west once turned.
    authored = np.empty(3 * len(zone))
    obj.data.polygons.foreach_get("normal", authored)
    facing = authored.reshape(-1, 3)[:, 1] > 0.5
    np.testing.assert_allclose(normal[facing], [[-1.0, 0.0, 0.0]], atol=1e-6)


def test_a_side_is_as_deep_as_it_sits_below_the_sheer(box) -> None:
    obj, zone = box
    wear.tag([obj], [zone], "steel")
    below_m = _read(obj.data, "wear_below_m")
    position_m = _read(obj.data, "wear_position_m")
    z = position_m[[loop.vertex_index for loop in obj.data.loops], 2]
    side = np.repeat(zone == ZONES.index("paint_hull"), 4)
    # The sheer is at 2 m: the side's top corners are at its edge, the bottom 3 m down.
    np.testing.assert_allclose(below_m[side], 2.0 - z[side], atol=1e-5)
    np.testing.assert_allclose(below_m[~side], 0.0, atol=1e-5)  # the deck is its top


def test_a_flared_side_finds_its_sheer_a_patch_outboard() -> None:
    """A waterline point under a sheer flared out by one patch, of another zone's
    point higher still."""
    position_m = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 3.0], [0.0, 0.0, 9.0]])
    patch_m = 100.0 / 200.0
    position_m = np.vstack([position_m, [[100.0, 0.0, 0.0]]])  # sets the length
    position_m[1, 0] = patch_m
    zone = np.array([0, 0, 1, 0])
    np.testing.assert_allclose(
        wear.below_edge_m(position_m, zone), [3.0, 0.0, 0.0, 0.0]
    )


def test_each_face_weighs_its_zone_and_only_steel_rusts(box) -> None:
    obj, zone = box
    wear.tag([obj], [zone], "grp")
    deck = zone == ZONES.index("deck")
    for i, layer in enumerate(wear.LAYERS):
        got = _read(obj.data, f"wear_{layer}")
        want = 0.0 if layer == "rust" else wear.PROFILE["deck"][i]
        np.testing.assert_allclose(got[deck], want)
    wear.tag([obj], [zone], "steel")
    np.testing.assert_allclose(
        _read(obj.data, "wear_rust")[deck],
        wear.PROFILE["deck"][wear.LAYERS.index("rust")],
    )


def test_a_skinned_mesh_is_tagged_as_posed(skinned) -> None:
    wear.tag([skinned], [np.zeros(1, dtype=np.int64)], "steel")
    np.testing.assert_allclose(
        _read(skinned.data, "wear_position_m")[:, 2], 5.0, atol=1e-5
    )
