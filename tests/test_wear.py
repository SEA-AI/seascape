import bpy
import numpy as np
import pytest

from seascape.blender import wear
from seascape.blender.vessels import Hulls, _vessel
from seascape.config import load
from seascape.physics.thermal import ZONE_NAMES
from tests.scenarios import BASELINE


def _read(mesh: bpy.types.Mesh, name: str) -> np.ndarray:
    attribute = mesh.attributes[name]
    vector = attribute.data_type == "FLOAT_VECTOR"
    out = np.empty(len(attribute.data) * (3 if vector else 1), dtype=np.float32)
    attribute.data.foreach_get("vector" if vector else "value", out)
    return out.reshape(-1, 3) if vector else out


@pytest.fixture
def box() -> tuple[bpy.types.Object, np.ndarray]:
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
    zone = np.where(up, ZONE_NAMES.index("deck"), ZONE_NAMES.index("paint_hull"))
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


def test_a_mirrored_part_keeps_the_normals_blender_renders(box) -> None:
    """Blender takes a face's normal from its winding in world space, which a mirror
    reverses."""
    obj, zone = box
    obj.scale.x = -1.0
    wear.tag([obj], [zone], "steel")
    world = obj.data.copy()
    world.transform(obj.matrix_world)
    rendered = np.empty(3 * len(zone))
    world.polygons.foreach_get("normal", rendered)
    np.testing.assert_allclose(
        _read(obj.data, "wear_normal"), rendered.reshape(-1, 3), atol=1e-6
    )


def test_a_side_is_as_deep_as_it_sits_below_the_sheer(box) -> None:
    obj, zone = box
    wear.tag([obj], [zone], "steel")
    below_m = _read(obj.data, "wear_below_m")
    position_m = _read(obj.data, "wear_position_m")
    z = position_m[[loop.vertex_index for loop in obj.data.loops], 2]
    side = np.repeat(zone == ZONE_NAMES.index("paint_hull"), 4)
    # Sheer at 2 m.
    np.testing.assert_allclose(below_m[side], 2.0 - z[side], atol=1e-5)
    np.testing.assert_allclose(below_m[~side], 0.0, atol=1e-5)  # the deck is its top


def test_a_flared_side_finds_its_sheer_a_patch_outboard() -> None:
    """The sheer sits one patch outboard; a higher point of another zone is ignored."""
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
    deck = zone == ZONE_NAMES.index("deck")
    for i, layer in enumerate(wear.LAYERS):
        got = _read(obj.data, f"wear_{layer}")
        want = 0.0 if layer == "rust" else wear.PROFILE["deck"][i]
        np.testing.assert_allclose(got[deck], want)
    wear.tag([obj], [zone], "steel")
    np.testing.assert_allclose(
        _read(obj.data, "wear_rust")[deck],
        wear.PROFILE["deck"][wear.LAYERS.index("rust")],
    )


def test_cloth_sheds_runoff_and_a_hull_streaks(box) -> None:
    obj, zone = box
    zone[:] = ZONE_NAMES.index("paint_hull")
    zone[0] = ZONE_NAMES.index("fabric")
    wear.tag([obj], [zone], "steel")
    np.testing.assert_array_equal(_read(obj.data, "wear_runoff"), zone != zone[0])


def test_a_skinned_mesh_is_tagged_as_posed(skinned) -> None:
    wear.tag([skinned], [np.zeros(1, dtype=np.int64)], "steel")
    np.testing.assert_allclose(
        _read(skinned.data, "wear_position_m")[:, 2], 5.0, atol=1e-5
    )


def _zodiacs(hulls: Hulls, wears: tuple[float, ...]) -> list[bpy.types.Object]:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    return [
        _vessel(
            "zodiac",
            t_k=None,
            engine_off_min=0.0,
            paint_absorptance=None,
            wear=amount,
            sky=load(BASELINE).sky,
            heat=None,
            hulls=hulls,
        )
        for amount in wears
    ]


def _meshes(anchor: bpy.types.Object) -> list[bpy.types.Object]:
    return [o for o in anchor.children_recursive if o.type == "MESH"]


def test_a_build_tags_a_worn_asset_once_and_prefixes_their_names() -> None:
    clean, worn = _zodiacs(Hulls(frozenset({"zodiac"})), (0.0, 0.6))
    assert (clean["wear"], worn["wear"]) == (0.0, 0.6)
    for anchor in (clean, worn):
        assert all("wear_position_m" in o.data.attributes for o in _meshes(anchor))
        assert all(o.name.startswith("zodiac/") for o in _meshes(anchor))


def test_a_clean_build_is_not_tagged() -> None:
    (clean,) = _zodiacs(Hulls(), (0.0,))
    assert not any("wear_position_m" in o.data.attributes for o in _meshes(clean))
