import bpy
import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--shard",
        metavar="I/N",
        help="run every Nth test, from the Ith: CI's jobs split the suite this way",
    )


def _shard(config: pytest.Config) -> tuple[int, int] | None:
    shard = config.getoption("shard")
    if shard is None:
        return None
    index, _, count = shard.partition("/")
    if not (index.isdigit() and count.isdigit() and 1 <= int(index) <= int(count)):
        raise pytest.UsageError(f"--shard {shard}: give I/N, with I from 1 to N")
    return int(index), int(count)


def pytest_configure(config: pytest.Config) -> None:
    _shard(config)


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    shard = _shard(config)
    if shard is None:
        return
    index, count = shard
    # Round robin in collection order: a test's parametrisations, which tend to cost
    # alike, land on different shards.
    dropped = [i for n, i in enumerate(items) if n % count != index - 1]
    config.hook.pytest_deselected(items=dropped)
    items[:] = items[index - 1 :: count]


@pytest.fixture
def skinned() -> bpy.types.Object:
    """A triangle skinned to one bone, posed 5 m above its rest at the origin."""
    rig = bpy.data.objects.new("rig", bpy.data.armatures.new("rig"))
    bpy.context.scene.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    rig.data.edit_bones.new("bone").tail = (0.0, 0.0, 1.0)
    bpy.ops.object.mode_set(mode="OBJECT")
    rig.pose.bones["bone"].location = (0.0, 5.0, 0.0)  # bone y is world -z
    mesh = bpy.data.meshes.new("skinned")
    mesh.from_pydata([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [], [(0, 1, 2)])
    part = bpy.data.objects.new("skinned", mesh)
    bpy.context.scene.collection.objects.link(part)
    part.parent = rig
    part.vertex_groups.new(name="bone").add([0, 1, 2], 1.0, "REPLACE")
    part.modifiers.new("pose", "ARMATURE").object = rig
    bpy.context.view_layer.update()
    return part
