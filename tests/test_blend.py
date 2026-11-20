import bpy
import pytest

from seascape.blender.blend import Num, value


def _tree() -> bpy.types.NodeTree:
    return bpy.data.node_groups.new("arithmetic", "ShaderNodeTree")


def _num(tree: bpy.types.NodeTree, name: str = "x") -> Num:
    return Num(tree, value(tree, name))


# Each input is None where the operand `x` is linked, else the constant it holds.
@pytest.mark.parametrize(
    ("build", "operation", "inputs"),
    [
        (lambda x: x + 2.0, "ADD", (None, 2.0)),
        (lambda x: 2.0 + x, "ADD", (2.0, None)),
        (lambda x: x - 2.0, "SUBTRACT", (None, 2.0)),
        (lambda x: 2.0 - x, "SUBTRACT", (2.0, None)),
        (lambda x: x * 2.0, "MULTIPLY", (None, 2.0)),
        (lambda x: 2.0 * x, "MULTIPLY", (2.0, None)),
        (lambda x: x / 2.0, "DIVIDE", (None, 2.0)),
        (lambda x: 2.0 / x, "DIVIDE", (2.0, None)),
        (lambda x: x.madd(2.0, 3.0), "MULTIPLY_ADD", (None, 2.0, 3.0)),
    ],
)
def test_an_operand_keeps_its_side(build, operation: str, inputs: tuple) -> None:
    out = build(_num(_tree())).socket.node

    assert out.operation == operation
    for socket, held in zip(out.inputs, inputs, strict=False):
        assert socket.is_linked if held is None else socket.default_value == held


def test_a_num_refuses_what_would_silently_go_wrong() -> None:
    with pytest.raises(ValueError, match="different node trees"):
        _ = _num(_tree()) + _num(_tree())
    with pytest.raises(TypeError, match="truth value"):
        bool(_num(_tree()))
