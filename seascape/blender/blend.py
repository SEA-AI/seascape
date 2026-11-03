import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import bpy
import numpy as np

# A judgement.
CURVE_SAMPLES = 256

# On the world itself: a depsgraph links a driver to an ID's property, not to a node's
# (https://projects.blender.org/blender/blender/issues/142601).
SUN = ("sun_elevation", "sun_rotation", "aerosol_density")


def yaw(bearing_deg: float) -> float:
    """Bearing to Blender yaw, in radians: the one negation, as +Z turns to port."""
    return -math.radians(bearing_deg)


def place(obj: bpy.types.Object, east_m: float, north_m: float, up_m: float) -> None:
    """Position, rotation mode first: QUATERNION ignores `rotation_euler`."""
    obj.rotation_mode = "XYZ"
    obj.location = (east_m, north_m, up_m)


def animate(
    owner: bpy.types.bpy_struct,
    data_path: str,
    times_s: Sequence[float],
    value_at: Callable[[float], float | tuple[float, ...]],
    index: int = -1,
) -> None:
    """A key per frame, not an extrapolated curve: on the curved sea a hull follows
    neither a line nor a sine."""
    for frame, t in enumerate(times_s):
        if index < 0:
            setattr(owner, data_path, value_at(t))
        else:
            getattr(owner, data_path)[index] = value_at(t)
        if len(times_s) > 1:
            owner.keyframe_insert(data_path, index=index, frame=frame)


def sine(mean: float, amplitude: float, period_s: float) -> Callable[[float], float]:
    return lambda t: mean + amplitude * math.sin(2.0 * math.pi * t / period_s)


def drive(
    owner: bpy.types.bpy_struct,
    prop: str,
    expression: str,
    source: bpy.types.ID,
    **paths: str,
) -> None:
    """A render re-evaluates it; the live depsgraph keeps it stale after a change to
    `cycles.samples`.

    Variables and arithmetic only: Blender evaluates such a simple expression itself,
    so the driver runs in a `.blend` opened with Python scripts off.
    """
    curve = owner.driver_add(prop)
    # `driver_add` keys the curve at (0, 0) and (1, 1), and Blender snaps a value within
    # 1e-4 of a key onto it. Without keys the value passes through.
    curve.keyframe_points.clear()
    driver = curve.driver
    driver.type = "SCRIPTED"
    for name, path in paths.items():
        variable = driver.variables.new()
        variable.name, variable.type = name, "SINGLE_PROP"
        variable.targets[0].id_type = source.id_type
        variable.targets[0].id = source
        variable.targets[0].data_path = path
    driver.expression = expression


def curve_image(
    name: str, values: np.ndarray, float_buffer: bool = True
) -> bpy.types.Image:
    """A lookup the shader samples with a Combine XYZ into an Image Texture: one row,
    or rows bottom first, grey or RGB on a last axis. Without `float_buffer`, 8 bits
    of values in [0, 1]."""
    table = np.atleast_2d(np.asarray(values, dtype=np.float32))
    height, width = table.shape[:2]
    image = bpy.data.images.new(
        name, width, height, float_buffer=float_buffer, is_data=True
    )
    pixels = np.ones((height, width, 4), dtype=np.float32)
    pixels[..., :3] = table if table.ndim == 3 else table[..., None]
    image.pixels.foreach_set(pixels.ravel())
    # Unpacked, a generated image saves as its fill colour; EXR keeps floats.
    image.file_format = "OPEN_EXR" if float_buffer else "PNG"
    image.pack()
    return image


def lookup(
    tree: bpy.types.NodeTree,
    image: bpy.types.Image,
    x: bpy.types.NodeSocket,
    y: bpy.types.NodeSocket | None = None,
) -> bpy.types.NodeSocket:
    """`image` sampled at (x, y) in [0, 1], held at its edge outside."""
    coords = tree.nodes.new("ShaderNodeCombineXYZ")
    texture = tree.nodes.new("ShaderNodeTexImage")
    texture.image = image
    texture.extension = "EXTEND"
    tree.links.new(x, coords.inputs["X"])
    if y is not None:
        tree.links.new(y, coords.inputs["Y"])
    tree.links.new(coords.outputs["Vector"], texture.inputs["Vector"])
    return texture.outputs["Color"]


def feed(
    tree: bpy.types.NodeTree,
    socket: bpy.types.NodeSocket,
    value: bpy.types.NodeSocket | float | tuple[float, ...],
) -> None:
    if isinstance(value, bpy.types.NodeSocket):
        tree.links.new(value, socket)
    else:
        socket.default_value = value


def combine(
    tree: bpy.types.NodeTree, *xyz: bpy.types.NodeSocket | float
) -> bpy.types.NodeSocket:
    """Components left out read 0."""
    if len(xyz) > 3:
        raise ValueError(f"a Combine XYZ takes 3 components, not {len(xyz)}")
    made = tree.nodes.new("ShaderNodeCombineXYZ")
    for name, value in zip("XYZ", xyz, strict=False):
        feed(tree, made.inputs[name], value)
    return made.outputs["Vector"]


def vector(
    tree: bpy.types.NodeTree,
    operation: str,
    *inputs: bpy.types.NodeSocket | float | tuple[float, ...],
    name: str | None = None,
) -> bpy.types.NodeSocket:
    made = tree.nodes.new("ShaderNodeVectorMath")
    made.operation = operation
    if name:
        made.name = name
    sockets = list(made.inputs)
    if operation == "SCALE":
        sockets = [made.inputs["Vector"], made.inputs["Scale"]]
    for socket, value in zip(sockets, inputs, strict=False):
        feed(tree, socket, value)
    single = operation in {"DOT_PRODUCT", "LENGTH", "DISTANCE"}
    return made.outputs["Value" if single else "Vector"]


def scalar(
    tree: bpy.types.NodeTree,
    operation: str,
    *inputs: float | bpy.types.NodeSocket,
    name: str | None = None,
) -> bpy.types.NodeSocket:
    """A Math node. Unset inputs read 0.5."""
    made = tree.nodes.new("ShaderNodeMath")
    made.operation = operation
    if name:
        made.name = name
    for socket, value in zip(made.inputs, inputs, strict=False):
        feed(tree, socket, value)
    return made.outputs["Value"]


def clamped(tree: bpy.types.NodeTree, x: bpy.types.NodeSocket) -> bpy.types.NodeSocket:
    out = scalar(tree, "MAXIMUM", x, 0.0)
    out.node.use_clamp = True
    return out


type Operand = Num | bpy.types.NodeSocket | float


@dataclass(frozen=True, slots=True)
class Num:
    """A Math node's output; `a * b + c` makes the nodes in Python's evaluation order.
    `madd` fuses Multiply Add and `op` takes any other operation."""

    tree: bpy.types.NodeTree
    socket: bpy.types.NodeSocket

    def _raw(self, operand: Operand) -> bpy.types.NodeSocket | float:
        if not isinstance(operand, Num):
            return operand
        # Linking across trees drops the link and leaves the input at its default.
        if operand.tree is not self.tree:
            raise ValueError("operands belong to different node trees")
        return operand.socket

    def __bool__(self) -> bool:
        raise TypeError("a Num is a node's output, not a truth value")

    def op(self, operation: str, *others: Operand) -> "Num":
        return Num(
            self.tree,
            scalar(self.tree, operation, self.socket, *map(self._raw, others)),
        )

    def rop(self, operation: str, other: Operand) -> "Num":
        """`other` OPERATION self."""
        return Num(
            self.tree, scalar(self.tree, operation, self._raw(other), self.socket)
        )

    def madd(self, factor: Operand, term: Operand) -> "Num":
        return self.op("MULTIPLY_ADD", factor, term)

    def __add__(self, other: Operand) -> "Num":
        return self.op("ADD", other)

    def __radd__(self, other: Operand) -> "Num":
        return self.rop("ADD", other)

    def __sub__(self, other: Operand) -> "Num":
        return self.op("SUBTRACT", other)

    def __rsub__(self, other: Operand) -> "Num":
        return self.rop("SUBTRACT", other)

    def __mul__(self, other: Operand) -> "Num":
        return self.op("MULTIPLY", other)

    def __rmul__(self, other: Operand) -> "Num":
        return self.rop("MULTIPLY", other)

    def __truediv__(self, other: Operand) -> "Num":
        return self.op("DIVIDE", other)

    def __rtruediv__(self, other: Operand) -> "Num":
        return self.rop("DIVIDE", other)


def node(
    tree: bpy.types.NodeTree,
    kind: str,
    inputs: dict[str, bpy.types.NodeSocket | float | tuple[float, ...]] | None = None,
    **props: object,
) -> bpy.types.Node:
    """A node of `kind` with `props` set first, since a data type can change which
    sockets exist, then `inputs` by name."""
    made = tree.nodes.new(kind)
    for key, setting in props.items():
        setattr(made, key, setting)
    for key, source in (inputs or {}).items():
        feed(tree, made.inputs[key], source)
    return made


def mix_shader(
    tree: bpy.types.NodeTree,
    factor: bpy.types.NodeSocket | float,
    first: bpy.types.NodeSocket,
    second: bpy.types.NodeSocket,
) -> bpy.types.NodeSocket:
    """`first` where `factor` is 0, `second` where it is 1."""
    mix = node(
        tree,
        "ShaderNodeMixShader",
        {"Factor": factor, "Shader": first, "Shader_001": second},
    )
    return mix.outputs["Shader"]


def separate(
    tree: bpy.types.NodeTree, xyz: bpy.types.NodeSocket
) -> tuple[bpy.types.NodeSocket, bpy.types.NodeSocket, bpy.types.NodeSocket]:
    split = node(tree, "ShaderNodeSeparateXYZ", {"Vector": xyz})
    return split.outputs["X"], split.outputs["Y"], split.outputs["Z"]


def value(
    tree: bpy.types.NodeTree, name: str, initial: float = 0.0
) -> bpy.types.NodeSocket:
    """A named Value node, which `set_value` changes after the build."""
    made = tree.nodes.new("ShaderNodeValue")
    made.name = name
    assert made.name == name, f"{name} is taken"
    made.outputs["Value"].default_value = initial
    return made.outputs["Value"]


def set_value(tree: bpy.types.NodeTree, name: str, to: float) -> None:
    tree.nodes[name].outputs["Value"].default_value = to


def switch(
    tree: bpy.types.NodeTree,
    name: str,
    off: bpy.types.NodeSocket,
    on: bpy.types.NodeSocket,
) -> bpy.types.NodeSocket:
    """`on` or `off` for the whole frame, as `toggle` sets it: a pick, where a blend
    by 0 would turn an inf in the unpicked image into a nan."""
    made = tree.nodes.new("CompositorNodeSwitch")
    made.name = name
    assert made.name == name, f"{name} is taken"
    tree.links.new(off, made.inputs["Off"])
    tree.links.new(on, made.inputs["On"])
    return made.outputs["Image"]


def toggle(tree: bpy.types.NodeTree, name: str, on: bool) -> None:
    """The compositor evaluates both inputs of a Switch, so off mutes it: a muted
    Switch passes Off through and never evaluates On."""
    made = tree.nodes[name]
    made.inputs["Switch"].default_value = on
    made.mute = not on


def pixels(image: bpy.types.Image) -> np.ndarray:
    """An image's RGBA, (height, width, 4), bottom row first as Blender stores it."""
    width, height = image.size
    buffer = np.empty(width * height * 4, dtype=np.float32)
    image.pixels.foreach_get(buffer)
    return buffer.reshape(height, width, 4)


def read(path: Path) -> np.ndarray:
    """A file's `pixels`, the image loaded only to be read."""
    image = bpy.data.images.load(str(path))
    try:
        return pixels(image)
    finally:
        bpy.data.images.remove(image)
