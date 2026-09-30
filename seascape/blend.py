import math
from collections.abc import Callable, Sequence

import bpy
import numpy as np

# A judgement.
CURVE_SAMPLES = 256


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


def curve_image(name: str, values: np.ndarray) -> bpy.types.Image:
    """A lookup the shader samples with a Combine XYZ into an Image Texture: one row,
    or rows bottom first."""
    table = np.atleast_2d(np.asarray(values, dtype=np.float32))
    height, width = table.shape
    image = bpy.data.images.new(name, width, height, float_buffer=True, is_data=True)
    pixels = np.ones((height, width, 4), dtype=np.float32)
    pixels[..., :3] = table[..., None]
    image.pixels.foreach_set(pixels.ravel())
    # Unpacked, a generated image saves as its fill colour; EXR keeps floats.
    image.file_format = "OPEN_EXR"
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
