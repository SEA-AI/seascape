import math
from collections.abc import Callable, Sequence

import bpy
import numpy as np

CURVE_SAMPLES = 256


def yaw(bearing_deg: float) -> float:
    """Bearing to Blender yaw, in radians.

    Blender's +Z rotation turns a forward-facing object to port, so a bearing is negated
    on its way into a rotation. Only here: two negations cancel and look plausible.
    """
    return -math.radians(bearing_deg)


def place(obj: bpy.types.Object, east_m: float, north_m: float, up_m: float) -> None:
    """Position, with the rotation mode set first.

    `rotation_mode` is often QUATERNION, where assigning `rotation_euler` afterwards is
    ignored with no error.
    """
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
    """A 1-D lookup the shader samples with a Combine XYZ into an Image Texture."""
    image = bpy.data.images.new(name, len(values), 1, float_buffer=True, is_data=True)
    pixels = np.ones((len(values), 4), dtype=np.float32)
    pixels[:, :3] = np.asarray(values, dtype=np.float32)[:, None]
    image.pixels.foreach_set(pixels.ravel())
    return image
