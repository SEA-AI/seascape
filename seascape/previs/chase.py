"""The previs from behind: a camera astern and above the ownship, following it."""

import math
from typing import override

import bpy
import numpy as np
from mathutils import Matrix, Vector

from seascape import angles
from seascape.previs.lens import Lens
from seascape.previs.scene import Scene

# Judgement: the chase camera, in ownship lengths behind and above its waterline, and
# its pitch down.
CHASE_BACK, CHASE_UP = 2.2, 1.32
CHASE_PITCH_DEG = -18.5


def pose(length_m: float, east_m: float, north_m: float, look_rad: float) -> Matrix:
    """`T_world_camera`, in Blender's camera axes, behind and above an ownship
    `length_m` long at `east_m`, `north_m`, looking along the bearing `look_rad`.
    """
    bow = Vector(angles.direction(math.degrees(look_rad), 0.0))
    at = Vector((east_m, north_m, 0.0)) - CHASE_BACK * length_m * bow
    at.z = CHASE_UP * length_m
    toward = Vector(angles.direction(math.degrees(look_rad), CHASE_PITCH_DEG))
    return Matrix.LocRotScale(at, toward.to_track_quat("-Z", "Y"), None)


class Chase(Lens):
    """A perspective camera astern of the ownship, looking where its first rig looks.
    Its rings' labels sit inside its field, turning with it.
    """

    view = "chase"
    size_px = (1280, 720)
    framed = False

    @override
    def _optics(self, data: bpy.types.Camera) -> None:
        # Judgement: the chase camera's field of view.
        data.angle_x = math.radians(75.0)

    @override
    def _label_choices_deg(self, scene: Scene) -> np.ndarray:
        return math.degrees(scene.look_rad(0.0)) + np.arange(-30.0, 31.0, 5.0)

    @override
    def m_per_px(self, at: Vector) -> float:
        eye = self.camera.matrix_world.translation
        half = math.tan(self.camera.data.angle_x / 2)
        return (at - eye).length * 2.0 * half / self.size_px[0]

    @override
    def slant(self, at: Vector) -> float:
        """Judgement: no less than 0.15, past which a mark is a line."""
        return max((self.camera.matrix_world.translation - at).normalized().z, 0.15)

    @override
    def up(self) -> Vector:
        along = self.camera.matrix_world.to_3x3() @ Vector((0.0, 0.0, -1.0))
        return Vector((along.x, along.y, 0.0)).normalized()

    @override
    def aim(self, scene: Scene, frames: np.ndarray) -> None:
        """Nothing: `follow` places the camera each frame, and its scale varies with
        depth.
        """

    @override
    def follow(self, scene: Scene, t_s: float) -> float:
        at = scene.path(t_s)
        look = scene.look_rad(t_s)
        self.camera.matrix_world = pose(scene.length_m, at.east_m, at.north_m, look)
        # The labels turn with the view, to stay in its field.
        return self.label_deg + math.degrees(look - scene.look_rad(0.0))

    @override
    def key(self, frame: int) -> None:
        for path in ("location", "rotation_euler"):
            self.camera.keyframe_insert(path, frame=frame)
