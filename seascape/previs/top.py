"""The previs from above: a map, bow up, still over the run."""

import math
from typing import override

import bpy
import numpy as np
from mathutils import Matrix, Vector

from seascape.outputs.previs import LABEL_PX, TICK_LABEL_PX, TICK_STEP_DEG, top_view
from seascape.previs.lens import Lens
from seascape.previs.scene import Scene, on_water


def _hulls(scene: Scene, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Every hull's east and north over `frames`, (n, 2), and half its diagonal, the
    circle that holds it however it turns, (n,).
    """
    sc = bpy.context.scene
    sides = np.array(
        [(r.get("length_m", scene.length_m), r.get("beam_m", 0.0)) for r in scene.roots]
    )
    found = []
    for k in frames:
        sc.frame_set(int(k))
        found += [tuple(r.matrix_world.translation.xy) for r in scene.roots]
    return np.array(found), np.tile(np.hypot(*sides.T) / 2, len(frames))


class Top(Lens):
    """An orthographic camera straight down, the ownship's starting heading up. Its
    rings' labels sit astern, which the chase camera does not see, between the
    compass ticks.
    """

    view = "top"
    # Square, as range rings are round.
    size_px = (720, 720)
    framed = True

    @override
    def _optics(self, data: bpy.types.Camera) -> None:
        data.type = "ORTHO"

    @override
    def _label_choices_deg(self, scene: Scene) -> np.ndarray:
        return math.degrees(scene.heading_rad) + np.arange(105.0, 256.0, TICK_STEP_DEG)

    @override
    def m_per_px(self, at: Vector) -> float:
        return self.camera.data.ortho_scale / self.size_px[0]

    @override
    def slant(self, at: Vector) -> float:
        return 1.0

    @override
    def up(self) -> Vector:
        along = self.camera.matrix_world.to_3x3() @ Vector((0.0, 1.0, 0.0))
        return Vector((along.x, along.y, 0.0)).normalized()

    @override
    def aim(self, scene: Scene, frames: np.ndarray) -> float:
        """Frame every hull whole over `frames`, every ring's label and the compass
        with room for its labels. An object grows to at most `OBJECT_PX` across, well
        inside `MARGIN`, so its true size is fitted.
        """
        compass_deg = math.degrees(scene.heading_rad) + np.arange(0.0, 360.0, 90.0)
        outer = max(scene.rings, default=0.0)
        labels = on_water(self.label_deg, np.array(list(scene.rings)), 0.0)
        hulls, reach_m = _hulls(scene, frames)
        m_per_px, k = 0.0, self.k
        # How far the compass's labels reach follows the scale, so each pass fits what
        # the last one's scale draws. Judgement: three passes settle every shipped
        # scenario; iterate to a tolerance if one does not.
        for _ in range(3):
            pad_m = (TICK_LABEL_PX + LABEL_PX) * k * m_per_px if scene.rings else 0.0
            marks = np.concatenate(
                [labels[:, :2], on_water(compass_deg, outer + pad_m, 0.0)[:, :2]]
            )
            view = top_view(
                np.concatenate([hulls, marks]),
                scene.heading_rad,
                self.size_px,
                np.concatenate([reach_m, np.zeros(len(marks))]),
            )
            m_per_px = view.m_per_px
        self.camera.data.ortho_scale = m_per_px * self.size_px[0]
        self.camera.matrix_world = Matrix(view.matrix(0.5 * self.camera.data.clip_end))
        return m_per_px

    @override
    def follow(self, scene: Scene, t_s: float) -> float:
        return self.label_deg

    @override
    def key(self, frame: int) -> None:
        pass
