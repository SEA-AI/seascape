"""A previs view's camera: what every view shares, and what each must say."""

from abc import ABC, abstractmethod
from typing import ClassVar

import bpy
import numpy as np
import numpy.typing as npt
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

from seascape import angles
from seascape.outputs.previs import Px, View, label_bearing_deg
from seascape.previs.scene import Scene


class Lens(ABC):
    """A view's camera over `scene`, its size, and the bearing its rings' labels sit
    at at the start, where no hull is.

    Attributes:
        view: The view's name.
        size_px: The picture's width and height.
        framed: Whether a fan fades over the frame's border, so one past it ends
            without a hard line.
        camera: The view's camera, linked to the scene.
        label_deg: The rings' labels' bearing at the start.
    """

    view: ClassVar[View]
    size_px: ClassVar[tuple[int, int]]
    framed: ClassVar[bool]

    def __init__(self, scene: Scene) -> None:
        sc = bpy.context.scene
        self.camera = bpy.data.objects.new(self.view, bpy.data.cameras.new(self.view))
        self._optics(self.camera.data)
        sc.collection.objects.link(self.camera)
        self.camera.data.sensor_fit = "HORIZONTAL"
        self.camera.data.clip_start = 1.0
        # The far clip plane cuts in silence: past the sea's edge, from either camera.
        self.camera.data.clip_end = 2.0 * scene.reach_m
        sc.frame_set(0)
        hulls_deg = [
            angles.bearing_deg(*r.matrix_world.translation.xy) for r in scene.roots[1:]
        ]
        self.label_deg = label_bearing_deg(self._label_choices_deg(scene), hulls_deg)

    @property
    def k(self) -> float:
        """Pixels of this view per pixel of a 1000 px tall one."""
        return self.size_px[1] / 1000.0

    def px(self, at: npt.ArrayLike) -> Px | None:
        """Where `at` falls in the picture, in pixels from its top left; none behind
        the camera.
        """
        p = world_to_camera_view(bpy.context.scene, self.camera, Vector(at))
        if p.z <= 0.0:
            return None
        return p.x * self.size_px[0], (1.0 - p.y) * self.size_px[1]

    @abstractmethod
    def _optics(self, data: bpy.types.Camera) -> None:
        """Set the camera's projection and field."""

    @abstractmethod
    def _label_choices_deg(self, scene: Scene) -> np.ndarray:
        """Bearings the rings' labels may sit at at the start."""

    @abstractmethod
    def m_per_px(self, at: Vector) -> float:
        """What a pixel spans on the water at `at`."""

    @abstractmethod
    def slant(self, at: Vector) -> float:
        """How much a flat mark at `at` is foreshortened, seen at a slant: the sine of
        its elevation.
        """

    @abstractmethod
    def up(self) -> Vector:
        """The unit direction on the water that runs up the picture."""

    @abstractmethod
    def aim(self, scene: Scene, frames: np.ndarray) -> float | None:
        """Point the camera for `frames`; the metres a pixel spans, if one scale holds
        across the picture.
        """

    @abstractmethod
    def follow(self, scene: Scene, t_s: float) -> float:
        """Move the camera to `t_s`; the rings' labels' bearing then."""

    @abstractmethod
    def key(self, frame: int) -> None:
        """Key at `frame` what `follow` moved."""
