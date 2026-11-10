"""The rig view's drawing of a camera's field of view."""

import bpy
import numpy as np
import pytest

from seascape import rigview
from seascape.config import load
from seascape.physics import fisheye
from tests.scenarios import BASELINE


def test_a_fisheye_frustum_reaches_the_corners_its_lens_sees() -> None:
    line = 'rigs = { bow = { preset = "watchkeeper", height_m = 10.0 } }'
    scenario = load(BASELINE, [line])
    # The scenario's mounts swap the fisheye for its undistorted pinhole.
    spec = scenario.rigs["bow"].cameras["rgb1"]
    assert spec.lens.fisheye is not None
    (mount, *_) = scenario.mounts
    bpy.ops.wm.read_factory_settings(use_empty=True)
    camera = bpy.data.objects.new("eo", bpy.data.cameras.new("eo"))

    frustum = rigview._frustum(camera, mount._replace(camera=spec), 10.0)

    corners_px = np.array([(x, y) for x in (-1, 1) for y in (-1, 1)], float) / 2
    corners_px *= (spec.width_px, spec.height_px)
    seen = fisheye.direction(
        corners_px[:, 0], corners_px[:, 1], spec.lens.fisheye, spec.fisheye_focal_px
    )
    drawn = np.array([v.co.normalized() for v in frustum.data.vertices[1:]])
    for ray in seen:
        assert max(drawn @ ray) == pytest.approx(1.0, abs=1e-6)
