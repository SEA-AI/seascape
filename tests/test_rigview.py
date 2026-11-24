import bpy
import numpy as np
import pytest
from PIL import ImageColor

from seascape import rigview
from seascape.blender.blend import BLENDER_TO_CV
from seascape.config import load
from seascape.outputs import brand
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
    # The frustum is in Blender's camera axes.
    seen = (
        fisheye.direction(
            corners_px[:, 0], corners_px[:, 1], spec.lens.fisheye, spec.fisheye_focal_px
        )
        @ np.array(BLENDER_TO_CV.to_3x3()).T
    )
    drawn = np.array([v.co.normalized() for v in frustum.data.vertices[1:]])
    for ray in seen:
        assert max(drawn @ ray) == pytest.approx(1.0, abs=1e-6)


WIDE = len(rigview.VIEWS) * rigview.TILE[0]
LINES: list[rigview.Line] = [("hull below it at 3.74 m", brand.WHITE)] * 3


def test_rigs_sit_side_by_side_while_they_fit() -> None:
    one = rigview._caption({"bow": LINES}, WIDE)
    two = rigview._caption({"port": LINES, "stbd": LINES}, WIDE)
    stacked = rigview._caption({"port": LINES, "stbd": LINES}, rigview.TILE[0])
    assert two.size == one.size
    assert stacked.height > one.height


def test_rigs_take_a_view_each_so_a_fourth_starts_a_row() -> None:
    three = rigview._caption(dict.fromkeys("abc", LINES), WIDE)
    four = rigview._caption(dict.fromkeys("abcd", LINES), WIDE)
    assert four.height > three.height


def test_stacked_rigs_leave_more_room_between_them_than_between_lines() -> None:
    stacked = rigview._caption({"port": LINES, "stbd": LINES}, rigview.TILE[0])
    rows = np.asarray(stacked)[rigview.RULE_PX :]
    inked = np.flatnonzero(
        (rows != ImageColor.getrgb(brand.NIGHT_BLUE)).any(axis=(1, 2))
    )
    assert (np.diff(inked) - 1).max() > rigview.LINE_PX
