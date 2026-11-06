"""The sync check's containers and rig height."""

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from seascape import assets, sync
from seascape.config import Mount, Object
from seascape.physics import fisheye

SENTRY = Path("scenarios/sync-sentry.toml")
WATCHKEEPER = Path("scenarios/sync-watchkeeper.toml")
POD = {"rigs": {"mast": {"preset": "port", "height_m": 3.3}}}


def pixel(mount: Mount, east_m: float, north_m: float) -> tuple[float, float]:
    """Where a point on the water lands, in pixels right and down of the centre."""
    camera = mount.camera
    turn = math.radians(mount.nominal_bearing_deg)
    tilt = math.radians(camera.pitch_deg)
    east_m -= mount.rig.offset_x_m
    north_m -= mount.rig.offset_y_m
    ahead_m = east_m * math.sin(turn) + north_m * math.cos(turn)
    right_m = east_m * math.cos(turn) - north_m * math.sin(turn)
    up_m = -mount.rig.height_m
    forward_m = ahead_m * math.cos(tilt) + up_m * math.sin(tilt)
    down_m = ahead_m * math.sin(tilt) - up_m * math.cos(tilt)
    off_axis_m = math.hypot(right_m, down_m)
    theta = math.atan2(off_axis_m, forward_m)
    if (k := camera.lens.fisheye) is None:
        focal_px = camera.width_px / 2 / math.tan(math.radians(camera.hfov_deg) / 2)
        radius_px = focal_px * math.tan(theta)
    else:
        angle = np.array(theta)
        radius_px = float(fisheye.radius_px(angle, k, camera.fisheye_focal_px))
    return right_m * radius_px / off_axis_m, down_m * radius_px / off_axis_m


def mid_frame(mount: Mount, target: Object) -> bool:
    """Its waterline's middle in the central half of the frame's columns and rows, its
    ends in the frame and no further apart than the central half is wide.
    """
    bearing, course = math.radians(target.bearing_deg), math.radians(target.course_deg)
    east_m = target.range_m * math.sin(bearing)
    north_m = target.range_m * math.cos(bearing)
    container = assets.manifest()["container"]
    assert isinstance(container, assets.Debris)
    half_m = container.length_m / 2
    along = half_m * math.sin(course), half_m * math.cos(course)
    x, y = pixel(mount, east_m, north_m)
    ends = [
        pixel(mount, east_m + s * along[0], north_m + s * along[1]) for s in (-1, 1)
    ]
    width, height = mount.camera.width_px, mount.camera.height_px
    return (
        abs(x) <= width / 4
        and abs(y) <= height / 4
        and all(abs(end_x) <= width / 2 for end_x, _ in ends)
        and abs(ends[1][0] - ends[0][0]) <= width / 2
    )


@pytest.mark.parametrize(
    ("path", "overrides"),
    [
        (Path("scenarios/sync.toml"), []),
        (SENTRY, []),
        (WATCHKEEPER, []),
        (WATCHKEEPER, ["outputs.format = 'exr'"]),
        (SENTRY, [POD]),
    ],
    ids=["stand-in", "sentry", "watchkeeper", "watchkeeper fisheye", "side pod"],
)
def test_every_camera_sees_a_container_whole_in_its_frame_s_central_half(
    path: Path, overrides: list[Any]
) -> None:
    scenario = sync.scene(path, overrides)

    for mount in scenario.mounts:
        assert any(mid_frame(mount, o) for o in scenario.objects), mount.name


def test_a_sync_scene_holds_only_its_containers() -> None:
    ring = {
        "targets": {
            "asset": "container",
            "count": 2,
            "range_m": 500.0,
            "bearing_deg": [0.0, 90.0],
        }
    }
    scenario = sync.scene(SENTRY, [ring])

    assert scenario.targets is None
    assert [o.asset for o in scenario.objects] == ["container"]


def test_a_rig_height_that_moves_a_container_off_mid_frame_is_an_error() -> None:
    with pytest.raises(ValueError, match="set it within"):
        sync.scene(SENTRY, ["rigs.mast.height_m = 5.0"])
