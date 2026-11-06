"""The sync check's scene: 20 ft containers that every camera sees whole and near the
middle of its frame, at the rig heights that keep them there.

"Near the middle" is the central half of a frame's field across and down, where lens
distortion and vignetting are least.
"""

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from seascape import assets
from seascape.config import Camera, Mount, Scenario, load
from seascape.physics import fisheye, waves


class Container(NamedTuple):
    """A container as its rig sees it."""

    bearing_deg: float
    range_m: float
    mounts: list[Mount]


def half_vfov_rad(camera: Camera) -> float:
    """Half the angle the camera sees down its image."""
    if (k := camera.lens.fisheye) is None:
        tan_h = math.tan(math.radians(camera.hfov_deg) / 2)
        return math.atan(tan_h * camera.height_px / camera.width_px)
    half_h = np.array(camera.height_px / 2)
    return float(fisheye.theta(half_h, k, camera.fisheye_focal_px))


def _central_columns_deg(mount: Mount) -> tuple[float, float]:
    quarter_deg = mount.camera.hfov_deg / 4
    return (
        mount.nominal_bearing_deg - quarter_deg,
        mount.nominal_bearing_deg + quarter_deg,
    )


def containers(mounts: Sequence[Mount]) -> list[Container]:
    """The fewest containers that put one in the central half of every camera's field
    across, each mid-way through the halves it falls in, and spanning half the field
    of the narrowest camera that sees it, so it is whole in all of them.

    Args:
        mounts: One rig's cameras.
    """
    # Limitation: bearings do not wrap, so 180 deg astern can take two containers.
    groups: list[tuple[float, float, list[Mount]]] = []
    # Greedy interval stabbing, by right edge: the fewest points that stab every half.
    for mount in sorted(mounts, key=lambda m: _central_columns_deg(m)[1]):
        left, right = _central_columns_deg(mount)
        if groups and left <= groups[-1][1]:
            low, high, seen = groups[-1]
            groups[-1] = (max(low, left), high, [*seen, mount])
        else:
            groups.append((left, right, [mount]))
    box = assets.manifest()["container"]
    assert isinstance(box, assets.Debris)  # the manifest's kind
    placed = []
    for low, high, seen in groups:
        quarter = math.radians(min(m.camera.hfov_deg for m in seen)) / 4
        placed.append(
            Container((low + high) / 2, box.length_m / 2 / math.tan(quarter), seen)
        )
    return placed


def heights_m(placed: Sequence[Container], floor_m: float) -> tuple[float, float]:
    """The lowest and highest rig heights that put every container in the central
    half of its cameras' fields down.

    Args:
        placed: One rig's containers.
        floor_m: The lowest height allowed.

    Raises:
        ValueError: No height does it for every camera.
    """
    low, high = (floor_m, "the waves' significant height"), (math.inf, "")
    for container in placed:
        for mount in container.mounts:
            down = -math.radians(mount.camera.pitch_deg)
            quarter = half_vfov_rad(mount.camera) / 2
            reach_m = container.range_m * math.tan(down - quarter)
            low = max(low, (reach_m, mount.name))
            reach_m = container.range_m * math.tan(down + quarter)
            high = min(high, (reach_m, mount.name))
    if low[0] > high[0]:
        raise ValueError(
            f"no rig height keeps every container mid-frame: {low[1]} needs "
            f"{low[0]:.1f} m or more, {high[1]} {high[0]:.1f} m or less"
        )
    return low[0], high[0]


def scene(path: Path, overrides: Sequence[str | Mapping[str, Any]] = ()) -> Scenario:
    """The scenario at `path`, its objects the containers.

    Raises:
        ValueError: A rig is pitched, or its height keeps a container off mid-frame.
    """
    scenario = load(path, overrides)
    # Judgement: a rig lower than Hs looks into crests.
    floor_m = waves.significant_height_m(scenario.wind_waves() + scenario.swell_waves())
    objects: list[dict[str, Any]] = []
    # Limitation: placed for a level ownship; its mean pitch and roll tilt every frame.
    for name, rig in scenario.rigs.items():
        if rig.pitch_deg:
            # Its pitch turns an off-axis camera's bearing and tilt from their sums.
            raise ValueError(
                f"sync takes level rigs: {name} has pitch_deg {rig.pitch_deg}"
            )
        placed = containers([m for m in scenario.mounts if m.rig_name == name])
        low_m, high_m = heights_m(placed, floor_m)
        if not low_m <= rig.height_m <= high_m:
            raise ValueError(
                f"{name}: height_m {rig.height_m} keeps a container off mid-frame: "
                f"set it within {low_m:.2f} to {high_m:.2f}"
            )
        for container in placed:
            bearing = math.radians(container.bearing_deg)
            # Ranges and bearings are from the ownship's origin, not the rig.
            east_m = rig.offset_x_m + container.range_m * math.sin(bearing)
            north_m = rig.offset_y_m + container.range_m * math.cos(bearing)
            objects.append(
                {
                    "asset": "container",
                    "range_m": math.hypot(east_m, north_m),
                    "bearing_deg": math.degrees(math.atan2(east_m, north_m)),
                    # Broadside to its rig, its length across the view.
                    "heading_deg": container.bearing_deg + 90.0,
                }
            )
    only = {"objects": objects, "targets": None, "farms": []}
    return load(path, [*overrides, only])
