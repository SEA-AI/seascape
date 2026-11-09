"""The sync check's scene: containers that every camera sees whole and near the
middle of its frame, at the rig heights that keep them there, and the check that the
ownship's motion keeps them whole in every frame.

"Near the middle" is the central half of a frame's field across and down, where lens
distortion and vignetting are least.
"""

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from seascape import assets
from seascape.config import Camera, Mount, Scenario, load
from seascape.outputs.labels import Labels
from seascape.physics import earth, fisheye, waves


class Container(NamedTuple):
    """A container as its rig sees it."""

    bearing_deg: float
    range_m: float
    mounts: tuple[Mount, ...]


def _angle_rad(camera: Camera, rows_px: float) -> float:
    """The angle off the axis of a point `rows_px` down the image's middle column."""
    if (k := camera.lens.fisheye) is None:
        tan_h = math.tan(math.radians(camera.hfov_deg) / 2)
        return math.atan(rows_px * tan_h / (camera.width_px / 2))
    return float(fisheye.theta(np.array(rows_px), k, camera.fisheye_focal_px))


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
    groups: list[tuple[float, float, tuple[Mount, ...]]] = []
    # Greedy interval stabbing, by right edge.
    for mount in sorted(mounts, key=lambda m: _central_columns_deg(m)[1]):
        left, right = _central_columns_deg(mount)
        if groups and left <= groups[-1][1]:
            low, high, seen = groups[-1]
            groups[-1] = (max(low, left), high, (*seen, mount))
        else:
            groups.append((left, right, (mount,)))
    box = assets.manifest()["container"]
    assert isinstance(box, assets.Debris)
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
            quarter = _angle_rad(mount.camera, mount.camera.height_px / 4)
            off_axis = math.radians(container.bearing_deg - mount.nominal_bearing_deg)
            # Limitation: exact for a pinhole; a fisheye's rows bend off its axis.
            ahead_m = container.range_m * math.cos(off_axis)
            reach_m = ahead_m * math.tan(down - quarter)
            low = max(low, (reach_m, mount.name))
            reach_m = ahead_m * math.tan(down + quarter)
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
        ValueError: The ownship is tilted, a rig has no model or is pitched, or its
            height keeps a container off mid-frame.
    """
    scenario = load(path, overrides)
    # Judgement: a rig lower than Hs looks into crests.
    floor_m = waves.significant_height_m(scenario.wave_field())
    objects: list[dict[str, Any]] = []
    # Placed for a still, level ownship; `check` catches what its motion takes out.
    if (ship := scenario.ownship).pitch_deg or ship.roll_deg:
        raise ValueError(
            f"sync takes a level ownship: pitch_deg {ship.pitch_deg}, "
            f"roll_deg {ship.roll_deg}"
        )
    for name, rig in scenario.rigs.items():
        if rig.model is None:
            raise ValueError(f"sync records a product: {name} has no model")
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
            ahead = container.range_m * earth.direction(container.bearing_deg, 0.0)
            # Ranges and bearings are from the ownship's origin, not the rig.
            east_m = rig.offset_x_m + float(ahead[0])
            north_m = rig.offset_y_m + float(ahead[1])
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


def check(scenario: Scenario, seen: Labels) -> None:
    """Fail if a frame sees fewer containers whole than `scene` placed for its camera.

    Args:
        scenario: As `scene` returned it.
        seen: Its render's labels.

    Raises:
        ValueError: The ownship's motion took a container out of a frame.
    """
    exr = scenario.outputs.format == "exr"
    placed: Counter[str] = Counter()
    for name in scenario.rigs:
        for container in containers([m for m in scenario.mounts if m.rig_name == name]):
            placed.update({_recorded_as(m, exr) for m in container.mounts})
    whole = Counter(a.image_id for a in seen.annotations if not a.truncated)
    for image in seen.images:
        if whole[image.id] < placed[image.camera]:
            raise ValueError(
                f"{image.camera} sees {whole[image.id]} of its "
                f"{placed[image.camera]} containers whole at {image.time_s:.2f} s: "
                "calm the ownship's motion or the sea"
            )


def _recorded_as(mount: Mount, exr: bool) -> str:
    video = None if exr else mount.rig.recording(mount.camera_name)
    return f"{mount.rig_name}_{video[0]}" if video else mount.name
