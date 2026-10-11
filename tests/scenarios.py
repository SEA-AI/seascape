"""Scenarios for tests: a shipped one, changed in Python.

Tests say what they change; how a scenario spells it lives here. Loader and CLI tests
write TOML, since the spelling is what they test.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from seascape.config import Camera, Object, Scenario, load

SCENARIOS = Path(__file__).parents[1] / "scenarios"
BASELINE = SCENARIOS / "baseline.toml"
OPEN_SEA = SCENARIOS / "open-sea.toml"
DRIFTING = SCENARIOS / "drifting.toml"
UNDERWAY = SCENARIOS / "underway.toml"


def target(
    asset: str, range_m: float, bearing_deg: float = 0.0, **fields: Any
) -> Object:
    return Object(asset=asset, range_m=range_m, bearing_deg=bearing_deg, **fields)


def preset_target(
    preset: str, range_m: float, bearing_deg: float = 0.0, **fields: Any
) -> dict[str, Any]:
    """A shipped object preset at a place, resolved by the loader as a scenario's."""
    return {"preset": preset, "range_m": range_m, "bearing_deg": bearing_deg, **fields}


def _plain(value: Any) -> Any:
    """Models as the tables they were given as: an unset default stays unset, so a
    validator that reads which fields were set reads the same."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", exclude_unset=True)
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    return value


def variant(
    base: Path = OPEN_SEA,
    *,
    rig: dict[str, Any] | None = None,
    cameras: Mapping[str, Camera | dict[str, Any]] | None = None,
    objects: Sequence[Object | dict[str, Any]] | None = None,
    **sections: Any,
) -> Scenario:
    """`base` with these sections overridden, as `--set` would. `rig` and `cameras`
    apply to its only rig. A `Camera` sets every field of the camera it names; a dict,
    only its own."""
    table = _plain(sections)
    if rig is not None or cameras is not None:
        rigs = list(load(base).rigs)
        if len(rigs) != 1:
            raise ValueError(f"rig= and cameras= need a base with one rig, not {rigs}")
        (name,) = rigs
        whole = {
            key: value.model_dump(mode="json") if isinstance(value, Camera) else value
            for key, value in (cameras or {}).items()
        }
        table["rigs"] = {
            name: _plain(rig or {}) | ({"cameras": whole} if whole else {})
        }
    if objects is not None:
        table["objects"] = _plain(objects)
    return load(base, [table])


def two_rigs(**sections: Any) -> Scenario:
    """Two rigs of several cameras on a hull-less ownship, rolled and pitched, with
    yachts ahead. Optics are the test's own, so no shipped preset is pinned."""
    eo = {"band": "eo", "hfov_deg": 49.0, "width_px": 3840, "height_px": 2160}
    ir = {"band": "ir", "hfov_deg": 24.0, "width_px": 640, "height_px": 480}
    down = {"pitch_deg": -10.0}
    return variant(
        ownship={"roll_deg": 2.0, "pitch_deg": 0.5},
        rigs={
            "bow": {
                "offset_x_m": 1.5,
                "offset_y_m": 6.0,
                "cameras": {
                    "eo": eo | down,
                    "eo_p": eo | down | {"yaw_deg": -40.0},
                    "eo_s": eo | down | {"yaw_deg": 40.0},
                    "ir": ir,
                },
            },
            "stern": {
                "height_m": 8.0,
                "yaw_deg": 180.0,
                "offset_x_m": -3.0,
                "offset_y_m": -20.0,
                "cameras": {"eo": eo | down, "ir": ir},
            },
        },
        targets={
            "asset": ["yacht"],
            "range_m": 400.0,
            "count": 4,
            "bearing_deg": [-60, 60],
        },
        **sections,
    )
