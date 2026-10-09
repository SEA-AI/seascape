"""Scenarios for tests, as a shipped one with changes in Python.

Tests say what they change; how a scenario spells it lives here alone. Loader and CLI
tests write TOML on purpose: the spelling is what they test.
"""

import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from seascape.config import CFG_DIR, Camera, Object, Scenario, load

SCENARIOS = Path(__file__).parents[1] / "scenarios"
BASELINE = SCENARIOS / "baseline.toml"
OPEN_SEA = SCENARIOS / "open-sea.toml"
DRIFTING = SCENARIOS / "drifting.toml"
UNDERWAY = SCENARIOS / "underway.toml"
TWIN_POD = SCENARIOS / "twin-pod.toml"


def preset_camera(preset: str, **fields: Any) -> Camera:
    """A shipped camera preset, with `fields` changed."""
    with (CFG_DIR / "cameras" / f"{preset}.toml").open("rb") as handle:
        return Camera.model_validate(tomllib.load(handle) | fields)


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
    """`base` loaded with its only rig's fields, some of its cameras, its objects and
    any other section overridden, as `--set` would. A camera model sets every field of
    the camera it names; a dict, only its own."""
    table = _plain(sections)
    if rig is not None or cameras is not None:
        (name,) = load(base).rigs
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
