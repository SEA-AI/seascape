"""A TOML loader that never guesses.

A scenario is a TOML file. Two forms of reuse, both resolved here:

- `extends = "baseline.toml"` at the top level, for a variant that is a diff.
- `preset = "pod"` inside a block, next to the overrides it applies to.

Rules:

1. `preset` is a key inside its block, so there are no precedence rules between
   parents.
2. Syntax decides what a name is. A bare name is a preset shipped under `presets/`,
   anything with `/` or ending in `.toml` is a path relative to the including file.
   Never try one form and fall back to the other.
3. Tables merge, everything else replaces. A list, or a table naming a preset, is
   replaced whole.

Any field can instead be a draw, `{ uniform = [lo, hi] }`, `{ log_uniform = [lo, hi] }`,
`{ triangular = [lo, mode, hi] }` or `{ choice = [...] }`, resolved after both forms of
reuse and before validation."""

import tomllib
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from seascape.config.draws import DRAWS, Choice, OpticalDepth, kind, substream
from seascape.config.scenario import Scenario
from seascape.config.world import Sky

PRESETS = Path(__file__).parent / "presets"


def _is_draw(node: Any) -> bool:
    return kind(node) is not None


def _is_table(node: Any) -> bool:
    return isinstance(node, dict) and not _is_draw(node)


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    return base | {
        key: _merge_value(base.get(key), value) for key, value in over.items()
    }


def _merge_value(base: Any, over: Any) -> Any:
    """Two tables merge, and a table over a choice of tables merges into each option;
    anything else, a draw or a table naming a preset included, replaces outright."""
    if not _is_table(over) or "preset" in over:
        return over
    if _is_table(base):
        return _merge(base, over)
    options = base.get("choice") if _is_draw(base) else None
    if isinstance(options, list) and all(_is_table(option) for option in options):
        # The choice's own fields, its weights, stay with the choice.
        own = {k: v for k, v in over.items() if k in Choice.model_fields}
        rest = {k: v for k, v in over.items() if k not in own}
        return base | own | {"choice": [_merge(option, rest) for option in options]}
    return over


def _include_path(name: Any, base: Path, block: str | None = None) -> Path:
    """Resolve rule 2. `block` is the preset directory; None allows only a path."""
    if not isinstance(name, str):
        raise TypeError(f"include must be a string, got {name!r}")
    if "/" in name or name.endswith(".toml"):
        return base / name
    if block is None:
        raise ValueError(f"{name!r} must be a path: it needs a '/' or a '.toml' suffix")
    return PRESETS / block / f"{name}.toml"


# Tables of named entries: an entry's preset is one of its table's, never of its name.
_KEYED = {"rigs", "cameras"}


def _expand(
    node: Any,
    block: str | None,
    base: Path,
    chain: tuple[Path, ...],
    keyed: bool = False,
) -> Any:
    """Resolve every `preset` key in the tree, innermost first."""
    if isinstance(node, list):
        return [_expand(item, block, base, chain, keyed) for item in node]
    if not isinstance(node, dict):
        return node
    # A draw holds a list of options; an entry by a draw's name would read as one.
    if keyed and (
        named := [k for k in node if k in DRAWS and not isinstance(node[k], list)]
    ):
        raise ValueError(f"{named[0]!r} is a draw, not a name: rename it")
    # A draw's options are presets of the field it sits on, never of `choice`.
    if _is_draw(node):
        return {key: _expand(v, block, base, chain, keyed) for key, v in node.items()}
    if keyed:
        return {key: _expand(v, block, base, chain) for key, v in node.items()}
    out = {
        key: _expand(value, key, base, chain, key in _KEYED)
        for key, value in node.items()
    }
    if (name := out.pop("preset", None)) is None:
        return out
    return _merge(_read(_include_path(name, base, block), chain), out)


def _read(path: Path, chain: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.resolve()
    if path in chain:
        cycle = " -> ".join(str(p) for p in (*chain, path))
        raise ValueError(f"circular include: {cycle}")
    chain = (*chain, path)

    with path.open("rb") as handle:  # TOML is UTF-8 by spec, so never read_text
        data = tomllib.load(handle)

    if (parent := data.pop("extends", None)) is not None:
        data = _merge(_read(_include_path(parent, path.parent), chain), data)
    return _expand(data, None, path.parent, chain)


_SHARED = {"sensor", "lens", "processing"}


def _per_rig(path: str) -> str:
    """A rig's cameras share one sensor, lens and ISP setting: their draws skip the
    camera's name and every pick above it, under a name `Name` refuses a camera."""
    parts = path.split("/")
    for i, part in enumerate(parts):
        head = [p for p in parts[:i] if p != "choice"]
        if (
            part in _SHARED
            and head[:2] == ["draw", "rigs"]
            and head[3:4] == ["cameras"]
            and len(head) == 5
        ):
            return "/".join([*head[:3], "cameras:shared", *parts[i:]])
    return path


def _draw(node: Any, seed: int, path: str) -> Any:
    """Replace every draw in the tree by a value from a substream named for where it
    sits, `draw/sky/visibility_km`."""
    if isinstance(node, list):
        return [_draw(item, seed, f"{path}/{i}") for i, item in enumerate(node)]
    if not isinstance(node, dict):
        return node
    if (name := kind(node)) is not None:
        rng = substream(seed, _per_rig(path))
        value = DRAWS[name].model_validate(node).draw(rng)
        # A chosen draw on the pick's own substream would replay the pick's state.
        return _draw(value, seed, f"{path}/{name}" if _is_draw(value) else path)
    return {key: _draw(value, seed, f"{path}/{key}") for key, value in node.items()}


def _copies(objects: list[Any], seed: int) -> list[Any]:
    """Each entry `count` times, every copy drawn from a substream of its own."""
    out = []
    for i, spec in enumerate(objects):
        if not isinstance(spec, dict):
            out.append(spec)  # validation names what it should have been
            continue
        path = f"draw/objects/{i}"
        count = _draw(spec.get("count", 1), seed, f"{path}/count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(
                f"objects[{i}]: count must be a whole number from 0, not {count}"
            )
        if count > 1 and "orbit" in spec:
            raise ValueError(f"objects[{i}]: an orbit spaces its hulls by orbit.count")
        copies = [_draw(spec, seed, f"{path}/{k}") for k in range(count)]
        # A pick's count would come after the copies were made.
        if _is_draw(spec) and any(c.get("count", 1) != 1 for c in copies):
            raise ValueError(f"objects[{i}]: a drawn object cannot carry a count")
        out += [
            c | {"count": 1} | _optical_depth(c, seed, f"{path}/{k}/range_m")
            for k, c in enumerate(copies)
        ]
    return out


def _optical_depth(copy: dict[str, Any], seed: int, path: str) -> dict[str, Any]:
    """The copy's depth, drawn here so a change to an earlier entry's count leaves it
    be; `_range_by_optical_depth` turns it into metres."""
    depth = copy.get("range_m")
    if not (isinstance(depth, dict) and "optical_depth" in depth):
        return {}
    bounds = OpticalDepth.model_validate(depth).optical_depth
    return {"range_m": {"drawn_optical_depth": substream(seed, path).uniform(*bounds)}}


def load(
    path: str | Path, overrides: Iterable[str | Mapping[str, Any]] = ()
) -> Scenario:
    """Read a scenario TOML, resolving `extends` and `preset`, draw every random
    field, and validate it.

    Each override, a TOML assignment such as `rigs.bow.pitch_deg = -5` or the table it
    parses to, merges over the file and resolves as if it were a line in it.
    """
    path = Path(path)
    data = _read(path)
    for override in overrides:
        table = tomllib.loads(override) if isinstance(override, str) else dict(override)
        data = _expand(_merge(data, table), None, path.parent, ())
    seed = data.get("seed", Scenario.model_fields["seed"].default)
    if isinstance(seed, dict):
        raise ValueError("seed cannot be drawn: it seeds the draws")
    if isinstance(objects := data.get("objects"), list):
        data["objects"] = _copies(objects, seed)
    drawn = _draw(data, seed, "draw")
    _range_by_optical_depth(drawn)
    return Scenario.model_validate(drawn)


def _range_by_optical_depth(data: dict[str, Any]) -> None:
    """Replace each drawn optical depth with metres, once the visibility is drawn."""
    for i, spec in enumerate(data.get("objects") or []):
        depth = spec.get("range_m") if isinstance(spec, dict) else None
        if not (isinstance(depth, dict) and "drawn_optical_depth" in depth):
            continue
        extinction_per_m = Sky.model_validate(data.get("sky") or {}).extinction_per_m
        if not extinction_per_m:
            raise ValueError(f"objects[{i}]: optical_depth needs sky.visibility_km")
        spec["range_m"] = depth["drawn_optical_depth"] / extinction_per_m


# Seeds `render --variants` tries per variant before it gives up on hulls that meet.
TRIES_PER_VARIANT = 10


def seeds(first: int, variants: int) -> range:
    """Every seed `render --variants` may try, in order; a lone variant is never
    replaced."""
    return range(first, first + (1 if variants == 1 else TRIES_PER_VARIANT * variants))


def load_variants(
    path: str | Path, overrides: Iterable[str] = (), count: int = 1
) -> list[Scenario]:
    """`count` scenarios from consecutive seeds, the file's own first."""
    overrides = list(overrides)
    first = load(path, overrides)
    seeds = range(first.seed + 1, first.seed + count)
    return [first, *(load(path, [*overrides, f"seed = {seed}"]) for seed in seeds)]


def json_schema() -> dict[str, Any]:
    """The scenario's JSON schema, in which any field can be a draw."""
    schema = Scenario.model_json_schema()
    draws = [{"$ref": f"#/$defs/{model.__name__}"} for model in DRAWS.values()]
    for model in (schema, *schema["$defs"].values()):
        for name, spec in model.get("properties", {}).items():
            if model is schema and name == "seed":
                continue  # `load` refuses a drawn seed
            # Outside the anyOf, where an editor finds the hover doc.
            outer = {
                k: spec.pop(k) for k in ("title", "description", "default") if k in spec
            }
            values = spec.pop("anyOf", None) or [spec]
            model["properties"][name] = {**outer, "anyOf": [*values, *draws]}
    for model in DRAWS.values():
        schema["$defs"][model.__name__] = model.model_json_schema()
    schema["$defs"]["OpticalDepth"] = OpticalDepth.model_json_schema()
    schema["$defs"]["Object"]["properties"]["range_m"]["anyOf"].append(
        {"$ref": "#/$defs/OpticalDepth"}
    )
    return schema
