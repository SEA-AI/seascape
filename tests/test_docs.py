"""The docs scripts render through `--set` strings, which no type checker reads."""

import re
import tomllib
from pathlib import Path

import pytest

from docs.hero import SCENARIO, SKIES
from docs.primer.figures import FIGURES, FULL_HD, PRIMER
from docs.wear import SCENARIO as WEAR
from docs.wear import WEARS
from seascape.config import load

DOCS = Path(__file__).parents[1] / "docs"
SCENARIOS = DOCS.parent / "scenarios"

RUNS = {
    **{f"hero {sky}": (SCENARIO, sets) for sky, sets in SKIES.items()},
    "crops": (PRIMER, [FULL_HD]),
    "visibility": (Path(__file__).parents[1] / "docs" / "visibility.toml", []),
    **{f"wear {name}": (WEAR, sets) for name, sets in WEARS.items()},
    **{
        f"{name}: {caption}": (PRIMER, shared + sets)
        for name, (shared, panels) in FIGURES.items()
        for caption, sets in panels
    },
}


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize(("scenario", "overrides"), RUNS.values(), ids=RUNS.keys())
def test_every_override_the_docs_render_still_loads(
    scenario: Path, overrides: list[str]
) -> None:
    load(scenario, overrides)


EXAMPLES = re.findall(r"```toml\n(.*?)```", (DOCS / "scenarios.md").read_text(), re.S)


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("example", EXAMPLES, ids=[e.splitlines()[0] for e in EXAMPLES])
def test_every_scenario_example_in_the_docs_loads(example: str, tmp_path: Path) -> None:
    """A scenario that extends one under scenarios/, or a table over the baseline."""
    table = tomllib.loads(example)
    if "extends" not in table:
        load(SCENARIOS / "baseline.toml", [table])
        return
    path = tmp_path / "example.toml"
    path.write_text(example.replace('extends = "', f'extends = "{SCENARIOS}/', 1))
    load(path)
