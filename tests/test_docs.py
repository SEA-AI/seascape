"""The docs scripts render through `--set` strings, which no type checker reads."""

from pathlib import Path

import pytest

from docs.hero import SCENARIO, SKIES
from docs.primer.figures import FIGURES, FULL_HD, HERE
from seascape.config import load

PRIMER = HERE / "primer.toml"
RUNS = [
    *((SCENARIO, sets) for sets in SKIES.values()),
    (PRIMER, [FULL_HD]),
    *((PRIMER, shared + sets) for shared, runs in FIGURES.values() for _, sets in runs),
]


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize(("scenario", "overrides"), RUNS)
def test_every_override_the_docs_render_still_loads(
    scenario: Path, overrides: list[str]
) -> None:
    load(scenario, overrides)
