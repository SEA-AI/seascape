"""The docs scripts render through `--set` strings, which no type checker reads."""

from pathlib import Path

import pytest

from docs.hero import SCENARIO, SKIES
from docs.primer.figures import FIGURES, FULL_HD, PRIMER
from seascape.config import load

RUNS = {
    **{f"hero {sky}": (SCENARIO, sets) for sky, sets in SKIES.items()},
    "crops": (PRIMER, [FULL_HD]),
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
