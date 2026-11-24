"""Render a sweep scenario as a preview into `<scenario>-preview/` and check that
every video sees a buoy of each ring it serves whole, and the container whole at both
ends: `check.py SCENARIO [OVERRIDE ...]`, each override a `--set` line. When the
scenario stores another course than `seascape.buoyage` computes, or a check fails, it
prints the course for the rig and exits 1.
"""

import shutil
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

from seascape import buoyage
from seascape.config import Scenario, load
from seascape.config.loader import resolve
from seascape.outputs import coverage
from seascape.outputs.labels import Labels

# Full size, so a buoy stands `buoyage.BUOY_PX` tall.
PREVIEW = ("outputs.fps = 1", "outputs.samples = { eo = 1, ir = 1 }")


def fail(scenario: Scenario, missed: str) -> NoReturn:
    """Exit with `missed`, after the course to paste where one fits the rig."""
    try:
        printed = buoyage.suggested(scenario)
    except ValueError as why:
        raise SystemExit(f"{missed}; and no course fits its rig: {why}") from why
    raise SystemExit(f"{printed}\n\n{missed}; the course for its rig is above")


def main() -> None:
    """Run on the command line's scenario and overrides."""
    path, overrides = Path(sys.argv[1]), [*sys.argv[2:], *PREVIEW]
    scenario = load(path, overrides)
    try:
        stored = buoyage.holds(resolve(path, overrides), scenario)
    except ValueError as why:
        raise SystemExit(f"no course fits the rig: {why}") from why
    if not stored:
        fail(scenario, f"{path} stores another course")
    into = path.with_name(f"{path.stem}-preview")
    # The last check's preview: `render` refuses a folder that is not empty.
    shutil.rmtree(into, ignore_errors=True)
    sets = [arg for line in overrides for arg in ("--set", line)]
    render = [sys.executable, "-m", "seascape.cli", "render", str(path), *sets]
    if code := subprocess.run([*render, "-o", str(into)], check=False).returncode:
        raise SystemExit(code)
    seen = Labels.read(into)
    try:
        coverage.check(
            seen, buoyage.requirements(scenario, {i.camera for i in seen.images})
        )
    except ValueError as missed:
        fail(scenario, f"{path}: {missed}")
    print(f"every video sees the course: {into}")


if __name__ == "__main__":
    main()
