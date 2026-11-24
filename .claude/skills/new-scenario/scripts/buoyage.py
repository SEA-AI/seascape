"""Print the buoys and container a sweep scenario's rig and turn need, as TOML to
paste: `buoyage.py SCENARIO [OVERRIDE ...]`, each override a `--set` line.
"""

import sys
from pathlib import Path

from seascape import buoyage
from seascape.config import load


def main() -> None:
    """Run on the command line's scenario and overrides."""
    scenario = load(Path(sys.argv[1]), sys.argv[2:])
    try:
        print(buoyage.suggested(scenario))
    except ValueError as why:
        raise SystemExit(f"no course fits the rig: {why}") from why


if __name__ == "__main__":
    main()
