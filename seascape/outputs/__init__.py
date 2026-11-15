"""A render's frames as its camera writes them, and the files beside them."""

from typing import Any

from seascape.config import Scenario

# What Blender logs during a command, beside what it writes.
BLENDER_LOG = "blender.log"


def scenario_record(scenario: Scenario) -> dict[str, Any]:
    """`scenario` as a render's truth records it."""
    return scenario.model_dump(mode="json")
