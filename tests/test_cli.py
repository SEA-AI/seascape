"""The command line."""

from pathlib import Path

from click.testing import CliRunner

from seascape import cli

BASELINE = Path(__file__).parents[1] / "scenarios" / "baseline.toml"


def test_a_scenario_mistake_is_one_line_and_exit_1() -> None:
    """Validation fails before bpy loads, so this runs without Blender."""
    result = CliRunner().invoke(
        cli.main, ["build", str(BASELINE), "--set", "rig.height_m = -1"]
    )
    assert result.exit_code == 1
    assert result.output.startswith("Error: ")
    assert "Traceback" not in result.output


def test_a_missing_scenario_is_a_usage_error() -> None:
    result = CliRunner().invoke(cli.main, ["render", "no_such.toml"])
    assert result.exit_code == 2
    assert "does not exist" in result.output


def test_a_subcommand_s_help_exits_cleanly() -> None:
    result = CliRunner().invoke(cli.main, ["assets", "--help"])
    assert result.exit_code == 0
    assert "Error" not in result.output
