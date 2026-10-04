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


def test_show_tells_one_asset_and_refuses_an_unknown_one() -> None:
    shown = CliRunner().invoke(cli.main, ["assets", "show", "pallet"])
    assert shown.exit_code == 0
    assert "draught" in shown.output
    assert CliRunner().invoke(cli.main, ["assets", "show", "no_such"]).exit_code == 2


def test_asset_names_complete() -> None:
    shell = {
        "COMP_WORDS": "seascape assets show pa",
        "COMP_CWORD": "3",
        "_SEASCAPE_COMPLETE": "bash_complete",
    }
    completed = CliRunner().invoke(cli.main, env=shell, prog_name="seascape")
    assert "pallet" in completed.output
