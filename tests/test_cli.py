"""The command line."""

import os
from collections.abc import Callable
from pathlib import Path

import bpy
import pytest
from click.testing import CliRunner, Result

from seascape import assets, cli, skies

BASELINE = Path(__file__).parents[1] / "scenarios" / "baseline.toml"

type Run = Callable[..., Result]


@pytest.fixture
def run() -> Run:
    runner = CliRunner()
    return lambda *args, **kwargs: runner.invoke(
        cli.main, [str(a) for a in args], prog_name="seascape", **kwargs
    )


def _commands(group: cli.click.Group, path: tuple[str, ...] = ()) -> list[tuple]:
    found = []
    for name, command in group.commands.items():
        found.append((*path, name))
        if isinstance(command, cli.click.Group):
            found += _commands(command, (*path, name))
    return found


@pytest.mark.parametrize("command", _commands(cli.main), ids=" ".join)
def test_every_command_has_help(run: Run, command: tuple[str, ...]) -> None:
    helped = run(*command, "--help")
    assert helped.exit_code == 0
    assert helped.stdout.startswith(f"Usage: seascape {' '.join(command)}")
    assert not helped.stderr


def test_version(run: Run) -> None:
    assert run("--version").stdout.startswith("seascape, version ")


def test_a_scenario_mistake_is_an_error_on_stderr_and_exit_1(run: Run) -> None:
    failed = run("build", BASELINE, "--set", "rig.height_m = -1")
    assert failed.exit_code == 1
    assert failed.stderr.startswith("Error: ")
    assert "Traceback" not in failed.stderr
    assert not failed.stdout


@pytest.mark.parametrize(
    ("args", "says"),
    [
        (("render", "no_such.toml"), "does not exist"),
        (("build", BASELINE, "--band", "uv"), "'uv' is not one of"),
        (("assets", "show", "no_such"), "no mesh or sky"),
    ],
)
def test_a_usage_mistake_is_exit_2(run: Run, args: tuple, says: str) -> None:
    failed = run(*args)
    assert failed.exit_code == 2
    assert says in failed.stderr


def test_the_listing_names_every_mesh_and_sky(run: Run) -> None:
    listing = run("assets", "list").stdout
    for name in [*assets.manifest(), *skies.library()]:
        assert name in listing, name


def test_show_gives_one_asset(run: Run) -> None:
    shown = run("assets", "show", "pallet")
    assert shown.exit_code == 0
    assert "debris / pallet" in shown.stdout


def test_asset_names_complete(run: Run) -> None:
    shell = {
        "COMP_WORDS": "seascape assets show pa",
        "COMP_CWORD": "3",
        "_SEASCAPE_COMPLETE": "bash_complete",
    }
    assert run(env=shell).stdout.splitlines() == ["plain,pallet"]


def test_measure_prints_only_the_manifest_lines(
    run: Run, tmp_path: Path, capfd: pytest.CaptureFixture
) -> None:
    """Blender's importer logs to file descriptor 1, which Click's runner cannot see
    and the toml must not carry."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_cube_add()
    cube = tmp_path / "cube.glb"
    bpy.ops.export_scene.gltf(filepath=str(cube))
    capfd.readouterr()

    measured = run("assets", "measure", cube)

    assert measured.stdout.splitlines() == [
        f'sha256 = "{assets.digest(cube)}"',
        "triangles = 12",
        "texture_px = []",
    ]
    assert "glTF" not in capfd.readouterr().out


def test_build_prints_only_what_it_wrote(run: Run, tmp_path: Path) -> None:
    blend = tmp_path / "scene.blend"
    built = run("build", BASELINE, "-o", blend)
    assert built.exit_code == 0
    assert blend.exists()
    assert built.stdout.splitlines() == [f"{blend}: eo, 2 cameras (eo, ir) at 12.0 m"]


def test_blenders_own_output_goes_to_the_log(
    tmp_path: Path, capfd: pytest.CaptureFixture
) -> None:
    log = tmp_path / "blender.log"
    with cli._blender_log(log):
        os.write(1, b"Fra:1 Mem:12M\n")
    assert log.read_bytes() == b"Fra:1 Mem:12M\n"
    assert not capfd.readouterr().out
