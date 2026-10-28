"""The command line."""

import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import bpy
import pytest
from click.testing import CliRunner, Result

from seascape import assets, cli, server
from seascape.assets import skies
from seascape.config import load
from seascape.outputs import labels

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


def test_serve_takes_its_scenarios_folder(
    run: Run, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    served: list[server.State] = []
    monkeypatch.setattr(server, "serve", lambda state, *_: served.append(state))
    assert run("serve", "--scenarios", tmp_path).exit_code == 0
    assert run("serve", env={"SEASCAPE_SCENARIOS": str(tmp_path)}).exit_code == 0
    assert [s.settings.scenarios for s in served] == [tmp_path] * 2


def test_a_scenario_mistake_is_an_error_on_stderr_and_exit_1(run: Run) -> None:
    failed = run("build", BASELINE, "--set", "rigs.bow.height_m = -1")
    assert failed.exit_code == 1
    assert failed.stderr.startswith("Error: ")
    assert "Traceback" not in failed.stderr
    assert not failed.stdout


def test_schema_and_a_scenario_mistake_never_load_blender() -> None:
    # A fresh interpreter: this one has bpy loaded already.
    code = f"""
import sys
from click.testing import CliRunner
from seascape import cli
run = CliRunner().invoke
mistake = ["build", {str(BASELINE)!r}, "--set", "rigs.bow.height_m = -1"]
assert run(cli.main, ["schema"]).exit_code == 0
assert run(cli.main, mistake).exit_code == 1
assert "bpy" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


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
    names = run(env=shell).stdout.splitlines()

    assert "plain,pallet" in names
    assert all(name.startswith("plain,pa") for name in names)


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

    # The foam's frames sit beside the .blend, which reopened finds them.
    bpy.ops.wm.open_mainfile(filepath=str(blend))
    sequences = [i for i in bpy.data.images if i.source == "SEQUENCE"]
    assert sequences
    for image in sequences:
        assert image.filepath.startswith("//")
        frame = Path(bpy.path.abspath(image.filepath))
        assert frame.exists()
        assert frame.is_relative_to(blend.with_suffix(".foam"))


def test_blenders_own_output_goes_to_the_log(
    tmp_path: Path, capfd: pytest.CaptureFixture
) -> None:
    log = tmp_path / "blender.log"
    with cli._blender_log(log):
        os.write(1, b"Fra:1 Mem:12M\n")
    assert log.read_bytes() == b"Fra:1 Mem:12M\n"
    assert not capfd.readouterr().out


YACHT = '{ asset = "yacht", range_m = 900.0, bearing_deg = { choice = [0.0, 20.0] } }'
TINY = [
    f"objects = [{YACHT}, {YACHT}]",
    'outputs.bands = ["eo"]',
    'outputs.format = "png"',
    "outputs.samples.eo = 1",
    "rigs.bow.cameras.eo = { width_px = 64, height_px = 36 }",
]


def _meet(seed: int) -> bool:
    """Two yachts at one range meet when they draw one bearing."""
    a, b = load(BASELINE.with_name("open-sea.toml"), [*TINY, f"seed = {seed}"]).objects
    return a.bearing_deg == b.bearing_deg


@pytest.mark.render
def test_variants_skip_a_seed_whose_hulls_meet_and_merge_the_rest(
    run: Run, tmp_path: Path
) -> None:
    first = next(s for s in range(1000) if _meet(s) and not _meet(s + 1) + _meet(s + 2))
    sets = [x for line in [*TINY, f"seed = {first}"] for x in ("--set", line)]

    result = run(
        "render", BASELINE.with_name("open-sea.toml"), "-o", tmp_path, "--variants", 2,
        *sets,
    )  # fmt: skip

    assert result.exit_code == 0, result.output
    assert f"seed {first} skipped" in result.stderr
    folders = sorted(p.name for p in tmp_path.iterdir() if p.is_dir())
    assert folders == [str(first + 1), str(first + 2)]
    merged = labels.Labels.model_validate_json((tmp_path / "labels.json").read_text())
    assert sorted(merged.info["scenarios"]) == folders
    assert all((tmp_path / image.file_name).exists() for image in merged.images)


THREE = [x for line in TINY for x in ("--set", line)]
# Three yachts on two bearings: two always share one.
THREE += ["--set", f"objects = [{YACHT}, {YACHT}, {YACHT}]"]


@pytest.mark.render
def test_one_scene_whose_hulls_meet_is_an_error(run: Run, tmp_path: Path) -> None:
    result = run("render", BASELINE.with_name("open-sea.toml"), "-o", tmp_path, *THREE)

    assert result.exit_code == 1
    assert "inside each other" in result.stderr


@pytest.mark.render
def test_variants_give_up_when_every_seed_has_hulls_meeting(
    run: Run, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "TRIES_PER_VARIANT", 1)

    result = run(
        "render", BASELINE.with_name("open-sea.toml"), "-o", tmp_path, "--variants", 2,
        *THREE,
    )  # fmt: skip

    assert result.exit_code == 1
    assert "0 of 2 variants in 2 seeds" in result.stderr
    assert not [p for p in tmp_path.iterdir() if p.is_dir()]


def test_a_folder_holding_an_earlier_run_is_refused_and_kept(
    run: Run, tmp_path: Path
) -> None:
    (tmp_path / "kept.png").write_bytes(b"from an earlier run")

    result = run("render", BASELINE.with_name("open-sea.toml"), "-o", tmp_path)

    assert result.exit_code == 1
    assert "is not empty" in result.stderr
    assert [p.name for p in tmp_path.iterdir()] == ["kept.png"]
