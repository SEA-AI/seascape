"""The command line."""

import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import bpy
import click
import pytest
from click.testing import CliRunner, Result
from PIL import Image

from seascape import assets, cli, rigview, server
from seascape.assets import skies
from seascape.config import Scenario, load, loader
from seascape.outputs import labels
from seascape.outputs.calibration import Calibration, CameraCalibration
from seascape.sync import staging
from tests.scenarios import BASELINE, OPEN_SEA, SYNC_SENTRY

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
    monkeypatch.chdir(tmp_path)
    assert run("serve").exit_code == 0
    shipped = Path(cli.__file__).parents[1] / "scenarios"
    assert [s.settings.scenarios for s in served] == [tmp_path, tmp_path, shipped]


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
        (("assets", "top", "no_such"), "no mesh 'no_such'"),
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


@pytest.mark.usefixtures("empty_blend")
def test_measure_prints_only_the_manifest_lines(
    run: Run, tmp_path: Path, capfd: pytest.CaptureFixture
) -> None:
    """Blender's importer logs to file descriptor 1, which Click's runner cannot see
    and the toml must not carry.
    """
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
    a, b = load(OPEN_SEA, [*TINY, f"seed = {seed}"]).objects
    return a.bearing_deg == b.bearing_deg


@pytest.mark.render
def test_variants_skip_a_seed_whose_hulls_meet_and_merge_the_rest(
    run: Run, tmp_path: Path
) -> None:
    first = next(s for s in range(1000) if _meet(s) and not _meet(s + 1) + _meet(s + 2))
    sets = [x for line in [*TINY, f"seed = {first}"] for x in ("--set", line)]

    result = run("render", OPEN_SEA, "-o", tmp_path, "--variants", 2, *sets)

    assert result.exit_code == 0, result.output
    assert f"seed {first} skipped" in result.stderr
    folders = sorted(p.name for p in tmp_path.iterdir() if p.is_dir())
    assert folders == [str(first + 1), str(first + 2)]
    # Each folder renders again on its own; the merged root has no one scenario.
    assert [load(tmp_path / f / loader.FILENAME).seed for f in folders] == [
        first + 1,
        first + 2,
    ]
    assert not (tmp_path / loader.FILENAME).exists()
    merged = labels.Labels.model_validate_json((tmp_path / "labels.json").read_text())
    assert sorted(merged.info["scenarios"]) == folders
    assert all((tmp_path / image.file_name).exists() for image in merged.images)


THREE = [x for line in TINY for x in ("--set", line)]
# Three yachts on two bearings: two always share one.
THREE += ["--set", f"objects = [{YACHT}, {YACHT}, {YACHT}]"]


@pytest.mark.render
def test_one_scene_whose_hulls_meet_is_an_error(run: Run, tmp_path: Path) -> None:
    result = run("render", OPEN_SEA, "-o", tmp_path, *THREE)

    assert result.exit_code == 1
    assert "inside each other" in result.stderr


@pytest.mark.render
def test_variants_give_up_when_every_seed_has_hulls_meeting(
    run: Run, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(loader, "TRIES_PER_VARIANT", 1)

    result = run("render", OPEN_SEA, "-o", tmp_path, "--variants", 2, *THREE)

    assert result.exit_code == 1
    assert "0 of 2 variants in 2 seeds" in result.stderr
    assert not [p for p in tmp_path.iterdir() if p.is_dir()]


def test_a_render_s_folder_holds_the_scenario_it_rendered(
    run: Run, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rendered: list[Scenario] = []

    def render(built: Scenario, _: Path) -> list[Path]:
        rendered.append(built)
        return []

    monkeypatch.setattr(cli, "_render", render)

    result = run("render", OPEN_SEA, "-o", tmp_path, "--set", "outputs.fps = 3")

    assert result.exit_code == 0, result.output
    again = load(tmp_path / loader.FILENAME)
    assert again.model_dump() == rendered[0].model_dump()
    assert again.outputs.fps == 3


def test_a_folder_holding_an_earlier_run_is_refused_and_kept(
    run: Run, tmp_path: Path
) -> None:
    (tmp_path / "kept.png").write_bytes(b"from an earlier run")

    result = run("render", OPEN_SEA, "-o", tmp_path)

    assert result.exit_code == 1
    assert "is not empty" in result.stderr
    assert [p.name for p in tmp_path.iterdir()] == ["kept.png"]


def test_an_invalid_setting_is_an_error_without_the_secret(
    run: Run, tmp_path: Path
) -> None:
    result = run(
        "serve", "--scenarios", tmp_path, env={"SEASCAPE_MCP_SECRET": "tooshort"}
    )
    assert result.exit_code == 1
    assert "SEASCAPE_MCP_SECRET needs 32 or more" in result.output
    assert "tooshort" not in result.output


def test_an_invalid_setting_names_the_field(run: Run, tmp_path: Path) -> None:
    result = run("serve", "--scenarios", tmp_path, env={"SEASCAPE_MAX_RUN_S": "-1"})
    assert result.exit_code == 1
    assert "max_run_s" in result.output


def test_serve_keeps_the_secret_out_of_the_environment(
    run: Run, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[bool] = []
    monkeypatch.setattr(
        server, "serve", lambda *_: seen.append("SEASCAPE_MCP_SECRET" in os.environ)
    )
    secret = {"SEASCAPE_MCP_SECRET": "s" * 32}
    assert run("serve", "--scenarios", tmp_path, env=secret).exit_code == 0
    assert seen == [False]


def test_rig_view_needs_a_hull_to_draw_on(run: Run, tmp_path: Path) -> None:
    failed = run("rig-view", OPEN_SEA, "-o", tmp_path / "rig.png")
    assert failed.exit_code == 1
    assert "no asset to show its rigs on" in failed.stderr
    assert not (tmp_path / "rig.png").exists()


@pytest.mark.render
def test_rig_view_writes_one_image(run: Run, tmp_path: Path) -> None:
    scenario = (
        Path(cli.__file__).parents[1] / "scenarios" / "sync-watchkeeper-1280-gen1.toml"
    )
    out = tmp_path / "rig.png"
    drawn = run("rig-view", scenario, "-o", out)
    assert drawn.exit_code == 0, drawn.stderr
    assert drawn.stdout == f"{out}\n"
    drawn_px = Image.open(out).size
    assert drawn_px[0] == len(rigview.VIEWS) * rigview.TILE[0]
    built = load(scenario)
    lines = len(built.rigs) + len(built.mounts)
    assert drawn_px[1] == rigview.TILE[1] + 16 + rigview.LINE_PX * lines


@pytest.mark.render
def test_a_shader_cycles_draws_black_fails_the_command(tmp_path: Path) -> None:
    """Nested Mix Shaders each hold their factor on the SVM stack, past its end."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.samples = 1
    sc.render.resolution_x = sc.render.resolution_y = 8
    bpy.ops.mesh.primitive_plane_add()
    bpy.ops.object.camera_add(location=(0.0, 0.0, 5.0))
    sc.camera = bpy.context.object
    material = bpy.data.materials.new("deep")
    bpy.data.objects["Plane"].data.materials.append(material)
    tree = material.node_tree
    x = tree.nodes.new("ShaderNodeSeparateXYZ")
    tree.links.new(
        tree.nodes.new("ShaderNodeNewGeometry").outputs["Position"], x.inputs[0]
    )
    surface = tree.nodes.new("ShaderNodeBsdfDiffuse").outputs["BSDF"]
    for i in range(300):
        # Not a constant, or Cycles folds the Mix away.
        factor = tree.nodes.new("ShaderNodeMath")
        factor.operation = "MULTIPLY"
        tree.links.new(x.outputs["X"], factor.inputs[0])
        factor.inputs[1].default_value = 1e-12 * (i + 1)
        mix = tree.nodes.new("ShaderNodeMixShader")
        tree.links.new(factor.outputs[0], mix.inputs[0])
        tree.links.new(surface, mix.inputs[1])
        tree.links.new(
            tree.nodes.new("ShaderNodeBsdfDiffuse").outputs["BSDF"], mix.inputs[2]
        )
        surface = mix.outputs[0]
    output = tree.nodes["Material Output"]
    tree.links.new(surface, output.inputs["Surface"])
    with (
        pytest.raises(click.ClickException, match=r"see blender\.log"),
        cli._blender_log(tmp_path / "blender.log"),
    ):
        bpy.ops.render.render()


def test_a_preview_with_no_buoys_prints_the_course_and_renders_nothing(
    run: Run, tmp_path: Path
) -> None:
    out = tmp_path / "preview"
    refused = run("sync", SYNC_SENTRY, "--preview", "--set", "objects = []", "-o", out)

    assert refused.exit_code == 1
    assert "holds no buoys" in refused.stderr
    assert staging.suggested(load(SYNC_SENTRY)) in refused.stderr
    assert not out.exists()


def test_a_course_sheet_is_one_image(run: Run, tmp_path: Path) -> None:
    out = tmp_path / f"{SYNC_SENTRY.stem}-course.png"

    drawn = run("sync", SYNC_SENTRY, "--sheet", "-o", tmp_path)

    assert drawn.exit_code == 0, drawn.stderr
    assert drawn.stdout == f"{out}\n"
    Image.open(out).verify()


class _Band:
    """A band's render that exits with `code` on its `polls`-th poll, unless killed."""

    def __init__(self, code: int, polls: int = 1) -> None:
        self.code, self.polls, self.killed, self.returncode = code, polls, False, None

    def poll(self) -> int | None:
        self.polls -= 1
        if self.killed or self.polls <= 0:
            self.returncode = -9 if self.killed else self.code
        return self.returncode

    def kill(self) -> None:
        self.killed = True

    def wait(self) -> int | None:
        return self.poll()


def test_a_failed_band_stops_the_others_and_names_its_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started: dict[str, _Band] = {}

    def start(command: list[str], **_: object) -> _Band:
        band = _band_of(command)
        # eo would render on for a while; ir fails at once.
        started[band] = _Band(1) if band == "ir" else _Band(0, polls=5)
        return started[band]

    monkeypatch.setattr(subprocess, "Popen", start)
    monkeypatch.setattr(cli.time, "sleep", lambda _: None)

    with pytest.raises(click.ClickException, match=r"\.ir\.log") as failed:
        cli._render_apart(SYNC_SENTRY, (), load(SYNC_SENTRY), tmp_path)

    assert ".eo.log" not in str(failed.value)
    assert started["eo"].killed


def test_an_interrupt_kills_every_band(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started: list[_Band] = []

    def start(*_: object, **__: object) -> _Band:
        started.append(_Band(0, polls=5))
        return started[-1]

    def interrupted(_: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(subprocess, "Popen", start)
    monkeypatch.setattr(cli.time, "sleep", interrupted)

    with pytest.raises(KeyboardInterrupt):
        cli._render_apart(SYNC_SENTRY, (), load(SYNC_SENTRY), tmp_path)

    assert [band.killed for band in started] == [True, True]


def test_sync_parallel_renders_each_band_apart_with_the_user_s_overrides(
    run: Run, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rendered: dict[str, object] = {}

    def start(command: list[str], **_: object) -> _Band:
        sets = [command[i + 1] for i, arg in enumerate(command) if arg == "--set"]
        scene = load(Path(command[command.index("render") + 1]), sets)
        (band,) = scene.outputs.bands
        rendered[band] = scene.outputs.fps
        return _Band(1)

    def together(*_: object) -> None:
        raise AssertionError("rendered the bands together")

    monkeypatch.setattr(subprocess, "Popen", start)
    monkeypatch.setattr(cli, "_render", together)

    run(
        "sync",
        SYNC_SENTRY,
        "--parallel",
        *("--set", "outputs.fps = 3", "--set", 'outputs.bands = ["eo", "ir"]'),
        *("-o", tmp_path),
    )

    assert rendered == {"eo": 3, "ir": 3}


@pytest.mark.parametrize(
    ("overrides", "ir_cameras"),
    [(('outputs.bands = ["eo"]',), True), ((), False)],
    ids=["one band listed", "one band with cameras"],
)
def test_a_one_band_scene_renders_in_this_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    overrides: tuple[str, ...],
    ir_cameras: bool,
) -> None:
    rendered = []

    def start(*_: object, **__: object) -> None:
        raise AssertionError("started a process for one band")

    monkeypatch.setattr(subprocess, "Popen", start)
    monkeypatch.setattr(cli, "_render", lambda *args: rendered.append(args))
    built = load(SYNC_SENTRY, list(overrides))
    mounts = type(built).band_mounts
    monkeypatch.setattr(
        type(built),
        "band_mounts",
        lambda scene, band: mounts(scene, band) if ir_cameras or band != "ir" else [],
    )

    cli._render_apart(SYNC_SENTRY, overrides, built, tmp_path)

    assert rendered == [(built, tmp_path)]


@pytest.mark.parametrize(
    "args",
    [
        ("sync", SYNC_SENTRY, "--parallel", "--preview"),
        ("sync", SYNC_SENTRY, "--parallel", "--sheet"),
    ],
    ids=["preview", "sheet"],
)
def test_parallel_refuses_what_it_would_ignore(
    run: Run, args: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    def loaded(*_: object) -> None:
        raise RuntimeError("loaded past the flags")

    # Refused before the scenario loads, so a missing refusal fails here, unrendered.
    monkeypatch.setattr(cli, "load", loaded)

    refused = run(*args)

    assert refused.exit_code == 2
    assert "--parallel" in refused.output


def test_the_bands_end_as_one_render_of_the_scene(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    temporary = set()
    started: list[_Band] = []

    def start(command: list[str], env: dict[str, str], **_: object) -> _Band:
        temporary.add(env["TMPDIR"])
        _write_part(command, _band_of(command))
        # eo finishes while ir still renders.
        started.append(_Band(0, polls=1 if _band_of(command) == "eo" else 3))
        return started[-1]

    monkeypatch.setattr(subprocess, "Popen", start)
    monkeypatch.setattr(cli.time, "sleep", lambda _: None)
    built = load(SYNC_SENTRY)
    (tmp_path / loader.FILENAME).write_text("the whole scene's\n")

    cli._render_apart(SYNC_SENTRY, (), built, tmp_path)

    assert (tmp_path / loader.FILENAME).read_text() == "the whole scene's\n"
    truth = labels.Labels.read(tmp_path)
    assert truth.info["scenario"] == built.model_dump(mode="json")
    assert [i.file_name for i in truth.images] == ["eo.png", "ir.png"]
    calibration = Calibration.read(tmp_path)
    assert calibration.cameras == calibration.raw == [_camera("eo"), _camera("ir")]
    assert (tmp_path / "blender.log").read_text() == "eo\nir\n"
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "blender.log",
        "calibration.json",
        "eo.png",
        "ir.png",
        "labels.json",
        loader.FILENAME,
    ]
    assert not any(band.killed for band in started)
    # Both bands share one temporary folder, gone after the run.
    assert [Path(folder).exists() for folder in temporary] == [False]


def _write_part(command: list[str], band: str) -> None:
    """The folder a band's render leaves at the command's `-o`: a frame named for
    `band`, its truth, its scenario and its log.
    """
    part = Path(command[command.index("-o") + 1])
    part.mkdir()
    image = labels.Image(
        id=1,
        file_name=f"{band}.png",
        width=1,
        height=1,
        camera=band,
        band=band,
        hfov_deg=1.0,
        time_s=0.0,
        horizon_px=[],
    )
    labels.Labels(info={"scenario": {}}, images=[image]).write(part)
    Calibration(cameras=[_camera(band)], raw=[_camera(band)]).write(part)
    (part / f"{band}.png").write_bytes(b"frame")
    (part / "blender.log").write_text(f"{band}\n")
    (part / loader.FILENAME).write_text(f'outputs.bands = ["{band}"]\n')


def _band_of(command: list[str]) -> str:
    return next(a for a in command if a.startswith("outputs.bands")).split('"')[1]


def _camera(band: str) -> CameraCalibration:
    """Any valid calibration of `band`'s camera."""
    mount = next(m for m in load(SYNC_SENTRY).mounts if m.camera.band == band)
    return CameraCalibration.planned(mount.name, mount.camera, mount.rig_name)
