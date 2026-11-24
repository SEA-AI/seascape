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

from seascape import assets, cli, previs, processes, rigview, server
from seascape import render as renderer
from seascape.assets import skies
from seascape.config import Scenario, load, loader
from seascape.outputs import labels
from seascape.outputs.calibration import Calibration, CameraCalibration
from tests.scenarios import BASELINE, OPEN_SEA, SWEEP_SENTRY, SWEEP_WATCHKEEPER

type Run = Callable[..., Result]


@pytest.fixture
def run() -> Run:
    runner = CliRunner()
    return lambda *args, **kwargs: runner.invoke(
        cli.main, [str(a) for a in args], prog_name="seascape", **kwargs
    )


def test_a_subcommand_s_help_exits_0(run: Run) -> None:
    helped = run("assets", "show", "--help")
    assert helped.exit_code == 0
    assert helped.stdout.startswith("Usage: seascape assets show")
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
    assert load(tmp_path / cli.SCENARIO).seed == first
    assert not any((tmp_path / f / cli.SCENARIO).exists() for f in folders)
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


@pytest.mark.parametrize(
    ("command", "scenario"), [("render", OPEN_SEA), ("recording", SWEEP_SENTRY)]
)
def test_a_render_s_folder_holds_the_scenario_it_rendered(
    command: str,
    scenario: Path,
    run: Run,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rendered: list[Scenario] = []
    monkeypatch.setattr(cli, "_render", lambda built, _: rendered.append(built) or [])
    monkeypatch.setattr(cli, "_encode", lambda *_: None)
    monkeypatch.setattr(cli.recording, "export", lambda _: [])

    result = run(command, scenario, "-o", tmp_path, "--set", "outputs.fps = 3")

    assert result.exit_code == 0, result.output
    again = load(tmp_path / cli.SCENARIO)
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
    scenario = SWEEP_WATCHKEEPER
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


def _in_turn(
    work: Callable[..., object],
    jobs: dict[str, tuple[object, ...]],
    logs: dict[str, Path],
    tick: Callable[[], None],
) -> list[str]:
    """`processes.at_once`, one job after the other in this process, none failing."""
    for key, args in jobs.items():
        logs[key].touch()
        work(*args)
    tick()
    return []


def test_a_failed_band_names_its_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(processes, "at_once", lambda *_: ["ir"])

    with pytest.raises(click.ClickException, match=r"\.ir\.log") as failed:
        cli._render_apart(load(SWEEP_SENTRY), tmp_path)

    assert ".eo.log" not in str(failed.value)


@pytest.mark.parametrize("command", ["render", "recording"])
def test_parallel_renders_each_band_apart_with_the_user_s_overrides(
    command: str, run: Run, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rendered: dict[str, object] = {}

    def render(scene: Scenario, part: Path) -> None:
        (band,) = scene.outputs.bands
        rendered[band] = scene.outputs.fps
        _write_part(part, band)

    def together(*_: object) -> None:
        raise AssertionError("rendered the bands together")

    monkeypatch.setattr(processes, "at_once", _in_turn)
    monkeypatch.setattr(renderer, "render", render)
    monkeypatch.setattr(cli, "_render", together)
    monkeypatch.setattr(cli, "_encode", lambda *_: [])
    monkeypatch.setattr(cli.recording, "export", lambda _: [])

    result = run(
        command,
        SWEEP_SENTRY,
        "--parallel",
        *("--set", "outputs.fps = 3", "--set", 'outputs.bands = ["eo", "ir"]'),
        *("-o", tmp_path),
    )

    assert result.exit_code == 0, result.output
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
    files = [tmp_path / "frame.png"]

    def apart(*_: object) -> None:
        raise AssertionError("started a process for one band")

    monkeypatch.setattr(processes, "at_once", apart)
    monkeypatch.setattr(cli, "_render", lambda *args: rendered.append(args) or files)
    built = load(SWEEP_SENTRY, list(overrides))
    mounts = type(built).band_mounts
    monkeypatch.setattr(
        type(built),
        "band_mounts",
        lambda scene, band: mounts(scene, band) if ir_cameras or band != "ir" else [],
    )

    assert cli._render_apart(built, tmp_path) == files
    assert rendered == [(built, tmp_path)]


@pytest.mark.parametrize(
    "args",
    [
        ("render", SWEEP_SENTRY, "--parallel", "--variants", 2),
        ("recording", SWEEP_SENTRY.parent, "--parallel"),
        ("recording", SWEEP_SENTRY.parent, "-o", "x"),
        ("recording", SWEEP_SENTRY.parent, "--set", "seed = 1"),
    ],
    ids=["--parallel with variants", "--parallel", "-o", "--set"],
)
def test_flags_it_would_ignore_are_refused(
    run: Run, args: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    def rendered(*_: object) -> None:
        raise AssertionError("went on past the flags")

    monkeypatch.setattr(cli, "_pin", rendered)
    monkeypatch.setattr(cli.recording, "export", rendered)

    refused = run(*args)

    assert refused.exit_code == 2
    assert args[2] in refused.output


def test_a_scenario_s_recording_is_rendered_and_encoded_beside_it(
    run: Run, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenario = tmp_path / "sweep.toml"
    scenario.write_text(f"extends = {str(SWEEP_SENTRY)!r}\n")
    steps: list[tuple[str, Path]] = []
    monkeypatch.setattr(
        cli, "_render", lambda _, into: steps.append(("render", into)) or []
    )
    monkeypatch.setattr(
        cli, "_encode", lambda into, quality: steps.append((quality, into)) or []
    )
    monkeypatch.setattr(
        cli.recording, "export", lambda into: steps.append(("export", into)) or []
    )

    result = run("recording", scenario)

    assert result.exit_code == 0, result.output
    into = tmp_path / "sweep"
    assert steps == [("render", into), (cli._QUALITY, into), ("export", into)]


def test_a_render_s_folder_is_only_exported(
    run: Run, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exported: list[Path] = []

    def never(*_: object) -> None:
        raise AssertionError("rendered or encoded a render's folder")

    monkeypatch.setattr(cli, "_render", never)
    monkeypatch.setattr(cli, "_encode", never)
    monkeypatch.setattr(cli.recording, "export", lambda f: exported.append(f) or [])

    assert run("recording", tmp_path).exit_code == 0
    assert exported == [tmp_path]


def test_the_bands_end_as_one_render_of_the_scene(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenarios: set[Path] = set()

    def render_band(scenario: Path, band: str, part: Path) -> None:
        scenarios.add(scenario)
        _write_part(part, band)

    monkeypatch.setattr(processes, "at_once", _in_turn)
    monkeypatch.setattr(cli, "_band", render_band)
    built = load(SWEEP_SENTRY)
    (tmp_path / cli.SCENARIO).write_text("the whole scene's\n")

    written = cli._render_apart(built, tmp_path)

    assert sorted(p.name for p in written) == [
        "calibration.json",
        "eo.png",
        "ir.png",
        "labels.json",
    ]
    assert (tmp_path / cli.SCENARIO).read_text() == "the whole scene's\n"
    # Each band renders the scene the parent pinned, never the source's files.
    assert scenarios == {tmp_path / cli.SCENARIO}
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
        cli.SCENARIO,
    ]


def _write_part(part: Path, band: str) -> None:
    """The folder `part` a band's render leaves: a frame named for `band`, its truth,
    its scenario and its log.
    """
    part.mkdir(exist_ok=True)
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
    (part / cli.SCENARIO).write_text(f'outputs.bands = ["{band}"]\n')


def _camera(band: str) -> CameraCalibration:
    """Any valid calibration of `band`'s camera."""
    mount = next(m for m in load(SWEEP_SENTRY).mounts if m.camera.band == band)
    return CameraCalibration.planned(mount.name, mount.camera, mount.rig_name)


@pytest.mark.parametrize(
    ("chosen", "views"), [([], ("top", "chase")), (["--view", "top"], ("top",))]
)
def test_a_previs_video_is_drawn_beside_the_scenario_at_its_rate(
    run: Run,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    chosen: list[str],
    views: tuple[str, ...],
) -> None:
    scenario = tmp_path / "sweep.toml"
    scenario.write_text(f"extends = {str(SWEEP_SENTRY)!r}\n")
    calls: list[tuple[Path, str, float, tuple[str, ...]]] = []

    def draw(
        _: Scenario,
        folder: Path,
        title: str,
        video_fps: float,
        __: object,
        views: tuple[str, ...],
    ) -> previs.Drawn:
        calls.append((folder, title, video_fps, views))
        return previs.Drawn({"top": folder / "top.png"}, {"top": folder / "top.mp4"})

    monkeypatch.setattr(previs, "draw", draw)

    drawn = run("previs", scenario, "--video", *chosen)

    assert drawn.exit_code == 0, drawn.output
    assert calls == [(tmp_path, "sweep", load(SWEEP_SENTRY).outputs.fps, views)]
    assert drawn.stdout == f"{tmp_path / 'top.png'}\n{tmp_path / 'top.mp4'}\n"


def test_a_previs_rate_without_a_video_is_refused(
    run: Run, monkeypatch: pytest.MonkeyPatch
) -> None:
    def draw(*_: object, **__: object) -> None:
        raise AssertionError("drew a previs it should refuse")

    monkeypatch.setattr(previs, "draw", draw)

    refused = run("previs", SWEEP_SENTRY, "--fps", "5")

    assert refused.exit_code == 2
    assert "--video" in refused.output
