"""Publishing a recording to W&B, against a stand-in for wandb."""

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from click.testing import CliRunner, Result
from PIL import Image

from seascape import cli, logger, previs, rigview
from seascape import render as renderer
from seascape.config import load
from seascape.outputs import labels, provenance
from seascape.render import video
from tests.scenarios import OPEN_SEA, SWEEP_SENTRY


class Artifact:
    def __init__(self, name: str, type: str, **fields: Any) -> None:
        self.name, self.type, self.fields = name, type, fields
        self.files: list[tuple[str, str]] = []
        self.dirs: list[tuple[str, str]] = []

    def add_file(self, path: str, name: str) -> None:
        self.files.append((path, name))

    def add_dir(self, path: str, name: str) -> None:
        self.dirs.append((path, name))


class Config:
    """A run's config, which refuses a changed value unless told it may change."""

    def __init__(self) -> None:
        self.values: dict[str, Any] = {}

    def update(self, values: dict[str, Any], *, allow_val_change: bool) -> None:
        assert allow_val_change
        self.values.update(values)


class FakeRun:
    """What a run records, in place of sending it."""

    def __init__(self, mode: str = "online", **init: Any) -> None:
        self.init = init
        self.offline, self.disabled = mode == "offline", mode == "disabled"
        self.logged: list[dict[str, Any]] = []
        self.config = Config()
        self.summary: dict[str, Any] = {}
        self.used: list[Artifact] = []
        self.outputs: list[Artifact] = []

    def __enter__(self) -> "FakeRun":
        return self

    def __exit__(self, *_: object) -> None:
        pass

    def log(self, values: dict[str, Any]) -> None:
        self.logged.append(values)

    def use_artifact(self, artifact: Artifact) -> None:
        self.used.append(artifact)

    def log_artifact(self, artifact: Artifact) -> SimpleNamespace:
        self.outputs.append(artifact)
        return SimpleNamespace(wait=lambda: None)


class WandbError(Exception):
    pass


def fake_wandb(refusal: str | None = None, mode: str = "online") -> SimpleNamespace:
    """A wandb whose runs are in `runs`, or whose `init` raises `refusal`."""
    runs: list[FakeRun] = []

    def init(**kwargs: Any) -> FakeRun:
        if refusal is not None:
            raise WandbError(refusal)
        runs.append(FakeRun(mode, **kwargs))
        return runs[-1]

    return SimpleNamespace(
        init=init,
        runs=runs,
        errors=SimpleNamespace(Error=WandbError),
        # A picture, or the path of one.
        Image=lambda picture: (
            "image",
            picture if isinstance(picture, str) else picture.size,
        ),
        Video=lambda path: ("video", Path(path).name),
        Artifact=Artifact,
    )


def live(run: FakeRun, name: str = "sweep") -> logger.Run:
    return logger.Run(cast("Any", run), cast("Any", fake_wandb()), name)


@pytest.fixture
def wandb(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """A logged-in wandb, in sys.modules."""
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    fake = fake_wandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    return fake


def cli_run(*args: object) -> Result:
    return CliRunner().invoke(cli.main, [str(a) for a in args], prog_name="seascape")


def rendered(into: Path, commit: str | None) -> Path:
    """A render's labels naming `commit` and its scenario; its recording's folder."""
    labels.Labels(info={"version": "1.2.3", "commit": commit}).write(into)
    (into / cli.SCENARIO).write_text("seed = 1\n")
    recorded = into / "recordings" / "mast" / "Sentry_recordings_ts"
    recorded.mkdir(parents=True)
    return recorded


def test_publishing_logs_the_recording_made_from_its_scenario(tmp_path: Path) -> None:
    recorded = rendered(tmp_path, "abc1234")
    info = labels.Labels.read(tmp_path).info
    run = FakeRun()

    live(run).publish(info, tmp_path / cli.SCENARIO, [recorded])

    (source,) = run.used
    assert (source.name, source.type) == ("sweep-scenario", "scenario")
    assert source.files == [(str(tmp_path / cli.SCENARIO), "sweep.toml")]
    (output,) = run.outputs
    assert (output.name, output.type) == ("sweep", "recording")
    assert output.dirs == [(str(recorded), "mast/Sentry_recordings_ts")]
    assert output.fields["description"] == "seascape 1.2.3 at abc1234"
    assert run.config.values == output.fields["metadata"] == info


@pytest.mark.parametrize(
    ("flags", "env", "project", "entity", "name"),
    [
        ((), {}, "seascape", None, "x"),
        ((), {"WANDB_PROJECT": "p", "WANDB_ENTITY": "e"}, "p", "e", "x"),
        (("--project", "q", "--entity", "f", "--name", "n"), {}, "q", "f", "n"),
    ],
    ids=["defaults", "environment", "flags"],
)
def test_a_published_render_s_folder_logs_its_recording(
    wandb: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    flags: tuple[str, ...],
    env: dict[str, str],
    project: str,
    entity: str | None,
    name: str,
) -> None:
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    folder = tmp_path / "x"
    folder.mkdir()
    recorded = rendered(folder, "abc1234")
    video = recorded / "Sentry_mast_record_ts.mp4"
    monkeypatch.setattr(cli.recording, "export", lambda _: [video])

    result = cli_run("recording", folder, "--publish", *flags)

    assert result.exit_code == 0, result.output
    (run,) = wandb.runs
    assert run.init == {"project": project, "entity": entity, "name": name}
    (output,) = run.outputs
    assert (output.name, output.dirs) == (
        name,
        [(str(recorded), "mast/Sentry_recordings_ts")],
    )


@pytest.mark.parametrize(
    ("wandb_module", "error"),
    [
        (fake_wandb("api_key not configured"), "W&B started no run: api_key"),
        (None, "publishing needs wandb"),
    ],
    ids=["logged out", "not installed"],
)
def test_publish_without_a_run_is_refused(
    wandb_module: SimpleNamespace | None,
    error: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """None in sys.modules makes `import wandb` raise ImportError."""
    monkeypatch.setitem(sys.modules, "wandb", wandb_module)
    rendered(tmp_path, "abc1234")

    def exported(_: Path) -> list[Path]:
        raise AssertionError("exported a recording it cannot publish")

    monkeypatch.setattr(cli.recording, "export", exported)

    result = cli_run("recording", tmp_path, "--publish")

    assert result.exit_code == 1
    assert error in result.output


@pytest.mark.parametrize("commit", ["abc1234-dirty", None])
def test_publishing_a_render_s_folder_from_no_clean_commit_is_refused(
    wandb: SimpleNamespace, tmp_path: Path, commit: str | None
) -> None:
    rendered(tmp_path, commit)

    result = cli_run("recording", tmp_path, "--publish")

    assert result.exit_code == 2
    assert "clean commit" in result.output
    assert not wandb.runs


@pytest.mark.parametrize("commit", ["abc1234-dirty", None])
def test_publishing_a_scenario_from_no_clean_commit_renders_nothing(
    wandb: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    commit: str | None,
) -> None:
    def rendered(*_: object) -> list[Path]:
        raise AssertionError("rendered a scene it cannot publish")

    monkeypatch.setattr(provenance, "commit", lambda: commit)
    monkeypatch.setattr(cli, "_render", rendered)

    result = cli_run("recording", OPEN_SEA, "-o", tmp_path / "x", "--publish")

    assert result.exit_code == 2
    assert "clean commit" in result.output
    assert not wandb.runs


def test_a_render_logs_each_camera_s_frame_each_second(
    tmp_path: Path,
) -> None:
    built = load(OPEN_SEA, ["outputs.duration_s = 2", "outputs.fps = 5"])
    camera = built.mounts[0].name
    frames = [tmp_path / camera / f"{n:04d}.png" for n in range(10)]
    frames[0].parent.mkdir()
    for path in frames:
        Image.new("RGB", (4, 3)).save(path)
    run = FakeRun()
    log = cli._logging(live(run), built)

    for path in [*frames, tmp_path / "radiance.exr"]:
        log(built.mounts[0], path)

    # Frames 0 and 5: one a second at 5 fps.
    assert run.logged == [{f"frame/{camera}": ("image", (4, 3))}] * 2


def test_a_render_logs_its_progress_image_by_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    built = load(OPEN_SEA, ["outputs.duration_s = 1", "outputs.fps = 2"])

    def render(_: object, into: Path, advance: Any) -> list[Path]:
        for _ in range(built.images):
            advance(built.mounts[0], into / "radiance.exr")
        return []

    monkeypatch.setattr(renderer, "render", render)
    run = FakeRun()

    cli._render(built, tmp_path, live(run))

    assert [v["images_done"] for v in run.logged] == list(range(1, built.images + 1))


def test_a_published_scenario_logs_its_previews_and_publishes_without_a_montage(
    wandb: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    into = tmp_path / "x"
    steps: list[str] = []
    monkeypatch.setattr(provenance, "commit", lambda: "abc1234")
    monkeypatch.setattr(cli, "_previews", lambda *_: steps.append("previews"))

    def rendered(_: object, folder: Path, run: object) -> list[Path]:
        assert isinstance(run, logger.Run)
        steps.append("render")
        labels.Labels(info={"version": "1.2.3", "commit": "abc1234"}).write(folder)
        return []

    def montage(*_: object) -> Path:
        raise FileNotFoundError("x: no frames of any camera")

    monkeypatch.setattr(cli, "_render", rendered)
    monkeypatch.setattr(cli, "_encode", lambda *_: [])
    monkeypatch.setattr(video, "montage_video", montage)
    mp4 = into / "recordings" / "mast" / "Sentry_recordings_ts" / "a.mp4"
    monkeypatch.setattr(cli.recording, "export", lambda _: [mp4])

    result = cli_run("recording", OPEN_SEA, "-o", into, "--publish")

    assert result.exit_code == 0, result.output
    assert steps == ["previews", "render"]
    assert "no montage video logged: x: no frames of any camera" in result.stderr
    (run,) = wandb.runs
    assert [a.name for a in run.outputs] == ["x"]


def test_the_previews_log_the_rig_view_and_both_previs_views(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def draw(_: object, folder: Path, *__: object) -> previs.Drawn:
        return previs.Drawn(
            {v: folder / f"{v}.png" for v in previs.VIEWS},
            {v: folder / f"{v}.mp4" for v in previs.VIEWS},
        )

    monkeypatch.setattr(rigview, "draw", lambda *_: None)
    monkeypatch.setattr(previs, "draw", draw)
    run = FakeRun()

    cli._previews(load(SWEEP_SENTRY), tmp_path, live(run))

    assert [next(iter(v)) for v in run.logged] == [
        "rig_view",
        "top",
        "chase",
        "top_video",
        "chase_video",
    ]


def test_a_preview_that_fails_is_skipped_with_a_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    def failed(*_: object) -> None:
        raise RuntimeError("drawing top failed")

    monkeypatch.setattr(previs, "draw", failed)
    run = FakeRun()

    cli._previews(load(OPEN_SEA), tmp_path, live(run))

    assert run.logged == []
    assert "no previews logged: drawing top failed" in capsys.readouterr().err


def test_a_video_is_logged_by_its_file_name(tmp_path: Path) -> None:
    run = FakeRun()

    live(run).log("top_video", tmp_path / "top.mp4")

    assert run.logged == [{"top_video": ("video", "top.mp4")}]


def test_a_name_without_publish_is_refused(tmp_path: Path) -> None:
    result = cli_run("recording", OPEN_SEA, "-o", tmp_path / "x", "--name", "n")

    assert result.exit_code == 2
    assert "--name needs --publish" in result.output


def test_publishing_into_a_folder_not_empty_starts_no_run(
    wandb: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(provenance, "commit", lambda: "abc1234")
    (tmp_path / "earlier.png").touch()

    result = cli_run("recording", OPEN_SEA, "-o", tmp_path, "--publish")

    assert result.exit_code == 1
    assert "not empty" in result.output
    assert not wandb.runs


@pytest.mark.parametrize(
    ("built_at", "after", "moved"),
    [
        ("abc1234", "abc1234-dirty", "abc1234-dirty"),
        ("abc1234", "def5678", "def5678"),
        ("abc1234-dirty", "abc1234", "abc1234-dirty"),
    ],
    ids=["a file edited", "another commit", "an edit undone before the end"],
)
def test_a_render_whose_checkout_moved_is_not_published(
    wandb: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    built_at: str,
    after: str,
    moved: str,
) -> None:
    commits = iter(["abc1234", after])
    monkeypatch.setattr(provenance, "commit", lambda: next(commits))
    monkeypatch.setattr(cli, "_previews", lambda *_: None)

    def rendered(_: object, folder: Path, *__: object) -> list[Path]:
        labels.Labels(info={"commit": built_at}).write(folder)
        return []

    monkeypatch.setattr(cli, "_render", rendered)
    monkeypatch.setattr(cli, "_encode", lambda *_: [])
    monkeypatch.setattr(cli, "_log_montage", lambda *_: None)
    monkeypatch.setattr(cli.recording, "export", lambda _: [])

    result = cli_run("recording", OPEN_SEA, "-o", tmp_path / "x", "--publish")

    assert result.exit_code == 1
    assert f"to {moved} while" in result.output
    (run,) = wandb.runs
    assert not run.outputs


@pytest.mark.parametrize(
    ("folder", "flags"),
    [("sweep", ("--name", "sweep test")), ("sweep test", ())],
    ids=["--name", "the folder's name"],
)
def test_a_name_w_and_b_refuses_renders_nothing(
    wandb: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    folder: str,
    flags: tuple[str, ...],
) -> None:
    def rendered(*_: object) -> list[Path]:
        raise AssertionError("rendered a scene it cannot publish")

    monkeypatch.setattr(provenance, "commit", lambda: "abc1234")
    monkeypatch.setattr(cli, "_render", rendered)

    result = cli_run(
        "recording", OPEN_SEA, "-o", tmp_path / folder, "--publish", *flags
    )

    assert result.exit_code == 2
    assert "cannot name an artifact" in result.output
    assert not wandb.runs


@pytest.mark.parametrize("mode", ["offline", "disabled"])
@pytest.mark.parametrize("source", ["scenario", "folder"])
def test_a_run_that_cannot_publish_is_refused_before_the_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, source: str
) -> None:
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb(mode=mode))
    monkeypatch.setattr(provenance, "commit", lambda: "abc1234")
    rendered(tmp_path, "abc1234")

    def started(*_: object) -> list[Path]:
        raise AssertionError("went on with a run that cannot publish")

    monkeypatch.setattr(cli, "_record", started)
    monkeypatch.setattr(cli.recording, "export", started)
    args = (OPEN_SEA, "-o", tmp_path / "x") if source == "scenario" else (tmp_path,)

    result = cli_run("recording", *args, "--publish")

    assert result.exit_code == 1
    assert mode in result.output
