"""Command line entry point."""

import json
import math
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Callable, Generator, Iterable
from contextlib import contextmanager, nullcontext, redirect_stdout
from itertools import chain
from pathlib import Path
from typing import Any, get_args

import click

from seascape import assets, logger, outputs, processes
from seascape.assets import skies
from seascape.config import (
    BANDS,
    Band,
    Mount,
    Quality,
    Scenario,
    dump,
    instantiate,
    json_schema,
    load,
    resolve,
    seeds,
)
from seascape.outputs import (
    agc,
    labels,
    montage,
    panorama,
    preview,
    provenance,
    recording,
)
from seascape.outputs.calibration import FILENAME as CALIBRATION
from seascape.outputs.calibration import Calibration, Frame
from seascape.outputs.previs import VIEWS, View

_BANDS = click.Choice(BANDS)
_FILE = click.Path(exists=True, dir_okay=False, path_type=Path)
_FOLDER = click.Path(exists=True, file_okay=False, path_type=Path)
_BLENDER_LOG = "blender.log"


class _Seascape(click.Group):
    def invoke(self, ctx: click.Context) -> Any:
        try:
            return super().invoke(ctx)
        # Click's own exit, after --help in a subcommand, is a RuntimeError too.
        except (click.exceptions.Exit, click.ClickException, click.Abort):
            raise
        except (
            OSError,
            ValueError,  # the loader, pydantic and tomllib raise it
            RuntimeError,  # bpy.ops.render.render, e.g. an unwritable output directory
        ) as error:
            # A traceback buries a scenario mistake.
            raise click.ClickException(str(error)) from error


def _scenario(
    output: str, kind: click.Path = _FILE
) -> Callable[[Callable[..., None]], Callable[..., None]]:
    """The scenario argument and its `--output` and `--set`; `output` is its help,
    `kind` what the argument may name.
    """

    def add(command: Callable[..., None]) -> Callable[..., None]:
        command = click.option(
            "--set",
            "overrides",
            multiple=True,
            metavar="KEY=VALUE",
            help="Override a field as TOML: 'rigs.bow.pitch_deg = -5'. Repeatable.",
        )(command)
        command = click.option(
            "-o", "--output", type=click.Path(path_type=Path), help=output
        )(command)
        return click.argument("scenario", type=kind)(command)

    return add


@contextmanager
def _blender_log(path: Path) -> Generator[None]:
    """Send Blender's log to a file. Blender logs to `sys.stdout` from Python and to
    file descriptor 1 from C, where either would tear the progress bar and mix into
    stdout.

    Fails where Cycles drew a shader black: past its SVM stack it logs, renders on,
    and returns no error.

    Args:
        path: The log, appended to; its folder is made if missing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout.flush()
    stdout = os.dup(1)
    with path.open("a") as log, redirect_stdout(log):
        start = log.tell()
        os.dup2(log.fileno(), 1)
        try:
            yield
        finally:
            log.flush()
            os.dup2(stdout, 1)
            os.close(stdout)
    with path.open(errors="replace") as log:
        log.seek(start)
        if "out of SVM stack space" in log.read():
            raise click.ClickException(
                f"Cycles ran out of SVM stack and drew a shader black: see {path.name}"
            )


def _progress(length: int, label: str) -> Any:
    return click.progressbar(length=length, label=label, file=sys.stderr)


def _steps(bar: Any) -> Callable[[int, int], None]:
    """Show `done` of `total` steps on `bar`, whose length need not be known yet."""

    def step(done: int, total: int) -> None:
        bar.length = total
        bar.update(done - bar.pos)

    return step


def _empty(into: Path) -> Path:
    # An earlier run's frames would pass for this one's.
    if into.is_dir() and any(into.iterdir()):
        raise click.ClickException(f"{into} is not empty: remove it or pick another")
    return into


SCENARIO = "scenario.toml"


def _pin(
    scenario: Path, overrides: Iterable[str], into: Path
) -> tuple[dict[str, Any], Scenario]:
    """Load `scenario` with `overrides`, and write its resolved tree into `into`."""
    tree = resolve(scenario, overrides)
    built = instantiate(tree)
    into.mkdir(parents=True, exist_ok=True)
    dump(tree, into / SCENARIO)
    return tree, built


def _previews(built: Scenario, into: Path, run: logger.Run) -> None:
    """Log what `rig-view` draws, when the ownship has a hull to draw it on, and what
    `previs --video` draws.

    Drawn before the render builds its scene: each resets Blender to its factory
    settings, and so does every build, which then picks its render device.
    """
    from seascape import previs, rigview

    with tempfile.TemporaryDirectory() as tmp, _blender_log(into / _BLENDER_LOG):
        try:
            if built.ownship.asset is not None:
                image = Path(tmp) / "rig-view.png"
                rigview.draw(built, image)
                run.log("rig_view", image)
            with _progress(0, "Drawing") as bar:
                drawn = previs.draw(
                    built, Path(tmp), "preview", built.outputs.fps, _steps(bar)
                )
        # The render is what is published; it goes on without its previews.
        except (OSError, RuntimeError, ValueError) as error:
            click.echo(f"no previews logged: {error}", err=True)
            return
        for view, path in drawn.stills.items():
            run.log(view, path)
        for view, path in drawn.videos.items():
            run.log(f"{view}_video", path)


def _logging(run: logger.Run | None, built: Scenario) -> Callable[[Mount, Path], None]:
    """Log each camera's frame about once a second of the scenario's time, under
    `frame/<camera>`: a thermal one toned as its camera's AGC sets.
    """
    every = max(1, round(built.outputs.fps))

    def log(mount: Mount, image: Path) -> None:
        if run is None or image.suffix == ".exr":  # radiance, not a picture
            return
        # A sequence's frame is `<camera>/<frame>`, a still `<camera>`: `stem`. A
        # cropped video's frames carry the video's name.
        frame = image.stem
        camera, number = (
            (image.parent.name, int(frame)) if frame.isdigit() else (frame, 0)
        )
        if number % every == 0:
            picture = preview.picture(image, agc.Agc(mount.camera.agc))
            run.log(f"frame/{camera}", picture)

    return log


def _advance(bar: Any, run: logger.Run | None, done: int) -> None:
    """Show `done` images on `bar`, and log its pace to `run`."""
    bar.update(done - bar.pos)
    if run is not None:
        run.progress(done=bar.pos, s_per_image=bar.time_per_iteration, eta_s=bar.eta)


def _render(built: Scenario, into: Path, run: logger.Run | None = None) -> list[Path]:
    from seascape import render as renderer

    log = _logging(run, built)

    def advance(mount: Mount, image: Path) -> None:
        _advance(bar, run, bar.pos + 1)
        log(mount, image)

    with (
        _progress(built.images, "Rendering") as bar,
        _blender_log(into / _BLENDER_LOG),
    ):
        return renderer.render(built, into, advance)


def _band(scenario: Path, band: Band, part: Path) -> None:
    """Render `band` of the scene pinned at `scenario` into `part`, as `seascape
    render` would.
    """
    from seascape import render as renderer

    _, built = _pin(scenario, [f'outputs.bands = ["{band}"]'], part)
    with _blender_log(part / _BLENDER_LOG):
        renderer.render(built, part)


_PARALLEL = click.option(
    "--parallel",
    is_flag=True,
    help="Render each band in its own process at once. Faster when one band leaves "
    "the GPU idle; needs one render's memory per band.",
)


def _render_apart(
    built: Scenario, into: Path, run: logger.Run | None = None
) -> list[Path]:
    """Render each band of `into`'s `scenario.toml` in a process of its own, all at
    once, then gather their frames and truth in `into` as one render writes
    them; the files written. A scene with one band renders here.

    One render leaves the GPU idle while Cycles readies the scene and seascape labels
    a frame; another band's render fills that time.
    """
    bands = [band for band in built.outputs.bands if built.band_mounts(band)]
    if len(bands) < 2:
        return _render(built, into, run)
    into.mkdir(parents=True, exist_ok=True)
    parts = {band: into / f".{band}" for band in bands}
    logs = {band: into / f".{band}.log" for band in bands}
    jobs = {band: (into / SCENARIO, band, part) for band, part in parts.items()}
    shown = ", ".join(map(str, logs.values()))
    click.echo(f"Rendering {', '.join(bands)} at once; see {shown}", err=True)
    fmt = built.outputs.format
    with _progress(built.images, "Rendering") as bar:

        def tick() -> None:
            done = sum(
                len(outputs.written(parts[b], m.name, fmt))
                for b in bands
                for m in built.band_mounts(b)
            )
            # --parallel logs progress only: a band's newest frame may be half
            # written.
            if done > bar.pos:
                _advance(bar, run, done)

        failed = processes.at_once(_band, jobs, logs, tick)
        if not failed:
            _advance(bar, run, built.images)
    if failed:
        raise click.ClickException(
            f"rendering {', '.join(failed)} failed: see "
            f"{', '.join(str(logs[b]) for b in failed)}; {into} keeps every part"
        )
    # One render loops band by band, so the parts join in `bands` order.
    truth = labels.concat({b: labels.Labels.read(part) for b, part in parts.items()})
    # Each part's scenario renders one band; the truth records the whole scene's.
    truth.info["scenario"] = built.model_dump(mode="json")
    calibrations = [Calibration.read(part) for part in parts.values()]
    written = [
        Calibration(
            cameras=list(chain.from_iterable(c.cameras for c in calibrations)),
            raw=list(chain.from_iterable(c.raw for c in calibrations)),
        ).write(into),
        truth.write(into),
    ]
    # The whole scene's copies are already in `into`.
    own = {labels.FILENAME, CALIBRATION, SCENARIO, _BLENDER_LOG}
    with (into / _BLENDER_LOG).open("ab") as joined:
        for band, part in parts.items():
            with (part / _BLENDER_LOG).open("rb") as log:
                shutil.copyfileobj(log, joined)
            for item in part.iterdir():
                if item.name not in own:
                    written.append(item.rename(into / item.name))
            shutil.rmtree(part)
            logs[band].unlink()
    return written


_QUALITY: Quality = "PERC_LOSSLESS"


def _encode(folder: Path, quality: Quality) -> list[Path]:
    from seascape.render import video as encoder

    with (
        _progress(len(encoder.cameras(folder)), "Encoding") as bar,
        _blender_log(folder / _BLENDER_LOG),
    ):
        return encoder.encode(folder, quality, lambda: bar.update(1))


@click.group(cls=_Seascape)
@click.version_option(package_name="seascape")
def main() -> None:
    """Synthetic maritime camera frames, with their calibration and labels."""


@main.command()
@_scenario("The .blend. Default: beside the scenario.")
# A scene is one band or the other: EO and LWIR share no units.
@click.option("--band", type=_BANDS, default="eo", show_default=True)
def build(
    scenario: Path, output: Path | None, overrides: tuple[str, ...], band: Band
) -> None:
    """Write a .blend from a scenario."""
    built = load(scenario, list(overrides))
    # Deferred: `schema` and a failed validation should not wait for bpy to load.
    import bpy

    from seascape.blender import scene, sea

    path = output or scenario.with_suffix(f".{band}.blend")
    with _blender_log(path.with_suffix(".log")):
        scene.build(built, band)
        sea.keep_foam(path.resolve())
        # The foam's frames are already relative to the .blend's folder.
        bpy.ops.wm.save_as_mainfile(filepath=str(path.resolve()), relative_remap=False)
    mounts = built.mounts
    bands = ", ".join(sorted({mount.camera.band for mount in mounts}))
    heights = ", ".join(f"{built.head_height_m(name)} m" for name in built.rigs)
    click.echo(f"{path}: {band}, {len(mounts)} cameras ({bands}) at {heights}")


@main.command()
@_scenario("A directory. Default: beside the scenario.")
@click.option(
    "--variants",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="Scenes from consecutive seeds, in a folder per seed when there are several.",
)
@_PARALLEL
def render(
    scenario: Path,
    output: Path | None,
    overrides: tuple[str, ...],
    variants: int,
    parallel: bool,
) -> None:
    """Write one image per camera and frame, and their labels. With --variants, a seed
    whose hulls meet is skipped for the next, and the labels merge into one file.
    """
    if parallel and variants > 1:
        raise click.UsageError("--parallel renders one variant")
    into = _empty(output or scenario.with_suffix(""))
    # Once, at the root: the seeds' own values are in their labels.
    tree, built = _pin(scenario, overrides, into)
    from seascape import render as renderer
    from seascape.blender.vessels import OverlapError

    if variants == 1:
        written = (_render_apart if parallel else _render)(built, into)
    else:
        written, folders = [], []
        with (
            _progress(variants, "Rendering") as bar,
            _blender_log(into / _BLENDER_LOG),
        ):
            tries = seeds(built.seed, variants)
            for seed in tries:
                folder = into / str(seed)
                variant = instantiate(tree, seed)
                try:
                    written += renderer.render(variant, folder)
                except OverlapError as error:
                    folder.rmdir()  # the build refuses before anything is written
                    click.echo(f"\nseed {variant.seed} skipped: {error}", err=True)
                    continue
                folders.append(folder)
                bar.update(1)
                if len(folders) == variants:
                    break
            else:
                raise click.ClickException(
                    f"{len(folders)} of {variants} variants in {len(tries)} seeds: "
                    "hulls meet in the rest"
                )
        written.append(labels.merge(into, folders).write(into))
    for path in written:
        click.echo(path)
    click.echo(f"{len(written)} files in {into}", err=True)


@main.command("montage")
@_scenario("The directory the frames are in. Default: beside the scenario.")
def montage_(scenario: Path, output: Path | None, overrides: tuple[str, ...]) -> None:
    """Lay rendered frames out for review."""
    built = load(scenario, list(overrides))
    click.echo(montage.compose(built, output or scenario.with_suffix("")))


@main.command("rig-view")
@_scenario("The image. Default: beside the scenario.")
def rig_view(scenario: Path, output: Path | None, overrides: tuple[str, ...]) -> None:
    """Draw the ownship with each rig camera's view on it, to confirm a mount."""
    built = load(scenario, list(overrides))
    if built.ownship.asset is None:
        raise click.ClickException("the ownship has no asset to show its rigs on")
    from seascape import rigview

    path = output or scenario.with_name(f"{scenario.stem}-rig.png")
    with _blender_log(Path(os.devnull)):
        rigview.draw(built, path)
    click.echo(path)


@main.command("previs")
@_scenario("The folder the pictures go in. Default: beside the scenario.")
@click.option(
    "--video",
    is_flag=True,
    help="Also replay each view over the run as an mp4 beside it, in real time.",
)
@click.option(
    "--view",
    "views",
    type=click.Choice(VIEWS),
    multiple=True,
    help="A view to draw; repeat for more. Default: every view.",
)
@click.option(
    "--fps",
    type=click.FloatRange(min=0.0, min_open=True),
    help="With --video, frames a second, rounded to a whole step of the scenario's "
    "rate. Default: the scenario's rate.",
)
def previs_(
    scenario: Path,
    output: Path | None,
    overrides: tuple[str, ...],
    video: bool,
    fps: float | None,
    views: tuple[View, ...],
) -> None:
    """Draw the scenario from above and from behind the ownship, each object's range
    ringed and each camera's field shaded.
    """
    if fps is not None and not video:
        raise click.UsageError("--fps needs --video")
    built = load(scenario, list(overrides))
    from seascape import previs

    video_fps = (fps or built.outputs.fps) if video else None
    with _blender_log(Path(os.devnull)), _progress(0, "Drawing") as bar:
        drawn = previs.draw(
            built,
            output or scenario.parent,
            scenario.stem,
            video_fps=video_fps,
            progress=_steps(bar),
            views=views or VIEWS,
        )
    for path in [*drawn.stills.values(), *drawn.videos.values()]:
        click.echo(path)


@main.command("panorama")
@click.argument("folder", type=_FOLDER)
@click.option(
    "--projection",
    type=click.Choice(list(panorama.PROJECTIONS)),
    default="cylindrical",
    show_default=True,
)
@click.option(
    "--frame",
    type=click.Choice(get_args(Frame.__value__)),
    default="world",
    show_default=True,
    help="An extrinsics frame in calibration.json, which sets what is level: world "
    "the horizon, ownship the hull, head the rig's head, panned and tilted.",
)
@click.option("--max-width", type=int, help="Pixels; native resolution when absent.")
@click.option("--ruler", is_flag=True, help="A strip of bearing ticks under the image.")
@click.option(
    "--overlay",
    is_flag=True,
    help="One image per rig: every band laid over the first, each frame outlined "
    "and named.",
)
def panorama_(
    folder: Path,
    projection: str,
    frame: Frame,
    max_width: int | None,
    ruler: bool,
    overlay: bool,
) -> None:
    """Stitch each rig's frames, from calibration.json."""
    for path in panorama.panoramas(
        folder, projection, frame, max_width, ruler, overlay=overlay
    ):
        click.echo(path)


@main.command()
@click.argument("folder", type=_FOLDER)
@click.option(
    "--quality",
    type=click.Choice(get_args(Quality.__value__), case_sensitive=False),
    default=_QUALITY,
    show_default=True,
    help="Blender's H.264 constant-quality preset.",
)
def video(folder: Path, quality: Quality) -> None:
    """Encode each camera's frames as an mp4, from labels.json."""
    for path in _encode(folder, quality):
        click.echo(path)


@main.command("recording")
@_scenario(
    "A directory. Default: beside the scenario.",
    click.Path(exists=True, path_type=Path),
)
@_PARALLEL
@click.option(
    "--publish",
    is_flag=True,
    help="Also log the recording, and the scenario as its input, as W&B artifacts. "
    "Needs a render from a clean commit.",
)
@click.option(
    "--project",
    envvar="WANDB_PROJECT",
    default="seascape",
    show_default=True,
    help="With --publish, the W&B project.",
)
@click.option(
    "--entity",
    envvar="WANDB_ENTITY",
    help="With --publish, the W&B team or user. Default: wandb's default entity.",
)
@click.option(
    "--name",
    help="With --publish, the W&B run's and artifacts' name. Default: the render's "
    "folder's.",
)
def recording_(
    scenario: Path,
    output: Path | None,
    overrides: tuple[str, ...],
    parallel: bool,
    publish: bool,
    project: str,
    entity: str | None,
    name: str | None,
) -> None:
    """Lay each rig's videos out as a recording. SCENARIO is a render's folder, after
    video, or a scenario to render and encode first.
    """
    rendering = scenario.is_file()
    if not rendering and (output or overrides or parallel):
        raise click.UsageError(
            "a render's folder takes neither -o, --set nor --parallel"
        )
    if name is not None and not publish:
        raise click.UsageError("--name needs --publish")
    folder = (output or scenario.with_suffix("")) if rendering else scenario
    if publish:
        # A render records this commit: refuse before hours of rendering.
        commit = (
            provenance.commit()
            if rendering
            else labels.Labels.read(folder).info.get("commit")
        )
        if not provenance.clean(commit):
            raise click.UsageError(
                f"--publish needs a render from a clean commit, not {commit}"
            )
    else:
        commit = None
    if rendering:
        _empty(folder)
    # `Path(".").name` is empty.
    title = name or folder.resolve().name
    # W&B's rule for an artifact's name, checked before hours of rendering.
    if publish and not re.fullmatch(r"[A-Za-z0-9_.-]+", title):
        raise click.UsageError(
            f"{title!r} cannot name an artifact: use letters, digits, '-', '_' or '.'"
        )
    with logger.run(project, entity, title) if publish else nullcontext() as run:
        if rendering:
            _record(scenario, overrides, folder, parallel, run, commit)
        written = recording.export(folder)
        for path in written:
            click.echo(path)
        if run is not None:
            _log_montage(folder, run)
            recorded = sorted({p.parent for p in written if p.suffix == ".mp4"})
            info = labels.Labels.read(folder).info
            run.publish(info, folder / SCENARIO, recorded)


def _record(
    scenario: Path,
    overrides: tuple[str, ...],
    folder: Path,
    parallel: bool,
    run: logger.Run | None,
    commit: str | None,
) -> None:
    """Render `scenario` into `folder` and encode its videos; with `run`, log the
    previews and progress, then refuse to publish if the checkout left `commit`, by
    an edit or a checkout, while it rendered.
    """
    _, built = _pin(scenario, overrides, folder)
    if run is not None:
        _previews(built, folder, run)
    (_render_apart if parallel else _render)(built, folder, run)
    _encode(folder, _QUALITY)
    if run is not None and (now := provenance.commit()) != commit:
        raise click.ClickException(
            f"the checkout went from {commit} to {now} while it rendered: not published"
        )


def _log_montage(folder: Path, run: logger.Run) -> None:
    """Log every video in `folder` laid out in one, warning rather than failing: the
    recording publishes without it.
    """
    from seascape.render import video as encoder

    with tempfile.TemporaryDirectory() as tmp, _blender_log(folder / _BLENDER_LOG):
        try:
            shown = load(folder / SCENARIO)
            out = encoder.montage_video(shown, folder, Path(tmp) / "montage.mp4")
            run.log("montage", out)
        # A failed encode raises RuntimeError.
        except (OSError, RuntimeError, ValueError) as error:
            click.echo(f"no montage video logged: {error}", err=True)


@main.command()
def schema() -> None:
    """Print the scenario JSON schema."""
    click.echo(json.dumps(json_schema(), indent=2))


@main.command()
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8765, show_default=True)
@click.option(
    "--scenarios",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path(__file__).parents[1] / "scenarios",
    envvar="SEASCAPE_SCENARIOS",
    show_default=True,
    help="The folder of scenarios clients render.",
)
def serve(host: str, port: int, scenarios: Path) -> None:
    """Render for MCP clients at http://HOST:PORT/mcp; see SEASCAPE_*."""
    from seascape import server

    settings = server.Settings(scenarios=scenarios)
    # Out of the environment, which the GPU probe and every render inherit.
    os.environ.pop("SEASCAPE_MCP_SECRET", None)
    server.serve(server.State(settings), host, port)


@main.group("assets")
def assets_() -> None:
    """The meshes and skies a scenario can name."""


def _table(header: tuple[str, ...], rows: list[tuple[str, ...]]) -> None:
    widths = [max(map(len, column)) for column in zip(header, *rows, strict=True)]
    right = [
        all(c.replace(",", "").isdigit() for c in col)
        for col in zip(*rows, strict=True)
    ]

    def line(cells: tuple[str, ...]) -> str:
        padded = (
            cell.rjust(w) if r else cell.ljust(w)
            for cell, w, r in zip(cells, widths, right, strict=True)
        )
        return "  ".join(padded).rstrip()

    click.secho(line(header), bold=True)
    for row in rows:
        click.echo(line(row))


def _cached(here: bool) -> str:
    return "yes" if here else "-"


@assets_.command("list")
def list_() -> None:
    """List every mesh and sky, and whether it is cached."""
    meshes = assets.manifest()
    _table(
        ("MESH", "KIND", "CLASS", "SIZE", "TRIANGLES", "TEXTURES", "LICENCE", "CACHED"),
        [
            (name, *mesh.row(), _cached(assets.cached(name)))
            for name, mesh in meshes.items()
        ],
    )
    click.echo()
    photos = skies.library()
    order = sorted(
        photos,
        key=lambda n: math.inf if (e := photos[n].sun_elevation_deg) is None else -e,
    )
    _table(
        ("SKY", "LOOK", "SUN ELEV", "LICENCE", "CACHED"),
        [
            (
                name,
                photos[name].label,
                photos[name].sun,
                photos[name].licence,
                _cached(assets.cache_path(name, photos[name].url).exists()),
            )
            for name in order
        ],
    )
    click.echo("\nMeshes go in a scenario's `asset`, skies in `sky.hdri`.")


def _names(_ctx: click.Context, _param: click.Parameter, incomplete: str) -> list[str]:
    return [
        n for n in [*assets.manifest(), *skies.library()] if n.startswith(incomplete)
    ]


@assets_.command()
@click.argument("name", shell_complete=_names)
def show(name: str) -> None:
    """Describe one mesh or sky."""
    if name in assets.manifest():
        mesh = assets.manifest()[name]
        fields = {
            "kind": mesh.kind,
            "class": f"{mesh.supercategory} / {mesh.category}",
            "size": mesh.size,
            "draught": f"{mesh.draught_m} m",
            "triangles": f"{mesh.triangles:,}",
            "textures": mesh.textures,
            "description": mesh.description,
            "licence": mesh.licence,
            "attribution": mesh.attribution,
            "url": mesh.url,
            "cached": str(assets.local(name)) if assets.cached(name) else "-",
        }
    elif name in skies.library():
        photo = skies.library()[name]
        path = assets.cache_path(name, photo.url)
        fields = {
            "look": photo.label,
            "cloud cover": f"{photo.cover_oktas} oktas, {photo.cloud_genus}",
            "sun elevation": photo.sun,
            "sun bearing": f"{photo.sun_bearing_deg:.1f} deg",
            "licence": photo.licence,
            "attribution": photo.attribution,
            "url": photo.url,
            "cached": str(path) if path.exists() else "-",
        }
    else:
        raise click.BadParameter(f"no mesh or sky {name!r}; see `seascape assets list`")
    width = max(map(len, fields))
    for key, value in fields.items():
        click.echo(f"{click.style(key.ljust(width), bold=True)}  {value}")


@assets_.command()
@click.argument("mesh", type=_FILE)
def measure(mesh: Path) -> None:
    """Print the manifest lines measured from a mesh file."""
    from seascape import catalog

    with _blender_log(Path(os.devnull)):
        lines = catalog.measure(mesh)
    click.echo(lines)


@assets_.command()
@click.argument("asset", shell_complete=_names)
@click.option(
    "--step-m",
    type=click.FloatRange(min=0, min_open=True),
    default=0.5,
    show_default=True,
    help="Station length along the hull.",
)
def top(asset: str, step_m: float) -> None:
    """Print a hull's highest point per station along its length.

    To mount a rig on a hardtop or radar arch, pick its station: offset_y_m is its
    y_m, offset_x_m its x_m, height_m just above its top_z_m.
    """
    if asset not in assets.manifest():
        raise click.BadParameter(f"no mesh {asset!r}; see `seascape assets list`")
    from seascape.blender import meshes

    with _blender_log(Path(os.devnull)):
        rows = meshes.top(asset, step_m)
    click.echo("y_m     top_z_m  x_m")
    for y_m, z_m, x_m in rows:
        click.echo(f"{y_m:6.2f}  {z_m:6.2f}  {x_m:6.2f}")


@assets_.command()
@click.argument("scenario", type=_FILE)
@click.argument("out", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--band", type=_BANDS, default="eo", show_default=True)
def sheet(scenario: Path, out: Path, band: Band) -> None:
    """Render every mesh in a scenario's sea and sky, one tile each."""
    from seascape import catalog

    with _blender_log(out.with_suffix(".log")):
        catalog.sheet(scenario, out, band)
    click.echo(out)


@main.command()
@click.argument("asset", shell_complete=_names)
@click.option(
    "-o",
    "--output",
    type=click.Path(file_okay=False, path_type=Path),
    help="A directory. Default: zones/ASSET.",
)
@click.option(
    "--islands",
    is_flag=True,
    help="Sheets per island, not per material slot: to split a slot of two zones.",
)
@click.option(
    "--material",
    help="Mark only this material's islands, the rest grey. Implies --islands.",
)
@click.option(
    "--preview",
    "contact",
    is_flag=True,
    help="One sheet coloured by the zones in assets.toml, for approval.",
)
def zones(
    asset: str,
    output: Path | None,
    islands: bool,
    material: str | None,
    contact: bool,
) -> None:
    """Render the sheets for giving an asset's parts their thermal zones."""
    if asset not in assets.manifest():
        raise click.BadParameter(f"no mesh {asset!r}; see `seascape assets list`")
    if contact and (islands or material):
        raise click.UsageError("--preview takes neither --islands nor --material")
    from seascape import zones as sheets

    into = output or Path("zones") / asset
    if not contact:
        _empty(into)
    with _blender_log(into / _BLENDER_LOG):
        if contact:
            written = [sheets.contact(asset, into / "zones.png")]
        else:
            kind = "islands" if islands or material else None
            written = sheets.sheets(asset, into, kind, material)
    for path in written:
        click.echo(path)


if __name__ == "__main__":
    main()
