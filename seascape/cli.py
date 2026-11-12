"""Command line entry point."""

import json
import math
import os
import sys
from collections.abc import Callable, Generator
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from typing import Any, get_args

import click

from seascape import assets
from seascape.assets import skies
from seascape.config import Band, Scenario, json_schema, load, seeds
from seascape.outputs import labels, montage, panorama, recording
from seascape.outputs.calibration import Frame

_BANDS = click.Choice(get_args(Band.__value__))
_FILE = click.Path(exists=True, dir_okay=False, path_type=Path)
_FOLDER = click.Path(exists=True, file_okay=False, path_type=Path)


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


def _scenario(output: str) -> Callable[[Callable[..., None]], Callable[..., None]]:
    """The scenario argument and its `--output` and `--set`; `output` is its help."""

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
        return click.argument("scenario", type=_FILE)(command)

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


def _empty(into: Path) -> Path:
    # An earlier run's frames would pass for this one's.
    if into.is_dir() and any(into.iterdir()):
        raise click.ClickException(f"{into} is not empty: remove it or pick another")
    return into


def _render(built: Scenario, into: Path) -> list[Path]:
    from seascape import render as renderer

    with (
        _progress(built.images, "Rendering") as bar,
        _blender_log(into / "blender.log"),
    ):
        return renderer.render(built, into, lambda: bar.update(1))


_QUALITY = "perc_lossless"


def _encode(folder: Path, quality: str) -> list[Path]:
    from seascape.render import video as encoder

    with (
        _progress(len(encoder.cameras(folder)), "Encoding") as bar,
        _blender_log(folder / "blender.log"),
    ):
        return encoder.encode(folder, quality.upper(), lambda: bar.update(1))


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
    heights = ", ".join(f"{rig.height_m} m" for rig in built.rigs.values())
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
def render(
    scenario: Path, output: Path | None, overrides: tuple[str, ...], variants: int
) -> None:
    """Write one image per camera and frame, and their labels. With --variants, a seed
    whose hulls meet is skipped for the next, and the labels merge into one file.
    """
    built = load(scenario, list(overrides))
    into = _empty(output or scenario.with_suffix(""))
    from seascape import render as renderer
    from seascape.blender.vessels import OverlapError

    if variants == 1:
        written = _render(built, into)
    else:
        written, folders = [], []
        with (
            _progress(variants, "Rendering") as bar,
            _blender_log(into / "blender.log"),
        ):
            tries = seeds(built.seed, variants)
            for seed in tries:
                folder = into / str(seed)
                variant = load(scenario, [*overrides, f"seed = {seed}"])
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


@main.command()
@_scenario("A directory. Default: beside the scenario.")
@click.option(
    "--preview",
    is_flag=True,
    help="Render at 1 fps and one sample, and check every video sees a buoy of each "
    "of its rings whole and the container at both ends; nothing is encoded.",
)
@click.option(
    "--sheet",
    is_flag=True,
    help="Draw the course on one image, <scenario>-course.png in the directory, and "
    "render nothing.",
)
def sync(
    scenario: Path,
    output: Path | None,
    overrides: tuple[str, ...],
    preview: bool,
    sheet: bool,
) -> None:
    """Render, encode and record a sync scenario, preview it, or draw its course."""
    from seascape.sync import preview as checks

    if sheet and preview:
        raise click.UsageError("--sheet and --preview are exclusive")
    if sheet:
        from seascape.sync.sheet import draw

        built = load(scenario, list(overrides))
        into = output or scenario.parent
        into.mkdir(parents=True, exist_ok=True)
        path = into / f"{scenario.stem}-course.png"
        with _blender_log(Path(os.devnull)):
            drawn = draw(built, path)
        click.echo(drawn)
        return
    if not preview:
        built = load(scenario, list(overrides))
        into = _empty(output or scenario.with_suffix(""))
        _render(built, into)
        _encode(into, _QUALITY)
        for path in recording.export(into):
            click.echo(path)
        return
    built = load(scenario, [*overrides, checks.PREVIEW])
    if not checks.placed(built):
        raise click.ClickException(checks.advice(built, f"{scenario} holds no buoys"))
    into = _empty(output or scenario.with_name(f"{scenario.stem}-preview"))
    _render(built, into)
    try:
        checks.check(labels.Labels.read(into), built)
    except ValueError as missed:
        raise click.ClickException(
            checks.advice(built, f"{scenario}: {missed}")
        ) from missed
    click.echo(f"every video sees the course: {into}")


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
def panorama_(
    folder: Path, projection: str, frame: Frame, max_width: int | None, ruler: bool
) -> None:
    """Stitch each rig's frames, from calibration.json."""
    for path in panorama.panoramas(folder, projection, frame, max_width, ruler):
        click.echo(path)


@main.command()
@click.argument("folder", type=_FOLDER)
@click.option(
    "--quality",
    type=click.Choice(
        ["lossless", "perc_lossless", "high", "medium", "low", "verylow", "lowest"]
    ),
    default=_QUALITY,
    show_default=True,
    help="Blender's H.264 constant-quality preset.",
)
def video(folder: Path, quality: str) -> None:
    """Encode each camera's frames as an mp4, from labels.json."""
    for path in _encode(folder, quality):
        click.echo(path)


@main.command("recording")
@click.argument("folder", type=_FOLDER)
def recording_(folder: Path) -> None:
    """Lay each rig's videos out as a recording, after video."""
    for path in recording.export(folder):
        click.echo(path)


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
    with _blender_log(into / "blender.log"):
        if contact:
            written = [sheets.contact(asset, into / "zones.png")]
        else:
            kind = "islands" if islands or material else None
            written = sheets.sheets(asset, into, kind, material)
    for path in written:
        click.echo(path)


if __name__ == "__main__":
    main()
