"""Command line entry point."""

import json
import math
import os
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from typing import Any, get_args

import click

from seascape import assets, montage, panorama, recording, skies
from seascape.config import Band, json_schema, load

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
            ValueError,  # pydantic and tomllib both raise it
            TypeError,
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
            help="Override a field, written as TOML: 'rig.pitch_deg = -5'. Repeatable.",
        )(command)
        command = click.option(
            "-o", "--output", type=click.Path(path_type=Path), help=output
        )(command)
        return click.argument("scenario", type=_FILE)(command)

    return add


@contextmanager
def _blender_log(path: Path) -> Iterator[None]:
    """Blender logs to `sys.stdout` from Python and to file descriptor 1 from C, where
    either would tear the progress bar and mix into stdout; both go to `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout.flush()
    stdout = os.dup(1)
    with path.open("a") as log, redirect_stdout(log):
        os.dup2(log.fileno(), 1)
        try:
            yield
        finally:
            log.flush()
            os.dup2(stdout, 1)
            os.close(stdout)


def _progress(length: int, label: str) -> Any:
    return click.progressbar(length=length, label=label, file=sys.stderr)


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

    from seascape import scene

    path = output or scenario.with_suffix(f".{band}.blend")
    with _blender_log(path.with_suffix(".log")):
        scene.build(built, band)
        bpy.ops.wm.save_as_mainfile(filepath=str(path.resolve()))
    mounts = built.rig.mounts
    kinds = ", ".join(sorted({mount.camera.kind for mount in mounts}))
    click.echo(
        f"{path}: {band}, {len(mounts)} cameras ({kinds}) at {built.rig.height_m} m"
    )


@main.command()
@_scenario("A directory. Default: beside the scenario.")
@click.option(
    "--variants",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="Scenes from consecutive seeds, each in a folder named for its seed.",
)
def render(
    scenario: Path, output: Path | None, overrides: tuple[str, ...], variants: int
) -> None:
    """Write one image per camera and frame, and their labels."""
    built = load(scenario, list(overrides))
    into = output or scenario.with_suffix("")
    scenes = [(into, built)]
    if variants > 1:
        seeds = range(built.seed, built.seed + variants)
        scenes = [
            (into / str(seed), load(scenario, [*overrides, f"seed = {seed}"]))
            for seed in seeds
        ]
    from seascape import render as renderer

    with (
        _progress(sum(s.images for _, s in scenes), "Rendering") as bar,
        _blender_log(into / "blender.log"),
    ):
        written = [
            path
            for folder, scene in scenes
            for path in renderer.render(scene, folder, lambda: bar.update(1))
        ]
    for path in written:
        click.echo(path)
    click.echo(f"{len(written)} files in {into}", err=True)


@main.command("montage")
@_scenario("The directory the frames are in. Default: beside the scenario.")
def montage_(scenario: Path, output: Path | None, overrides: tuple[str, ...]) -> None:
    """Lay rendered frames out for review."""
    built = load(scenario, list(overrides))
    click.echo(montage.compose(built, output or scenario.with_suffix("")))


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
    default="world",
    show_default=True,
    help="An extrinsics frame in calibration.json, which sets what is level: world "
    "the horizon, vessel the deck, pod the enclosure.",
)
@click.option("--max-width", type=int, help="Pixels; native resolution when absent.")
@click.option("--ruler", is_flag=True, help="A strip of bearing ticks under the image.")
def panorama_(
    folder: Path, projection: str, frame: str, max_width: int | None, ruler: bool
) -> None:
    """Stitch each pod's frames, from calibration.json."""
    for path in panorama.panoramas(folder, projection, frame, max_width, ruler):
        click.echo(path)


@main.command()
@click.argument("folder", type=_FOLDER)
@click.option(
    "--quality",
    type=click.Choice(
        ["lossless", "perc_lossless", "high", "medium", "low", "verylow", "lowest"]
    ),
    default="perc_lossless",
    show_default=True,
    help="Blender's H.264 constant-quality preset.",
)
def video(folder: Path, quality: str) -> None:
    """Encode each camera's frames as an mp4, from labels.json."""
    from seascape import video as encoder

    with (
        _progress(len(encoder.cameras(folder)), "Encoding") as bar,
        _blender_log(folder / "blender.log"),
    ):
        written = encoder.encode(folder, quality.upper(), lambda: bar.update(1))
    for path in written:
        click.echo(path)


@main.command("recording")
@click.argument("folder", type=_FOLDER)
def recording_(folder: Path) -> None:
    """Lay each pod's videos out as a recording, after video."""
    for path in recording.export(folder):
        click.echo(path)


@main.command()
def schema() -> None:
    """Print the scenario JSON schema."""
    click.echo(json.dumps(json_schema(), indent=2))


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
        ("SKY", "SUN ELEV", "LICENCE", "CACHED"),
        [
            (
                name,
                photos[name].sun,
                photos[name].licence,
                _cached(assets.cache_path(name, photos[name].url).exists()),
            )
            for name in order
        ],
    )
    click.echo("\nMeshes go in a scenario's `asset`, skies in `sky.hdri`.")


def _names(ctx: click.Context, param: click.Parameter, incomplete: str) -> list[str]:
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
@click.argument("scenario", type=_FILE)
@click.argument("out", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--band", type=_BANDS, default="eo", show_default=True)
def sheet(scenario: Path, out: Path, band: Band) -> None:
    """Render every mesh in a scenario's sea and sky, one tile each."""
    from seascape import catalog

    with _blender_log(out.with_suffix(".log")):
        catalog.sheet(scenario, out, band)
    click.echo(out)
