"""Command line entry point."""

import argparse
import json
import sys
from pathlib import Path
from typing import get_args

from seascape import montage, panorama
from seascape.config import Band, Scenario, load


def _build(
    scenario_path: Path, output: Path | None, band: Band, overrides: list[str]
) -> None:
    scenario = load(scenario_path, overrides)
    # Deferred: `schema` and a failed validation should not wait for bpy to load.
    import bpy

    from seascape import scene

    scene.build(scenario, band)
    path = output or scenario_path.with_suffix(f".{band}.blend")
    bpy.ops.wm.save_as_mainfile(filepath=str(path.resolve()))
    mounts = scenario.rig.mounts
    kinds = ", ".join(sorted({mount.camera.kind for mount in mounts}))
    print(
        f"{path}: {band}, {len(mounts)} cameras ({kinds}) at {scenario.rig.height_m} m"
    )


def _render(scenario_path: Path, output: Path | None, overrides: list[str]) -> None:
    scenario = load(scenario_path, overrides)
    from seascape import render

    into = output or scenario_path.with_suffix("")
    written = render.render(scenario, into)
    for path in written:
        print(path)
    print(f"{len(written)} files in {into}")


def _montage(scenario_path: Path, output: Path | None, overrides: list[str]) -> None:
    scenario = load(scenario_path, overrides)
    into = output or scenario_path.with_suffix("")
    print(montage.compose(scenario, into))


def _add_set(command: argparse.ArgumentParser) -> None:
    command.add_argument(
        "--set",
        action="append",
        default=[],
        dest="overrides",
        metavar="KEY=VALUE",
        help="override a field, written as TOML: 'rig.pitch_deg = -5'. Repeatable.",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="seascape")
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="write a .blend from a scenario")
    build.add_argument("scenario", type=Path)
    build.add_argument("-o", "--output", type=Path, help="default: alongside the input")
    # A scene is one band or the other: EO and LWIR share no units.
    build.add_argument("--band", choices=get_args(Band.__value__), default="eo")
    _add_set(build)

    shoot = commands.add_parser("render", help="write one image per camera")
    shoot.add_argument("scenario", type=Path)
    shoot.add_argument(
        "-o", "--output", type=Path, help="directory, default: alongside the input"
    )
    _add_set(shoot)

    lay = commands.add_parser("montage", help="lay rendered frames out for review")
    lay.add_argument("scenario", type=Path)
    lay.add_argument(
        "-o", "--output", type=Path, help="directory the frames are in, and the montage"
    )
    _add_set(lay)

    stitch = commands.add_parser(
        "panorama", help="stitch each pod's frames, from calibration.json"
    )
    stitch.add_argument("folder", type=Path, help="a render's output directory")
    stitch.add_argument(
        "--projection",
        choices=list(panorama.PROJECTIONS),
        default="cylindrical",
    )
    stitch.add_argument(
        "--frame",
        default="world",
        help="an extrinsics frame in calibration.json, which sets what is level: "
        "world the horizon, vessel the deck, pod the enclosure",
    )
    stitch.add_argument(
        "--max-width", type=int, help="pixels; native resolution when absent"
    )
    stitch.add_argument(
        "--ruler", action="store_true", help="a strip of bearing ticks under the image"
    )

    film = commands.add_parser(
        "video", help="encode each camera's frames as an mp4, from labels.json"
    )
    film.add_argument("folder", type=Path, help="a render's output directory")
    film.add_argument(
        "--quality",
        default="perc_lossless",
        choices=[
            "lossless",
            "perc_lossless",
            "high",
            "medium",
            "low",
            "verylow",
            "lowest",
        ],
        help="Blender's H.264 constant-quality preset",
    )

    commands.add_parser("schema", help="print the scenario JSON schema on stdout")

    args = parser.parse_args(argv)
    if args.command == "schema":
        print(json.dumps(Scenario.model_json_schema(), indent=2))
        return 0
    try:
        if args.command == "render":
            _render(args.scenario, args.output, args.overrides)
        elif args.command == "montage":
            _montage(args.scenario, args.output, args.overrides)
        elif args.command == "panorama":
            for path in panorama.panoramas(
                args.folder, args.projection, args.frame, args.max_width, args.ruler
            ):
                print(path)
        elif args.command == "video":
            from seascape import video

            for path in video.encode(args.folder, args.quality.upper()):
                print(path)
        else:
            _build(args.scenario, args.output, args.band, args.overrides)
    except (
        OSError,
        ValueError,
        TypeError,
        RuntimeError,  # bpy.ops.render.render, e.g. an unwritable output directory
    ):  # pydantic and tomllib both raise ValueError
        # A scenario mistake is the user's, not a crash; a traceback buries the line.
        print(sys.exception(), file=sys.stderr)
        return 1
    return 0
