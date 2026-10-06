"""Render scenarios for MCP clients, one job at a time on this machine's GPU.

A job runs the CLI in a process of its own, so Blender's crashes and leaks end with
it. Its files are served under /renders/ and deleted after `KEEP_S`.
"""

import asyncio
import json
import shutil
import subprocess
import sys
import time
import tomllib
import uuid
from collections.abc import Iterator
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import uvicorn
from mcp.server.mcpserver import Context, Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from PIL import Image as Picture
from pydantic import Field
from starlette.staticfiles import StaticFiles

from seascape import assets, montage, skies
from seascape.config import Scenario, folders, json_schema, load_variants

SCENARIOS = Path(__file__).parents[1] / "scenarios"
RENDERS = assets.CACHE / "renders"
KEEP_S = 24 * 3600
# A client may end a tool call after a minute.
WAIT_S = 45.0
# The longest side a model reads without downscaling.
PREVIEW_PX = 1568
PREVIEWS = 8

server = MCPServer(
    "seascape",
    instructions="Synthetic maritime camera frames with exact ground truth. Read "
    "`catalog` once, start a job with `render`, then call `render_result` until it "
    "is done. Without draws, variants differ only in their waves: for varied "
    "images, render `randomized`, or draw fields in `overrides`.",
)
_gpu = asyncio.Lock()


@dataclass
class Job:
    folder: Path
    scenes: list[Scenario]
    state: str = "queued"
    error: str | None = None
    task: asyncio.Task[None] | None = None

    @property
    def folders(self) -> list[Path]:
        return folders(self.folder, self.scenes)

    @property
    def images(self) -> int:
        suffix = f".{self.scenes[0].outputs.format}"
        return sum(path.suffix == suffix for path in self.folder.rglob("*"))


_jobs: dict[str, Job] = {}


def _scenarios() -> dict[str, Path]:
    return {path.stem: path for path in sorted(SCENARIOS.glob("*.toml"))}


def _includes(node: Any) -> Iterator[str]:
    """Every `extends` and `preset` in a TOML tree."""
    if isinstance(node, list):
        for item in node:
            yield from _includes(item)
    elif isinstance(node, dict):
        for key, value in node.items():
            if key in ("extends", "preset") and isinstance(value, str):
                yield value
            yield from _includes(value)


def _prune() -> None:
    cutoff = time.time() - KEEP_S
    for path in RENDERS.glob("*"):
        if path.stat().st_mtime < cutoff:
            _jobs.pop(path.stem, None)
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()


@server.tool()
def catalog() -> dict[str, Any]:
    """The scenarios `render` starts from, the meshes and skies a scenario can name,
    and the scenario's JSON schema. Any field can be a draw instead of a value:
    `{ uniform = [lo, hi] }` or `{ choice = [...] }`, drawn per seed."""
    return {
        "scenarios": {name: path.read_text() for name, path in _scenarios().items()},
        "meshes": {
            name: f"{mesh.kind}, {mesh.category}, {mesh.size}"
            for name, mesh in assets.manifest().items()
        },
        "skies": {name: photo.sun for name, photo in skies.library().items()},
        "schema": json_schema(),
    }


@server.tool()
async def render(
    scenario: str = "baseline",
    overrides: list[str] | None = None,
    variants: Annotated[int, Field(ge=1)] = 1,
) -> dict[str, Any]:
    """Queue a render of a scenario from `catalog` and return its job at once.

    `overrides` are TOML lines merged over the scenario: `sky.sun_elevation_deg = 5`,
    `sea.wind_speed_mps = { uniform = [2, 12] }`. `variants` renders that many
    consecutive seeds, each drawn anew.
    """
    if scenario not in _scenarios():
        raise ToolError(f"no scenario {scenario!r} in {sorted(_scenarios())}")
    lines = overrides or []
    path = _scenarios()[scenario]
    try:
        for line in lines:
            # A path would read any TOML here, and an error echoes what it read.
            if any(
                "/" in n or n.endswith(".toml") for n in _includes(tomllib.loads(line))
            ):
                raise ToolError(f"{line!r}: an override names presets, never files")
        scenes = await asyncio.to_thread(load_variants, path, lines, variants)
    except ValueError as error:  # pydantic and tomllib both raise it
        raise ToolError(str(error)) from error
    RENDERS.mkdir(parents=True, exist_ok=True)
    _prune()
    name = uuid.uuid4().hex
    job = _jobs[name] = Job(RENDERS / name, scenes)
    job.task = asyncio.create_task(_run(job, path, lines, variants))
    return {"job": name, "images": sum(scene.images for scene in scenes)}


async def _seascape(*args: str) -> None:
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "seascape.cli",
        *args,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()
    if process.returncode:
        raise RuntimeError(stderr.decode(errors="replace").strip()[-2000:])


async def _run(job: Job, path: Path, overrides: list[str], variants: int) -> None:
    sets = [arg for line in overrides for arg in ("--set", line)]
    try:
        async with _gpu:
            job.state = "running"
            await _seascape(
                "render", str(path), "-o", str(job.folder), "--variants",
                str(variants), *sets,
            )  # fmt: skip
            for scene, folder in zip(job.scenes, job.folders, strict=True):
                if len(scene.outputs.times_s) > 1:
                    await _seascape("video", str(folder))
        await asyncio.to_thread(_finish, job)
        job.state = "done"
    except Exception as error:  # whatever it was, the client polling is told
        job.state, job.error = "failed", f"{type(error).__name__}: {error}"


def _finish(job: Job) -> None:
    """A preview of each still, and an archive of everything."""
    for scene, folder in zip(job.scenes[:PREVIEWS], job.folders, strict=False):
        if len(scene.outputs.times_s) == 1 and scene.outputs.format != "exr":
            with Picture.open(montage.compose(scene, folder)) as picture:
                picture.thumbnail((PREVIEW_PX, PREVIEW_PX))
                picture.convert("RGB").save(folder / "preview.jpg", quality=85)
    shutil.make_archive(str(job.folder), "zip", job.folder)


# Unstructured: the previews are images, which structured content cannot carry.
@server.tool(structured_output=False)
async def render_result(job: str, ctx: Context) -> list[Any]:
    """Wait briefly for a job from `render`, then report it: its progress, or its
    error, or every file as a URL with a preview of each still."""
    found = _jobs.get(job)
    if found is None or found.task is None:
        raise ToolError(
            f"no job {job!r}: a job is forgotten when it expires or the server restarts"
        )
    with suppress(TimeoutError):
        await asyncio.wait_for(asyncio.shield(found.task), WAIT_S)
    total = sum(scene.images for scene in found.scenes)
    report: dict[str, Any] = {"status": found.state}
    previews: list[Image] = []
    if found.state == "failed":
        report["error"] = found.error
    elif found.state == "done":
        host = (ctx.headers or {}).get("host", "localhost")
        url = f"http://{host}/renders"
        # A sequence's frames are in the archive only.
        report["files"] = [f"{url}/{job}.zip"] + [
            f"{url}/{path.relative_to(RENDERS)}"
            for path in sorted(found.folder.rglob("*"))
            if path.is_file() and not path.stem.isdigit()
        ]
        previews = [
            Image(path=preview)
            for folder in found.folders
            if (preview := folder / "preview.jpg").exists()
        ]
    else:
        report["images"] = f"{min(found.images, total)} of {total}"
    return [json.dumps(report, indent=1), *previews]


def serve(host: str, port: int) -> None:
    """Serve MCP at /mcp and the jobs' files at /renders/."""
    # In a process of its own, so the server never holds Blender.
    probe = subprocess.run(
        [sys.executable, "-c", "from seascape import scene; print(scene.enable_gpu())"],
        capture_output=True,
        text=True,
        check=True,
    )
    backend = probe.stdout.split()[-1]
    print(
        f"Cycles renders on {'the CPU' if backend == 'None' else backend}", flush=True
    )
    RENDERS.mkdir(parents=True, exist_ok=True)
    app = server.streamable_http_app(host=host)
    app.mount("/renders", StaticFiles(directory=RENDERS))
    uvicorn.run(app, host=host, port=port)
