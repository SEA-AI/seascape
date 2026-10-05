"""Render scenarios for MCP clients, one job at a time on this machine's GPU.

A job runs the CLI in a process of its own, so Blender's crashes and leaks end with
it. Its files are served under /renders/ and deleted after `KEEP_S`.
"""

import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
import tomllib
import uuid
from collections.abc import Iterator
from contextlib import suppress
from dataclasses import dataclass, field
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


def _limit(name: str, default: float) -> float:
    """`SEASCAPE_<name>` from the environment, else `default`."""
    return float(os.environ.get(f"SEASCAPE_{name}", default))


# What one job may ask of a shared machine.
MAX_DURATION_S = _limit("MAX_DURATION_S", 60.0)
MAX_PIXELS = int(_limit("MAX_PIXELS", 3840 * 2160))
MAX_IMAGES = int(_limit("MAX_IMAGES", 10_000))
MIN_FREE_GB = _limit("MIN_FREE_GB", 10.0)

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
    scenario: str
    scenes: list[Scenario]
    created: float = field(default_factory=time.time)
    started: float | None = None
    state: str = "queued"
    error: str | None = None
    task: asyncio.Task[None] | None = None

    @property
    def folders(self) -> list[Path]:
        return folders(self.folder, self.scenes)

    @property
    def total(self) -> int:
        return sum(scene.images for scene in self.scenes)

    @property
    def images(self) -> int:
        """Written so far; a jpg job's preview.jpg shares the suffix."""
        suffix = f".{self.scenes[0].outputs.format}"
        written = sum(path.suffix == suffix for path in self.folder.rglob("*"))
        return min(written, self.total)

    @property
    def waiting(self) -> bool:
        return self.state in ("queued", "running")


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


def _url(ctx: Context) -> str:
    """Where this client reaches the jobs' files."""
    return f"http://{(ctx.headers or {}).get('host', 'localhost')}/renders"


def _refuse_oversized(scenes: list[Scenario]) -> None:
    images = sum(scene.images for scene in scenes)
    if images > MAX_IMAGES:
        raise ToolError(f"{images} images; a job writes at most {MAX_IMAGES}")
    for scene in scenes:
        if scene.outputs.duration_s > MAX_DURATION_S:
            raise ToolError(f"a clip lasts at most {MAX_DURATION_S} s")
        for mount in scene.rig.mounts:
            camera = mount.camera
            if camera.width_px * camera.height_px > MAX_PIXELS:
                raise ToolError(
                    f"{mount.name} is {camera.width_px} x {camera.height_px}; a "
                    f"camera has at most {MAX_PIXELS} pixels"
                )


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
    # Each variant writes an image at least, so the bound refuses before loading.
    variants: Annotated[int, Field(ge=1, le=MAX_IMAGES)] = 1,
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
    _refuse_oversized(scenes)
    RENDERS.mkdir(parents=True, exist_ok=True)
    _prune()
    if (free_gb := shutil.disk_usage(RENDERS).free / 2**30) < MIN_FREE_GB:
        raise ToolError(f"{free_gb:.0f} GB free; a job needs {MIN_FREE_GB:.0f}")
    ahead = sum(job.waiting for job in _jobs.values())
    name = uuid.uuid4().hex
    job = _jobs[name] = Job(RENDERS / name, scenario, scenes)
    job.task = asyncio.create_task(_run(job, path, lines, variants))
    return {"job": name, "images": job.total, "jobs_ahead": ahead}


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
            job.state, job.started = "running", time.time()
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
    report: dict[str, Any] = {"status": found.state}
    previews: list[Image] = []
    if found.state == "failed":
        report["error"] = found.error
    elif found.state == "done":
        url = _url(ctx)
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
    elif found.state == "queued":
        jobs = list(_jobs.values())
        report["jobs_ahead"] = sum(j.waiting for j in jobs[: jobs.index(found)])
    else:
        done, total = found.images, found.total
        report["images"] = f"{done} of {total}"
        if done and found.started is not None:
            pace_s = (time.time() - found.started) / done
            report["remaining_s"] = round(pace_s * (total - done))
    return [json.dumps(report, indent=1), *previews]


@server.tool()
def jobs(ctx: Context) -> dict[str, Any]:
    """Every job the server remembers, newest first, and the disk left for more."""
    url = _url(ctx)
    return {
        "free_gb": round(shutil.disk_usage(RENDERS).free / 2**30, 1),
        "jobs": [
            {
                "job": name,
                "scenario": job.scenario,
                "status": job.state,
                "images": job.total,
                "age_s": round(time.time() - job.created),
                # A running job deletes files as it goes, so only a finished one.
                **(
                    {
                        "archive_mb": round(
                            Path(f"{job.folder}.zip").stat().st_size / 2**20
                        ),
                        "archive": f"{url}/{name}.zip",
                    }
                    if job.state == "done"
                    else {}
                ),
            }
            for name, job in reversed(_jobs.items())
        ],
    }


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
