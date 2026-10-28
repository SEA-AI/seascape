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
import tempfile
import time
import tomllib
import uuid
from collections.abc import AsyncGenerator, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

import uvicorn
from mcp.server.mcpserver import Context, Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from PIL import Image as Picture
from pydantic import Field, PositiveFloat, PositiveInt, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict
from starlette.applications import Starlette
from starlette.staticfiles import StaticFiles

from seascape import assets
from seascape.assets import skies
from seascape.config import (
    Band,
    ImageFormat,
    Scenario,
    json_schema,
    load,
    load_variants,
)
from seascape.outputs import montage

KEEP_S = 24 * 3600
# A client may end a tool call after a minute.
WAIT_S = 45.0
# How often a waiting `render_result` reports progress, which keeps a client's idle
# timeout from ending it.
PROGRESS_S = 5.0
# PREVIEWS at this side, about 1.6k tokens each, fit a 25k-token tool result.
PREVIEW_PX = 1568
PREVIEWS = 8
# A judgement: longer than a driver takes to answer, short of a hung one.
PROBE_S = 60.0
# The hosts MCP answers when SEASCAPE_ALLOWED_HOSTS is unset: this machine only.
LOCAL_HOSTS = ("localhost:*", "127.0.0.1:*", "[::1]:*")


class Settings(BaseSettings):
    """The scenarios served, and what one job may ask of a shared machine."""

    # Compose passes a variable set empty in .env as "". nan would pass every
    # comparison and turn a limit off.
    model_config = SettingsConfigDict(
        env_prefix="SEASCAPE_", env_ignore_empty=True, allow_inf_nan=False, frozen=True
    )

    scenarios: Path
    allowed_hosts: Annotated[tuple[str, ...], NoDecode] = LOCAL_HOSTS
    max_duration_s: PositiveFloat = 60.0
    max_pixels: PositiveInt = 3840 * 2160
    max_images: PositiveInt = 10_000
    min_free_gb: PositiveFloat = 10.0
    max_run_s: PositiveFloat = 12 * 3600.0
    max_queued: PositiveInt = 20

    @field_validator("allowed_hosts", mode="before")
    @classmethod
    def _listed(cls, value: object) -> object:
        if isinstance(value, str):
            return tuple(h for h in value.split(",") if h) or LOCAL_HOSTS
        return value


# The most a pixel takes on disk, by band and format; a render test pins each. An exr
# and an EO png hold at most Blender's RGBA, 4 channels of float32 or 8 bits, and an
# LWIR png 16-bit grey. A jpg's bound leaves room for scenes noisier than the shipped
# ones.
BYTES_PER_PX: dict[tuple[Band, ImageFormat], float] = {
    ("eo", "exr"): 16.0,
    ("ir", "exr"): 16.0,
    ("eo", "png"): 4.0,
    ("ir", "png"): 2.0,
    ("eo", "jpg"): 1.0,
    ("ir", "jpg"): 0.5,
}


@dataclass
class Job:
    folder: Path
    scenario: str
    scenes: list[Scenario]
    created_s: float = field(default_factory=time.time)
    started_s: float | None = None
    state: str = "queued"
    error: str | None = None
    task: asyncio.Task[None] | None = None

    @property
    def bytes(self) -> float:
        """At most what the job writes: its images, and the archive of them."""
        return 2 * sum(
            len(scene.outputs.times_s)
            * mount.camera.width_px
            * mount.camera.height_px
            * BYTES_PER_PX[mount.camera.band, scene.outputs.format]
            for scene in self.scenes
            for mount in scene.mounts
            if mount.camera.band in scene.outputs.bands
        )

    @property
    def folders(self) -> list[Path]:
        """One per variant: `render` names each by its seed, skipping a seed whose
        hulls meet."""
        if len(self.scenes) == 1:
            return [self.folder]
        seeds = sorted(int(p.name) for p in self.folder.iterdir() if p.name.isdigit())
        return [self.folder / str(seed) for seed in seeds]

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


@dataclass
class State:
    """What the tools share; jobs take turns on `gpu`."""

    settings: Settings
    renders: Path = assets.CACHE / "renders"
    jobs: dict[str, Job] = field(default_factory=dict)
    gpu: asyncio.Lock = field(default_factory=asyncio.Lock)

    def scenarios(self) -> dict[str, Path]:
        return {p.stem: p for p in sorted(self.settings.scenarios.glob("*.toml"))}


def _state(ctx: Context[State, Any]) -> State:
    return ctx.request_context.lifespan_context


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


def _url(ctx: Context[Any, Any]) -> str:
    """Where this client reaches the jobs' files."""
    return f"http://{(ctx.headers or {}).get('host', 'localhost')}/renders"


def _refuse_oversized(scenes: list[Scenario], settings: Settings) -> None:
    images = sum(scene.images for scene in scenes)
    if images > settings.max_images:
        raise ToolError(f"{images} images; a job writes at most {settings.max_images}")
    for scene in scenes:
        if scene.outputs.duration_s > settings.max_duration_s:
            raise ToolError(f"a clip lasts at most {settings.max_duration_s} s")
        for mount in scene.mounts:
            camera = mount.camera
            if camera.width_px * camera.height_px > settings.max_pixels:
                raise ToolError(
                    f"{mount.name} is {camera.width_px} x {camera.height_px}; a "
                    f"camera has at most {settings.max_pixels} pixels"
                )


def _prune(state: State) -> None:
    cutoff = time.time() - KEEP_S
    for path in state.renders.glob("*"):
        if path.stat().st_mtime < cutoff:
            state.jobs.pop(path.stem, None)
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    # A cancelled job leaves no folder to age; a queued one has none yet.
    for name, job in list(state.jobs.items()):
        done = job.task is not None and job.task.done()
        if done and job.created_s < cutoff and not job.folder.exists():
            del state.jobs[name]


def catalog(ctx: Context[State, Any]) -> dict[str, Any]:
    """The scenarios `render` starts from, the meshes and skies a scenario can name,
    and the scenario's JSON schema. Any field can be a draw instead of a value:
    `{ uniform = [lo, hi] }` or `{ choice = [...] }`, drawn per seed."""
    return {
        "scenarios": {n: p.read_text() for n, p in _state(ctx).scenarios().items()},
        "meshes": {
            name: f"{mesh.kind}, {mesh.category}, {mesh.size}"
            for name, mesh in assets.manifest().items()
        },
        "skies": {name: photo.sun for name, photo in skies.library().items()},
        "schema": json_schema(),
    }


async def render(
    ctx: Context[State, Any],
    scenario: str = "baseline",
    overrides: list[str] | None = None,
    variants: Annotated[int, Field(ge=1)] = 1,
) -> dict[str, Any]:
    """Queue a render of a scenario from `catalog` and return its job at once.

    `overrides` are TOML lines merged over the scenario: `sky.sun_elevation_deg = 5`,
    `sea.wind_speed_mps = { uniform = [2, 12] }`. `variants` renders that many
    seeds from the scenario's own, each drawn anew, skipping a seed whose hulls meet.
    """
    state = _state(ctx)
    settings = state.settings
    if (path := state.scenarios().get(scenario)) is None:
        raise ToolError(f"no scenario {scenario!r} in {sorted(state.scenarios())}")
    # Each variant writes an image at least, so this refuses before loading.
    if variants > settings.max_images:
        raise ToolError(f"a job writes at most {settings.max_images} images")
    lines = overrides or []
    try:
        for line in lines:
            # A path would read any TOML here, and an error echoes what it read.
            if any(
                "/" in n or n.endswith(".toml") for n in _includes(tomllib.loads(line))
            ):
                raise ToolError(f"{line!r}: an override names presets, never files")
        first = await asyncio.to_thread(load, path, lines)
        # A scene the full check would refuse, before loading the other variants.
        _refuse_oversized([first], settings)
        scenes = await asyncio.to_thread(load_variants, path, lines, variants)
    # pydantic and tomllib raise ValueError, a bad preset TypeError or OSError.
    except (ValueError, TypeError, OSError) as error:
        raise ToolError(str(error)) from error
    _refuse_oversized(scenes, settings)
    state.renders.mkdir(parents=True, exist_ok=True)
    _prune(state)
    if (ahead := sum(j.waiting for j in state.jobs.values())) >= settings.max_queued:
        raise ToolError(
            f"{ahead} jobs queued; the server takes at most {settings.max_queued}"
        )
    name = uuid.uuid4().hex
    job = Job(state.renders / name, scenario, scenes)
    # A running job's images count twice, as written and as still to come; a
    # finishing one still writes its zip.
    writing = [j for j in state.jobs.values() if j.waiting or j.state == "finishing"]
    free = shutil.disk_usage(state.renders).free - sum(j.bytes for j in writing)
    free_gb = free / 1e9
    if (needs_gb := job.bytes / 1e9 + settings.min_free_gb) > free_gb:
        raise ToolError(
            f"{free_gb:.0f} GB free after the jobs queued; this job needs "
            f"{needs_gb:.0f}, {settings.min_free_gb:.0f} of it kept free"
        )
    state.jobs[name] = job
    job.task = asyncio.create_task(_run(state, job, path, lines, variants))
    return {"job": name, "images": job.total, "jobs_ahead": ahead}


async def _seascape(*args: str) -> None:
    # A killed Blender leaves its temporary files, the foam memory's frames among them;
    # here they go with the job.
    with tempfile.TemporaryDirectory(prefix="seascape-job-") as tmp:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "seascape.cli",
            *args,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, "TMPDIR": tmp},
        )
        try:
            _, stderr = await process.communicate()
        except asyncio.CancelledError:
            # asyncio leaves the process running, and Blender on the GPU.
            process.kill()
            await process.wait()
            raise
    if process.returncode:
        raise RuntimeError(stderr.decode(errors="replace").strip()[-2000:])


async def _run(
    state: State, job: Job, path: Path, overrides: list[str], variants: int
) -> None:
    sets = [arg for line in overrides for arg in ("--set", line)]
    max_run_s = state.settings.max_run_s
    try:
        async with state.gpu:
            job.state, job.started_s = "running", time.time()
            # Past it, the process awaited is killed as on `cancel`.
            async with asyncio.timeout(max_run_s):
                await _seascape(
                    "render", str(path), "-o", str(job.folder), "--variants",
                    str(variants), *sets,
                )  # fmt: skip
                if variants > 1:
                    job.scenes = await asyncio.to_thread(_written, job, path, overrides)
                for scene, folder in zip(job.scenes, job.folders, strict=True):
                    if len(scene.outputs.times_s) > 1:
                        await _seascape("video", str(folder))
        # Off the GPU: the next job starts while this one zips.
        job.state = "finishing"
        await asyncio.to_thread(_finish, job)
        job.state = "done"
    except TimeoutError:
        job.state, job.error = "failed", f"ran past {max_run_s:g} s"
    except Exception as error:  # whatever it was, the client polling is told
        job.state, job.error = "failed", f"{type(error).__name__}: {error}"


def _written(job: Job, path: Path, overrides: list[str]) -> list[Scenario]:
    """The scenario of each variant `render` wrote, by the seed its folder names."""
    return [load(path, [*overrides, f"seed = {f.name}"]) for f in job.folders]


def _finish(job: Job) -> None:
    """A preview of each of the first `PREVIEWS` stills, and an archive of all."""
    for scene, folder in zip(job.scenes[:PREVIEWS], job.folders, strict=False):
        if len(scene.outputs.times_s) == 1 and scene.outputs.format != "exr":
            with Picture.open(montage.compose(scene, folder)) as picture:
                picture.thumbnail((PREVIEW_PX, PREVIEW_PX))
                picture.convert("RGB").save(folder / "preview.jpg", quality=85)
    shutil.make_archive(str(job.folder), "zip", job.folder)


async def render_result(job: str, ctx: Context[State, Any]) -> list[Any]:
    """Wait briefly for a job from `render`, then report it: its progress, or its
    error, or every file as a URL with a preview of each of the first `PREVIEWS`
    stills."""
    state = _state(ctx)
    found = state.jobs.get(job)
    if found is None or found.task is None:
        raise ToolError(
            f"no job {job!r}: a job is forgotten when it expires or the server restarts"
        )
    deadline = time.monotonic() + WAIT_S
    while not found.task.done() and (left := deadline - time.monotonic()) > 0:
        await asyncio.wait({found.task}, timeout=min(PROGRESS_S, left))
        await ctx.report_progress(found.images, found.total)
    report: dict[str, Any] = {"status": found.state}
    previews: list[Image] = []
    if found.state == "failed":
        report["error"] = found.error
    elif found.state == "done":
        url = _url(ctx)
        # A sequence's frames are in the archive only.
        report["files"] = [f"{url}/{job}.zip"] + [
            f"{url}/{path.relative_to(state.renders)}"
            for path in sorted(found.folder.rglob("*"))
            if path.is_file() and not path.stem.isdigit()
        ]
        previews = [
            Image(path=preview)
            for folder in found.folders
            if (preview := folder / "preview.jpg").exists()
        ]
    elif found.state == "queued":
        jobs = list(state.jobs.values())
        report["jobs_ahead"] = sum(j.waiting for j in jobs[: jobs.index(found)])
    elif found.state == "running":
        done, total = found.images, found.total
        report["images"] = f"{done} of {total}"
        if done and found.started_s is not None:
            pace_s = (time.time() - found.started_s) / done
            report["remaining_s"] = round(pace_s * (total - done))
    return [json.dumps(report, indent=1), *previews]


async def cancel(job: str, ctx: Context[State, Any]) -> dict[str, Any]:
    """Stop a queued or running job from `render`, and delete what it wrote."""
    found = _state(ctx).jobs.get(job)
    if found is None or found.task is None:
        raise ToolError(f"no job {job!r}")
    if not found.waiting:
        raise ToolError(f"job {job!r} is {found.state}")
    found.task.cancel()
    found.state = "cancelled"
    # Once the task ends, the process is gone and the GPU free.
    await asyncio.wait({found.task})
    await asyncio.to_thread(shutil.rmtree, found.folder, ignore_errors=True)
    return {"job": job, "status": found.state}


def jobs(ctx: Context[State, Any]) -> dict[str, Any]:
    """Every job the server remembers, newest first, and the disk left for more."""
    url = _url(ctx)
    state = _state(ctx)
    return {
        "free_gb": round(shutil.disk_usage(state.renders).free / 1e9, 1),
        "jobs": [
            {
                "job": name,
                "scenario": job.scenario,
                "status": job.state,
                "images": job.total,
                "age_s": round(time.time() - job.created_s),
                # The archive is written when the job finishes.
                **(
                    {
                        "archive_mb": round(archive.stat().st_size / 2**20),
                        "archive": f"{url}/{name}.zip",
                    }
                    if (archive := Path(f"{job.folder}.zip")).exists()
                    else {}
                ),
            }
            for name, job in reversed(state.jobs.items())
        ],
    }


def mcp(state: State) -> MCPServer:
    state.renders.mkdir(parents=True, exist_ok=True)

    # The in-memory client enters the lifespan per connection; jobs outlive one.
    @asynccontextmanager
    async def lifespan(_: MCPServer) -> AsyncGenerator[State]:
        yield state

    served = MCPServer(
        "seascape",
        instructions="Synthetic maritime camera frames with exact ground truth. Read "
        "`catalog` once, start a job with `render`, then call `render_result` until "
        "it is done. Without draws, variants differ only in their waves: for varied "
        "images, render `randomized`, or draw fields in `overrides`.",
        lifespan=lifespan,
    )
    for tool in (catalog, render, cancel, jobs):
        served.add_tool(tool)
    # Unstructured: the previews are images, which structured content cannot carry.
    served.add_tool(render_result, structured_output=False)
    return served


def serve(state: State, host: str, port: int) -> None:
    """Serve `app` on `port`."""
    # In a process of its own, so the server never holds Blender.
    try:
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                "from seascape.blender import compositor; "
                "print(compositor.enable_gpu())",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=PROBE_S,
        )
    except subprocess.TimeoutExpired:
        print(f"The GPU probe gave no answer in {PROBE_S:g} s", flush=True)
    else:
        words = probe.stdout.split()
        backend = words[-1] if words else "None"
        print(
            f"Cycles renders on {'the CPU' if backend == 'None' else backend}",
            flush=True,
        )
    uvicorn.run(app(state, host), host=host, port=port)


def app(state: State, host: str) -> Starlette:
    """MCP at /mcp and the jobs' files at /renders/."""
    # A web page can point a browser on the network at this port by DNS rebinding,
    # so MCP refuses any Host or Origin not listed.
    hosts = list(state.settings.allowed_hosts)
    security = TransportSecuritySettings(
        allowed_hosts=hosts,
        allowed_origins=[f"{s}://{h}" for h in hosts for s in ("http", "https")],
    )
    served = mcp(state).streamable_http_app(host=host, transport_security=security)
    served.mount("/renders", StaticFiles(directory=state.renders))
    return served
