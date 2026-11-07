"""Render scenarios for MCP clients, one job at a time on this machine's GPU.

A job runs the CLI in a process of its own, so Blender's crashes and leaks end with
it. Its files are served under /renders/ and deleted after `KEEP_S`.
"""

import asyncio
import heapq
import json
import logging
import os
import re
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
from typing import Annotated, Any, Literal

import uvicorn
from mcp.server.mcpserver import Context, Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations
from PIL import Image as Picture
from pydantic import (
    AnyUrl,
    Field,
    PositiveFloat,
    PositiveInt,
    SecretStr,
    UrlConstraints,
    field_validator,
    model_validator,
)
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
    seeds,
)
from seascape.outputs import montage

KEEP_S = 24 * 3600
# A client may end a tool call after a minute.
WAIT_S = 45.0
# Progress report interval; keeps a client's idle timeout from ending the wait.
PROGRESS_S = 5.0
# The long edge Claude reads an image at (Anthropic vision docs).
PREVIEW_PX = 1568
PREVIEWS = 8
# Judgement: a result a client can read; the archive holds the rest.
MAX_LISTED = 200
# Judgement: far more and longer than a scenario's overrides.
MAX_OVERRIDES = 50
MAX_OVERRIDE_CHARS = 1000
# Judgement: loads hold the GIL, so a few at once cost every other client.
LOADING = 2
# A judgement: longer than a driver takes to answer, short of a hung one.
PROBE_S = 60.0
LOCAL_HOSTS = ("localhost:*", "127.0.0.1:*", "[::1]:*")
# Under uvicorn's handlers, so the lines reach the server's log.
log = logging.getLogger("uvicorn.error")


class Settings(BaseSettings):
    """The scenarios served, who may reach them, and what a job may ask of this host."""

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
    # Judgement: more seeds than a client reviews.
    max_variants: PositiveInt = 100
    # Judgement: bounds a frame's render time.
    max_samples: PositiveInt = 1024
    # The path is the credential: an MCP client may send no header.
    mcp_secret: SecretStr | None = None
    # The secret travels in the path, in the clear over http.
    public_url: Annotated[AnyUrl, UrlConstraints(allowed_schemes=["https"])] | None = (
        None
    )

    @field_validator("allowed_hosts", mode="before")
    @classmethod
    def _listed(cls, value: object) -> object:
        if isinstance(value, str):
            return tuple(h for h in value.split(",") if h) or LOCAL_HOSTS
        return value

    @model_validator(mode="after")
    def _secret_guards_the_path(self) -> "Settings":
        if self.public_url and not self.mcp_secret:
            raise ValueError("SEASCAPE_PUBLIC_URL needs SEASCAPE_MCP_SECRET")
        # The only credential on a public URL, and a route path, where `/` and `{}`
        # mean something else. Judgement: enough entropy to resist guessing.
        if self.mcp_secret and not re.fullmatch(
            r"[A-Za-z0-9_-]{32,}", self.mcp_secret.get_secret_value()
        ):
            raise ValueError(
                "SEASCAPE_MCP_SECRET needs 32 or more of A-Z, a-z, 0-9, _ and -"
            )
        return self


# The most a pixel takes on disk. exr: RGBA float32; EO png: RGBA 8-bit; LWIR png:
# 16-bit grey. A jpg's size follows the content, so its bound is a judgement.
BYTES_PER_PX: dict[tuple[Band, ImageFormat], float] = {
    ("eo", "exr"): 16.0,
    ("ir", "exr"): 16.0,
    ("eo", "png"): 4.0,
    ("ir", "png"): 2.0,
    ("eo", "jpg"): 1.0,
    ("ir", "jpg"): 0.5,
}


type JobState = Literal["queued", "running", "finishing", "done", "failed", "cancelled"]


@dataclass
class Job:
    """One `render` call and how far it got."""

    folder: Path
    scenario: str
    scenes: list[Scenario]
    created_s: float = field(default_factory=time.time)
    started_s: float | None = None
    state: JobState = "queued"
    error: str | None = None
    task: asyncio.Task[None] | None = None
    # The largest scenes a replacement seed may draw.
    worst_bytes: float = 0.0

    @property
    def size_bytes(self) -> float:
        """At most what the job writes: its images, and the archive of them."""
        return max(self.worst_bytes, sum(_bytes(scene) for scene in self.scenes))

    @property
    def folders(self) -> list[Path]:
        """One per variant: `render` names each by its seed, skipping a seed whose
        hulls meet.
        """
        if len(self.scenes) == 1:
            return [self.folder]
        names = sorted(int(p.name) for p in self.folder.iterdir() if p.name.isdigit())
        return [self.folder / str(name) for name in names]

    @property
    def total(self) -> int:
        """How many images the job writes."""
        return sum(scene.images for scene in self.scenes)

    @property
    def images(self) -> int:
        """Written so far; a jpg job's preview.jpg shares the suffix."""
        suffix = f".{self.scenes[0].outputs.format}"
        written = sum(path.suffix == suffix for path in self.folder.rglob("*"))
        return min(written, self.total)

    @property
    def waiting(self) -> bool:
        """Whether the job is queued or running."""
        return self.state in ("queued", "running")


@dataclass
class State:
    """What the tools share; jobs take turns on `gpu`."""

    settings: Settings
    renders: Path = assets.CACHE / "renders"
    jobs: dict[str, Job] = field(default_factory=dict)
    gpu: asyncio.Lock = field(default_factory=asyncio.Lock)
    loading: asyncio.Semaphore = field(
        default_factory=lambda: asyncio.Semaphore(LOADING)
    )

    def scenarios(self) -> dict[str, Path]:
        """The scenarios a job can name, by file stem."""
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


def _url(state: State, ctx: Context[Any, Any]) -> str:
    """Where this client reaches the jobs' files: the public URL when there is one."""
    if public := state.settings.public_url:
        return f"{str(public).rstrip('/')}/renders"
    return f"http://{(ctx.headers or {}).get('host', 'localhost')}/renders"


def _bytes(scene: Scenario) -> float:
    """What a scene writes: its images, and the archive of them."""
    return 2 * sum(
        scene.outputs.frames
        * mount.camera.width_px
        * mount.camera.height_px
        * BYTES_PER_PX[mount.camera.band, scene.outputs.format]
        for mount in scene.mounts
        if mount.camera.band in scene.outputs.bands
    )


def _refuse_paths(lines: list[str]) -> None:
    """Refuse an override that names a file, where a preset name belongs."""
    for line in lines:
        try:
            names = list(_includes(tomllib.loads(line)))
        except RecursionError as error:
            raise ValueError("an override nests too deeply") from error
        # A path would read any TOML here, and an error echoes what it read.
        if any("/" in name or name.endswith(".toml") for name in names):
            raise ToolError(f"{line!r}: an override names presets, never files")


def _tried(
    path: Path, lines: list[str], variants: int, settings: Settings
) -> tuple[list[Scenario], float]:
    """The first scenes, and the most bytes the seeds `render` may try can write.
    Refuses any of those seeds past a clip, sample or camera limit, and the largest
    of them past the image limit.

    Args:
        path: The scenario's TOML file.
        lines: TOML overrides merged over it.
        variants: How many scenes to return; the image and byte totals take the
            largest this many of the seeds tried.
        settings: The server's limits.
    """
    _refuse_paths(lines)
    first = load(path, lines)
    scenes: list[Scenario] = []
    images, sizes = [], []
    for seed in seeds(first.seed, variants):
        scene = first if seed == first.seed else load(path, [*lines, f"seed = {seed}"])
        if scene.outputs.duration_s > settings.max_duration_s:
            raise ToolError(f"a clip lasts at most {settings.max_duration_s} s")
        samples = scene.outputs.samples
        if max(samples.eo, samples.ir) > settings.max_samples:
            raise ToolError(f"at most {settings.max_samples} samples per pixel")
        for mount in scene.mounts:
            camera = mount.camera
            if camera.width_px * camera.height_px > settings.max_pixels:
                raise ToolError(
                    f"{mount.name} is {camera.width_px} x {camera.height_px}; a "
                    f"camera has at most {settings.max_pixels} pixels"
                )
        if len(scenes) < variants:
            scenes.append(scene)
        images.append(scene.images)
        sizes.append(_bytes(scene))
    if (most := sum(heapq.nlargest(variants, images))) > settings.max_images:
        raise ToolError(f"{most} images; a job writes at most {settings.max_images}")
    return scenes, sum(heapq.nlargest(variants, sizes))


def _expire(state: State) -> list[str]:
    """Delete what is older than `KEEP_S` and name the jobs it was. Runs in a thread."""
    cutoff = time.time() - KEEP_S
    # A running job's folder may sit unchanged past the cutoff.
    live = {
        name
        for name, job in list(state.jobs.items())
        if job.waiting or job.state == "finishing"
    }
    gone = []
    for path in state.renders.glob("*"):
        if path.stem in live:
            continue
        try:
            if path.stat().st_mtime >= cutoff:
                continue
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink()
        except FileNotFoundError:  # `cancel` removed it first
            continue
        gone.append(path.stem)
    return gone


async def _prune(state: State) -> None:
    """Delete expired files, and forget their jobs."""
    for name in await asyncio.to_thread(_expire, state):
        state.jobs.pop(name, None)
    cutoff = time.time() - KEEP_S
    # A cancelled job leaves no folder to age.
    for name, job in list(state.jobs.items()):
        done = job.task is not None and job.task.done()
        if done and job.created_s < cutoff and not job.folder.exists():
            del state.jobs[name]


def _ahead(state: State) -> int:
    """How many jobs wait or run; refuses one more past `max_queued`."""
    ahead = sum(j.waiting for j in state.jobs.values())
    if ahead >= state.settings.max_queued:
        raise ToolError(
            f"{ahead} jobs queued; the server takes at most {state.settings.max_queued}"
        )
    return ahead


def catalog(ctx: Context[State, Any]) -> dict[str, Any]:
    """The scenarios `render` starts from, the meshes and skies a scenario can name,
    and the scenario's JSON schema. Any field can be a draw instead of a value:
    `{ uniform = [lo, hi] }` or `{ choice = [...] }`, drawn per seed.
    """
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
    overrides: Annotated[
        list[Annotated[str, Field(max_length=MAX_OVERRIDE_CHARS)]],
        Field(max_length=MAX_OVERRIDES),
    ]
    | None = None,
    variants: Annotated[int, Field(ge=1)] = 1,
) -> dict[str, Any]:
    """Queue a render of a scenario from `catalog` and return its job at once.

    `overrides` are TOML lines merged over the scenario: `sky.sun_elevation_deg = 5`,
    `sea.wind_speed_mps = { uniform = [2, 12] }`. `variants` renders that many
    seeds from the scenario's own, each drawn anew, skipping a seed whose hulls meet,
    up to the server's variant and image limits. Every seed it may try must fit
    the clip, sample and camera limits.
    """
    state = _state(ctx)
    settings = state.settings
    if (path := state.scenarios().get(scenario)) is None:
        raise ToolError(f"no scenario {scenario!r} in {sorted(state.scenarios())}")
    # Each variant writes an image at least, so this refuses before loading.
    if variants > (most := min(settings.max_variants, settings.max_images)):
        raise ToolError(f"at most {most} variants")
    lines = overrides or []
    state.renders.mkdir(parents=True, exist_ok=True)
    await _prune(state)
    # Refuse before loading, which is the cost.
    _ahead(state)
    try:
        async with state.loading:
            scenes, worst_bytes = await asyncio.to_thread(
                _tried, path, lines, variants, settings
            )
    # The loader, pydantic and tomllib raise ValueError; a missing file OSError.
    except ValueError as error:
        raise ToolError(str(error)) from error
    except OSError as error:
        raise ToolError(
            f"{type(error).__name__}: a file the scenario names could not be read"
        ) from error
    # Other calls may have queued while this one loaded.
    ahead = _ahead(state)
    name = uuid.uuid4().hex
    job = Job(state.renders / name, scenario, scenes, worst_bytes=worst_bytes)
    # A running job's images count twice, as written and as still to come; a
    # finishing one still writes its zip.
    writing = [j for j in state.jobs.values() if j.waiting or j.state == "finishing"]
    free = shutil.disk_usage(state.renders).free - sum(j.size_bytes for j in writing)
    free_gb = free / 1e9
    if (needs_gb := job.size_bytes / 1e9 + settings.min_free_gb) > free_gb:
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
        text = stderr.decode(errors="replace").strip()
        log.error(
            "seascape %s exited %s: %s", args[0], process.returncode, text[-2000:]
        )
        # The client gets the last line: the full trace holds this server's paths.
        raise RuntimeError(
            text.splitlines()[-1][:500] if text else f"exit {process.returncode}"
        )


async def _run(
    state: State, job: Job, path: Path, overrides: list[str], variants: int
) -> None:
    sets = [arg for line in overrides for arg in ("--set", line)]
    max_run_s = state.settings.max_run_s
    limit: asyncio.Timeout | None = None
    try:
        async with state.gpu:
            job.state, job.started_s = "running", time.time()
            # Made here, since the deadline is fixed when it is made, not entered:
            # the wait for the GPU is not the run. Past it, the process awaited is
            # killed as on `cancel`.
            limit = asyncio.timeout(max_run_s)
            async with limit:
                await _seascape(
                    "render", str(path), "-o", str(job.folder), "--variants",
                    str(variants), *sets,
                )  # fmt: skip
                if variants > 1:
                    job.scenes = await asyncio.to_thread(_written, job, path, overrides)
                for scene, folder in zip(job.scenes, job.folders, strict=True):
                    if scene.outputs.frames > 1:
                        await _seascape("video", str(folder))
        # Off the GPU: the next job starts while this one zips.
        job.state = "finishing"
        await asyncio.to_thread(_finish, job)
        job.state = "done"
    except TimeoutError as error:
        job.state = "failed"
        # Any other TimeoutError came from inside the job.
        ran_out = limit is not None and limit.expired()
        job.error = f"ran past {max_run_s:g} s" if ran_out else f"TimeoutError: {error}"
    except Exception as error:  # whatever it was, the client polling is told
        # A failed render was logged with its stderr already.
        log.error(
            "job %s failed: %s", job.folder.name, error,
            exc_info=not isinstance(error, RuntimeError),
        )  # fmt: skip
        job.state, job.error = "failed", f"{type(error).__name__}: {error}"


def _written(job: Job, path: Path, overrides: list[str]) -> list[Scenario]:
    """The scenario of each variant `render` wrote, by the seed its folder names."""
    return [load(path, [*overrides, f"seed = {f.name}"]) for f in job.folders]


def _finish(job: Job) -> None:
    """A preview of each of the first `PREVIEWS` non-exr stills, and an archive."""
    for scene, folder in zip(job.scenes[:PREVIEWS], job.folders, strict=False):
        if scene.outputs.frames == 1 and scene.outputs.format != "exr":
            with Picture.open(montage.compose(scene, folder)) as picture:
                picture.thumbnail((PREVIEW_PX, PREVIEW_PX))
                # Judgement: a preview, so a smaller file over fidelity.
                picture.convert("RGB").save(folder / "preview.jpg", quality=85)
    shutil.make_archive(str(job.folder), "zip", job.folder)


def _files(job: Job, renders: Path) -> list[Path]:
    """The job's files, relative to `renders`, but its numbered frames."""
    return [
        path.relative_to(renders)
        for path in sorted(job.folder.rglob("*"))
        if path.is_file() and not path.stem.isdigit()
    ]


async def render_result(job: str, ctx: Context[State, Any]) -> list[Any]:
    """Wait briefly for a job from `render`, then report it: its progress, its error,
    or its files as URLs with a preview of the first few stills (none for exr).
    """
    state = _state(ctx)
    found = state.jobs.get(job)
    if found is None or found.task is None:
        raise ToolError(
            f"no job {job!r}: a job is forgotten when it expires or the server restarts"
        )
    deadline = time.monotonic() + WAIT_S
    while not found.task.done() and (left := deadline - time.monotonic()) > 0:
        await asyncio.wait({found.task}, timeout=min(PROGRESS_S, left))
        await ctx.report_progress(
            await asyncio.to_thread(lambda: found.images), found.total
        )
    report: dict[str, Any] = {"status": found.state}
    previews: list[Image] = []
    if found.state == "failed":
        report["error"] = found.error
    elif found.state == "done":
        url = _url(state, ctx)
        listed = await asyncio.to_thread(_files, found, state.renders)
        # A sequence's frames are in the archive only.
        report["files"] = [
            f"{url}/{job}.zip",
            *(f"{url}/{path}" for path in listed[:MAX_LISTED]),
        ]
        if len(listed) > MAX_LISTED:
            report["more_in_the_archive"] = len(listed) - MAX_LISTED
        previews = [
            Image(path=preview)
            for folder in found.folders
            if (preview := folder / "preview.jpg").exists()
        ]
    elif found.state == "queued":
        names = list(state.jobs)
        report["jobs_ahead"] = sum(
            state.jobs[n].waiting for n in names[: names.index(job)]
        )
    elif found.state == "running":
        done, total = await asyncio.to_thread(lambda: found.images), found.total
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
    state = _state(ctx)
    url = _url(state, ctx)
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
    """The MCP server, its tools sharing `state`."""
    state.renders.mkdir(parents=True, exist_ok=True)

    # The lifespan runs per connection; jobs outlive one.
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
    # A client may run a read-only tool without asking; `render` and `cancel` change
    # what the server holds.
    read_only = ToolAnnotations(read_only_hint=True, open_world_hint=False)
    served.add_tool(catalog, annotations=read_only)
    served.add_tool(jobs, annotations=read_only)
    served.add_tool(render, annotations=ToolAnnotations(open_world_hint=False))
    served.add_tool(
        cancel,
        annotations=ToolAnnotations(destructive_hint=True, open_world_hint=False),
    )
    # Unstructured: the previews are images, which structured content cannot carry.
    served.add_tool(render_result, annotations=read_only, structured_output=False)
    return served


def serve(state: State, host: str, port: int) -> None:
    """Probe the GPU, then serve `app` on `host`:`port`."""
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
    except subprocess.CalledProcessError as error:
        # Serving anyway: each job reports what the render itself finds.
        print(f"The GPU probe failed: {error.stderr.strip()[-500:]}", flush=True)
    else:
        words = probe.stdout.split()
        backend = words[-1] if words else "None"
        print(
            f"Cycles renders on {'the CPU' if backend == 'None' else backend}",
            flush=True,
        )
    # The access log would write the secret path.
    uvicorn.run(
        app(state, host),
        host=host,
        port=port,
        access_log=state.settings.mcp_secret is None,
    )


def app(state: State, host: str) -> Starlette:
    """MCP at /mcp/<secret> (/mcp with no secret), the jobs' files at /renders/."""
    secret = state.settings.mcp_secret
    path = f"/mcp/{secret.get_secret_value()}" if secret else "/mcp"
    # A web page can point a browser on the network at this port by DNS rebinding,
    # so MCP refuses any Host or Origin not listed.
    hosts = list(state.settings.allowed_hosts)
    if tunnel := state.settings.public_url:
        # The Host header carries the port when it is not the scheme's default.
        hosts += [tunnel.host or "", f"{tunnel.host}:{tunnel.port}"]
    security = TransportSecuritySettings(
        allowed_hosts=hosts,
        allowed_origins=[f"{s}://{h}" for h in hosts for s in ("http", "https")],
    )
    served = mcp(state).streamable_http_app(
        host=host, streamable_http_path=path, transport_security=security
    )
    served.mount("/renders", StaticFiles(directory=state.renders))
    return served
