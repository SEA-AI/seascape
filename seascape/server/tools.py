"""The MCP tools."""

import asyncio
import json
import shutil
import time
import uuid
from functools import partial
from pathlib import Path
from typing import Annotated, Any

from mcp.server.mcpserver import Context, Image
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

from seascape import assets
from seascape.assets import skies
from seascape.config import schema_section, schema_sections
from seascape.server.jobs import (
    PREVIEW_STEM,
    Job,
    State,
    drop,
    files,
    load,
    prune,
    run,
)

# A client may end a tool call after a minute.
WAIT_S = 45.0
# Progress report interval; keeps a client's idle timeout from ending the wait.
PROGRESS_S = 5.0
# Judgement: a result a client can read; the archive holds the rest.
MAX_LISTED = 200
# Judgement: far more and longer than a scenario's overrides.
MAX_OVERRIDES = 50
MAX_OVERRIDE_CHARS = 1000


def _state(ctx: Context[State, Any]) -> State:
    return ctx.request_context.lifespan_context


def _url(state: State, ctx: Context[Any, Any]) -> str:
    """Where this client reaches the jobs' files."""
    if public := state.settings.public_url:
        return f"{str(public).rstrip('/')}/renders"
    return f"http://{(ctx.headers or {}).get('host', 'localhost')}/renders"


def catalog(ctx: Context[State, Any]) -> dict[str, Any]:
    """The scenarios `render` starts from (`scenario` reads one), the meshes and skies
    a scenario can name, and the schema's sections (`schema` reads one). Any field
    can be a draw instead of a value: `{ uniform = [lo, hi] }` or `{ choice = [...] }`,
    drawn per seed.
    """
    return {
        "scenarios": list(_state(ctx).scenarios()),
        "schema_sections": schema_sections(),
        "meshes": {
            name: f"{mesh.kind}, {mesh.category}, {mesh.size}"
            for name, mesh in assets.manifest().items()
        },
        "skies": {name: photo.sun for name, photo in skies.library().items()},
    }


def scenario(name: str, ctx: Context[State, Any]) -> str:
    """The TOML of one scenario from `catalog`."""
    scenarios = _state(ctx).scenarios()
    if (path := scenarios.get(name)) is None:
        raise ToolError(f"no scenario {name!r} in {sorted(scenarios)}")
    return path.read_text(encoding="utf-8")


def schema(section: str) -> dict[str, Any]:
    """The schema of one section from `catalog`, with the definitions it uses."""
    try:
        return schema_section(section)
    except KeyError:
        raise ToolError(
            f"no section {section!r} in {sorted(schema_sections())}"
        ) from None


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
    `sea.wind_speed_mps = { uniform = [2, 12] }`. A list such as `objects` replaces the
    whole list. `variants` renders that many seeds from the scenario's own, each drawn
    anew, skipping a seed whose hulls meet, up to the server's variant and image
    limits. Every seed it may try must fit the clip, sample and camera limits.
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
    await prune(state)
    # Refuse before loading, which is the cost. No await from here to the job's place
    # in the queue, or a burst all sees it empty.
    ahead = sum(j.waiting for j in state.jobs.values())
    if ahead >= settings.max_queued:
        raise ToolError(
            f"{ahead} jobs queued; the server takes at most {settings.max_queued}"
        )
    name = uuid.uuid4().hex
    # Queued while it loads, so a burst fills the queue before any call pays for a
    # load.
    job = Job(state.renders / name, scenario, [])
    state.jobs[name] = job
    loading = asyncio.create_task(load(state, path, lines, variants))
    try:
        try:
            job.scenes, job.worst_bytes = await asyncio.shield(loading)
        # The loader, pydantic and tomllib raise ValueError; a missing file OSError.
        except ValueError as error:
            raise ToolError(str(error)) from error
        except OSError as error:
            raise ToolError(
                f"{type(error).__name__}: a file the scenario names could not be read"
            ) from error
        # A running job's images count twice, as written and as still to come; a
        # finishing one still writes its zip.
        writing = [j for j in state.jobs.values() if j is not job and j.writing]
        free = shutil.disk_usage(state.renders).free - sum(
            j.size_bytes for j in writing
        )
        free_gb = free / 1e9
        if (needs_gb := job.size_bytes / 1e9 + settings.min_free_gb) > free_gb:
            raise ToolError(
                f"{free_gb:.0f} GB free after the jobs queued; this job needs "
                f"{needs_gb:.0f}, {settings.min_free_gb:.0f} of it kept free"
            )
        job.task = asyncio.create_task(run(state, job, path, lines, variants))
        return {"job": name, "images": job.total, "jobs_ahead": ahead}
    finally:
        # Refused, failed or abandoned. The load thread cannot be stopped, so the
        # place is kept until it ends.
        if job.task is None:
            loading.add_done_callback(partial(drop, state, name))


async def render_result(job: str, ctx: Context[State, Any]) -> list[Any]:
    """Wait briefly for a job from `render`, then report it: its progress, its error,
    or its files as URLs with a preview per camera of the first variant (none for exr).
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
        listed = await asyncio.to_thread(files, found, state.renders)
        # A sequence's frames are in the archive only.
        report["files"] = [
            f"{url}/{job}.zip",
            *(f"{url}/{path}" for path in listed[:MAX_LISTED]),
        ]
        if len(listed) > MAX_LISTED:
            report["more_in_the_archive"] = len(listed) - MAX_LISTED
        previews = [
            Image(path=path)
            for path in sorted(found.folders[0].glob(f"{PREVIEW_STEM}*.jpg"))
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
