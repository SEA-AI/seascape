"""Jobs: what a render may ask for, how they queue, and how one runs.

A job runs the CLI in a process of its own, so Blender's crashes and leaks end with
it. Its files are deleted after `KEEP_S`.
"""

import asyncio
import heapq
import io
import logging
import os
import shutil
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Literal

from mcp.server.mcpserver.exceptions import ToolError
from PIL import Image as Picture

from seascape import assets
from seascape.config import (
    Band,
    ImageFormat,
    Scenario,
    instantiate,
    resolve,
    seeds,
)
from seascape.model import ScenarioWarning
from seascape.outputs import montage, preview, provenance
from seascape.server.settings import Settings

KEEP_S = 24 * 3600
# Judgement: how much longer than `KEEP_S` an expired file may stay.
SWEEP_S = 3600.0
# The long edge Claude reads an image at (Anthropic vision docs).
PREVIEW_PX = 1568
SHEET = "montage.jpg"
# `warnings.catch_warnings` swaps interpreter-wide state, so loads run one at a time.
LOADING = 1
# Under uvicorn's handlers, so the lines reach the server's log.
log = logging.getLogger("uvicorn.error")


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
    overrides: list[str] = field(default_factory=list)

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
        """Written so far; a jpg job's sheet shares the suffix."""
        suffix = f".{self.scenes[0].outputs.format}"
        written = sum(path.suffix == suffix for path in self.folder.rglob("*"))
        return min(written, self.total)

    @property
    def waiting(self) -> bool:
        """Whether the job is queued or running."""
        return self.state in ("queued", "running")

    @property
    def writing(self) -> bool:
        """Whether the job still writes files: waiting, or zipping."""
        return self.waiting or self.state == "finishing"


@dataclass
class State:
    """What the tools share; jobs take turns on `gpu`."""

    settings: Settings
    renders: Path = assets.CACHE / "renders"
    jobs: dict[str, Job] = field(default_factory=dict)
    gpu: asyncio.Lock = field(default_factory=asyncio.Lock)
    loading: asyncio.Semaphore = field(
        default_factory=partial(asyncio.Semaphore, LOADING)
    )

    def scenarios(self) -> dict[str, Path]:
        """The scenarios a job can name, by file stem."""
        return {p.stem: p for p in sorted(self.settings.scenarios.glob("*.toml"))}


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


def _tried(
    path: Path, lines: list[str], variants: int, settings: Settings
) -> tuple[list[Scenario], float, list[str]]:
    """`_checked`, and what the scenario warned of while loading, so the client hears
    of what it ignored. Other threads' warnings inside the window are swallowed.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scenes, worst_bytes = _checked(path, lines, variants, settings)
    heard = (str(w.message) for w in caught if issubclass(w.category, ScenarioWarning))
    return scenes, worst_bytes, list(dict.fromkeys(heard))


def _checked(
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
    try:
        tree = resolve(path, lines, files=False)
        first = instantiate(tree)
    except RecursionError as error:
        raise ValueError("an override nests too deeply") from error
    scenes: list[Scenario] = []
    images, sizes = [], []
    for seed in seeds(first.seed, variants):
        scene = first if seed == first.seed else instantiate(tree, seed)
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
    """Delete what is older than `KEEP_S` and name the jobs it was. Runs in a thread.

    A job's folder and its archive age together, by the newer of the two.
    """
    cutoff = time.time() - KEEP_S
    # A running job's folder may sit unchanged past the cutoff.
    live = {name for name, job in list(state.jobs.items()) if job.writing}
    by_job: dict[str, list[Path]] = {}
    for path in state.renders.glob("*"):
        by_job.setdefault(path.stem, []).append(path)
    gone = []
    for name, paths in by_job.items():
        if name in live:
            continue
        try:
            if max(path.stat().st_mtime for path in paths) >= cutoff:
                continue
        except FileNotFoundError:  # `cancel` removed it first
            continue
        for path in paths:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
        gone.append(name)
    return gone


async def prune(state: State) -> None:
    """Delete expired files, and forget their jobs."""
    for name in await asyncio.to_thread(_expire, state):
        state.jobs.pop(name, None)
    cutoff = time.time() - KEEP_S
    # A cancelled job leaves no folder to age.
    for name, job in list(state.jobs.items()):
        done = job.task is not None and job.task.done()
        if done and job.created_s < cutoff and not job.folder.exists():
            del state.jobs[name]


async def sweep(state: State) -> None:
    """Prune every `SWEEP_S`, so an idle server still frees its disk."""
    while True:
        try:
            await prune(state)
        except Exception:  # one failed sweep must not end the next
            log.exception("pruning the renders failed")
        await asyncio.sleep(SWEEP_S)


async def load(
    state: State, path: Path, lines: list[str], variants: int
) -> tuple[list[Scenario], float, list[str]]:
    """`_tried` in a thread under a `state.loading` slot, held until the thread ends."""
    async with state.loading:
        return await asyncio.to_thread(_tried, path, lines, variants, state.settings)


def drop(state: State, name: str, loading: asyncio.Task[Any]) -> None:
    """Forget a job that never started, once its load ends; read the outcome, which
    no caller will when it went away.
    """
    state.jobs.pop(name, None)
    if not loading.cancelled():
        loading.exception()


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


async def run(
    state: State, job: Job, path: Path, overrides: list[str], variants: int
) -> None:
    """Take the GPU in turn, render and encode the job, then zip it, with its sheets."""
    sets = [arg for line in overrides for arg in ("--set", line)]
    max_run_s = state.settings.max_run_s
    limit: asyncio.Timeout | None = None
    try:
        async with state.gpu:
            job.state, job.started_s = "running", time.time()
            # Made after the GPU wait, which the deadline must not count. Past it, the
            # awaited process is killed as on `cancel`.
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
        log.exception("job %s failed", job.folder.name)
        job.state, job.error = "failed", f"{type(error).__name__}: {error}"


def _written(job: Job, path: Path, overrides: list[str]) -> list[Scenario]:
    """The scenario of each variant `render` wrote, by the seed its folder names."""
    tree = resolve(path, overrides, files=False)
    return [instantiate(tree, int(f.name)) for f in job.folders]


def marked_jpeg(
    picture: Picture.Image, exif: Picture.Exif, *, below: bool = False
) -> bytes:
    """`picture` as a marked JPEG at most `PREVIEW_PX` a side, shrunk in place."""
    picture.thumbnail((PREVIEW_PX, PREVIEW_PX), Picture.Resampling.LANCZOS)
    buffer = io.BytesIO()
    # Judgement: a preview, so a smaller file over fidelity.
    preview.watermark(picture, below=below).save(buffer, "JPEG", quality=85, exif=exif)
    return buffer.getvalue()


def _finish(job: Job) -> None:
    """An archive of the frames, and beside it a watermarked sheet per variant (none
    for exr). A sheet that cannot be made is logged and left out: the render is done.
    """
    shutil.make_archive(str(job.folder), "zip", job.folder)
    if job.scenes[0].outputs.format != "exr":
        for scene, folder in zip(job.scenes, job.folders, strict=True):
            try:
                sheet = montage.sheet(scene, folder)
                exif = provenance.exif(job.scenario, scene.seed, job.overrides)
                (folder / SHEET).write_bytes(marked_jpeg(sheet, exif, below=True))
            # `montage.sheet`: ValueError for no frames, OSError for a missing one.
            except (OSError, ValueError):
                log.exception("job %s: no sheet for %s", job.folder.name, folder.name)
