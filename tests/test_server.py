"""The render server, without Blender."""

import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx2
import pytest
from mcp import Client
from mcp.types import TextContent
from PIL import Image

from seascape import server
from seascape.config import load_variants
from seascape.server import SCENARIOS

BASELINE = SCENARIOS / "baseline.toml"


def _call(tool: str, **arguments: Any) -> Any:
    async def call() -> Any:
        async with Client(server.server) as client:
            return await client.call_tool(tool, arguments)

    return asyncio.run(call())


@pytest.mark.parametrize(
    "line",
    [
        'objects = [{ preset = "/etc/hosts.toml" }]',
        'objects = [{ preset = "no_such_preset" }]',
        "objects = [{ preset = 5 }]",
        'extends = "baseline.toml"',
        "sky.sun_elevation_deg = 100.0",
        "sky.sun_elevation_deg =",
    ],
)
def test_a_bad_override_is_refused_before_any_job(line: str) -> None:
    result = _call("render", overrides=[line])
    assert result.is_error
    assert not server._jobs


def test_an_unknown_scenario_is_refused() -> None:
    result = _call("render", scenario="../baseline")
    assert result.is_error
    assert not server._jobs


@pytest.mark.parametrize(
    ("overrides", "variants", "refusal"),
    [
        (["outputs.duration_s = 61.0"], 1, "a clip lasts"),
        (
            ["rigs.bow.cameras.eo = { width_px = 7680, height_px = 4320 }"],
            1,
            "pixels",
        ),
        (["outputs.duration_s = 60.0", "outputs.fps = 100"], 2, "images"),
    ],
)
def test_an_oversized_job_is_refused(
    overrides: list[str], variants: int, refusal: str
) -> None:
    result = _call("render", overrides=overrides, variants=variants)
    assert result.is_error
    assert refusal in result.content[0].text
    assert not server._jobs


def test_a_full_disk_refuses_a_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server, "RENDERS", tmp_path)
    monkeypatch.setattr(server, "MIN_FREE_GB", 2.0**50)
    result = _call("render")
    assert result.is_error
    assert "GB free" in result.content[0].text
    assert not server._jobs


def test_jobs_reports_the_free_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server, "RENDERS", tmp_path)
    listed = _call("jobs").structured_content
    assert listed["jobs"] == []
    assert listed["free_gb"] > 0


def test_a_job_older_than_a_day_is_deleted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server, "RENDERS", tmp_path)
    old, new = tmp_path / "old", tmp_path / "new"
    for folder in (old, new):
        folder.mkdir()
        (folder / "labels.json").touch()
    (tmp_path / "old.zip").touch()
    a_day_ago = time.time() - server.KEEP_S - 1
    for path in (old, tmp_path / "old.zip"):
        os.utime(path, (a_day_ago, a_day_ago))
    server._prune()
    assert [path.name for path in tmp_path.iterdir()] == ["new"]


def test_a_variant_job_finds_the_seeds_render_kept(tmp_path: Path) -> None:
    scenes = load_variants(SCENARIOS / "open-sea.toml", count=2)
    first = scenes[0].seed
    # `render` skipped first + 1, whose hulls met.
    for seed in (first, first + 2):
        (tmp_path / str(seed)).mkdir()
    (tmp_path / "labels.json").touch()
    job = server.Job(tmp_path, "open-sea", scenes)
    assert job.folders == [tmp_path / str(first), tmp_path / str(first + 2)]
    assert [s.seed for s in server._written(job, SCENARIOS / "open-sea.toml", [])] == [
        first,
        first + 2,
    ]


@pytest.mark.parametrize(
    ("allowed", "host", "refused"),
    [
        ("", "localhost:8765", False),
        ("", "gpu-box:8765", True),
        ("gpu-box:*", "gpu-box:8765", False),
        ("gpu-box:*", "evil.example", True),
    ],
)
def test_listed_hosts_are_the_only_ones_mcp_answers(
    allowed: str,
    host: str,
    refused: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "RENDERS", tmp_path)
    monkeypatch.setenv("SEASCAPE_ALLOWED_HOSTS", allowed)
    served = server.app("0.0.0.0")

    async def post() -> int:
        async with served.router.lifespan_context(served):
            transport = httpx2.ASGITransport(app=served)
            async with httpx2.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                response = await client.post("/mcp", json={}, headers={"host": host})
                return response.status_code

    assert (asyncio.run(post()) == 421) is refused


def test_a_hung_gpu_probe_still_serves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(server, "RENDERS", tmp_path)

    def hang(*args: Any, **kwargs: Any) -> None:
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    served: list[Any] = []
    monkeypatch.setattr(server.subprocess, "run", hang)
    monkeypatch.setattr(server.uvicorn, "run", lambda app, **_: served.append(app))
    server.serve("127.0.0.1", 0)
    assert served
    assert "no answer" in capsys.readouterr().out


def test_jobs_lists_a_finished_job_whose_archive_is_gone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server, "RENDERS", tmp_path)
    scenes = load_variants(SCENARIOS / "open-sea.toml")
    job = server.Job(tmp_path / "gone", "open-sea", scenes, state="done")
    monkeypatch.setattr(server, "_jobs", {"gone": job})
    (listed,) = _call("jobs").structured_content["jobs"]
    assert listed["status"] == "done"
    assert "archive" not in listed


@pytest.mark.parametrize(("value", "expected"), [("", 7.0), ("2.5", 2.5)])
def test_a_limit_reads_the_environment(
    value: str, expected: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SEASCAPE_PROBE", value)
    assert server._limit("PROBE", 7.0) == expected


@pytest.mark.parametrize("value", ["4k", "nan", "inf", "0", "-1"])
def test_a_limit_that_is_not_a_positive_number_names_its_variable(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SEASCAPE_PROBE", value)
    with pytest.raises(ValueError, match="SEASCAPE_PROBE"):
        server._limit("PROBE", 7.0)


def test_a_finishing_job_is_not_ahead_of_the_next(tmp_path: Path) -> None:
    scenes = load_variants(SCENARIOS / "open-sea.toml")
    job = server.Job(tmp_path, "open-sea", scenes, state="finishing")
    assert not job.waiting


def test_a_finished_job_reports_its_files_and_a_preview_per_variant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server, "RENDERS", tmp_path)
    monkeypatch.setattr(server, "_jobs", {})
    # A lock binds to the first event loop that waits on it.
    monkeypatch.setattr(server, "_gpu", asyncio.Lock())

    async def render(*args: str) -> None:  # Blender's frames, without Blender
        (job,) = server._jobs.values()
        for scene in job.scenes:
            folder = job.folder / str(scene.seed)  # as `render --variants` names it
            folder.mkdir(parents=True, exist_ok=True)
            for mount in scene.mounts:
                frame = folder / f"{mount.name}.{scene.outputs.format}"
                Image.new("RGB", (64, 48)).save(frame)

    monkeypatch.setattr(server, "_seascape", render)

    async def call() -> Any:
        async with Client(server.server) as client:
            started = await client.call_tool("render", {"variants": 2})
            job = started.structured_content["job"]
            return await client.call_tool("render_result", {"job": job})

    result = asyncio.run(call())
    report = json.loads(result.content[0].text)
    assert report["status"] == "done", report
    assert report["files"][0].endswith(".zip")
    assert sum(url.endswith("/preview.jpg") for url in report["files"]) == 2
    assert [(c.type, c.mime_type) for c in result.content[1:]] == [
        ("image", "image/jpeg")
    ] * 2


def _sleeper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`seascape render` as a process that writes into its job's folder and waits."""
    script = tmp_path / "sleeper"
    # $6 is the folder after `-m seascape.cli render <scenario> -o`.
    script.write_text('#!/bin/sh\nmkdir -p "$6"\necho $$ > "$6/pid"\nexec sleep 60\n')
    script.chmod(0o755)
    monkeypatch.setattr(sys, "executable", str(script))
    monkeypatch.setattr(server, "RENDERS", tmp_path / "renders")
    monkeypatch.setattr(server, "_jobs", {})
    # Bound to the loop that first waits on it, so one per test's loop.
    monkeypatch.setattr(server, "_gpu", asyncio.Lock())


async def _started(name: str) -> int:
    pid = server._jobs[name].folder / "pid"
    while not pid.exists() or not pid.read_text():
        await asyncio.sleep(0.01)
    return int(pid.read_text())


def test_cancel_stops_a_job_and_hands_the_gpu_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _sleeper(tmp_path, monkeypatch)

    async def call() -> None:
        async with Client(server.server) as client:
            first, queued, last = [
                (await client.call_tool("render", {})).structured_content["job"]
                for _ in range(3)
            ]
            pid = await _started(first)

            cancelled = await client.call_tool("cancel", {"job": queued})
            assert cancelled.structured_content["status"] == "cancelled"
            assert server._jobs[first].state == "running"

            await client.call_tool("cancel", {"job": first})
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
            assert not server._jobs[first].folder.exists()
            reported = await client.call_tool("render_result", {"job": first})
            assert isinstance(report := reported.content[0], TextContent)
            assert json.loads(report.text) == {"status": "cancelled"}

            await _started(last)
            assert server._jobs[last].state == "running"
            await client.call_tool("cancel", {"job": last})
            again = await client.call_tool("cancel", {"job": last})
            assert again.is_error
            assert isinstance(refusal := again.content[0], TextContent)
            assert "cancelled" in refusal.text

    asyncio.run(call())
    assert not server._gpu.locked()


@pytest.mark.parametrize(("fmt", "refused"), [("jpg", False), ("exr", True)])
def test_a_job_is_refused_when_its_images_would_fill_the_disk(
    fmt: str, refused: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With room for one jpg job, beside one already queued, and no margin."""
    monkeypatch.setattr(server, "RENDERS", tmp_path)
    monkeypatch.setattr(server, "MIN_FREE_GB", 0.0)
    queued = server.Job(tmp_path / "queued", "baseline", load_variants(BASELINE))
    monkeypatch.setattr(server, "_jobs", {"queued": queued})
    room = SimpleNamespace(free=2 * queued.bytes)
    monkeypatch.setattr(shutil, "disk_usage", lambda _: room)

    async def run(*_: object) -> None:
        pass

    monkeypatch.setattr(server, "_run", run)
    result = _call("render", overrides=[f'outputs.format = "{fmt}"'])
    assert result.is_error is refused
    assert len(server._jobs) == 1 + (not refused)


def test_a_job_past_its_deadline_fails_and_frees_the_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _sleeper(tmp_path, monkeypatch)
    monkeypatch.setattr(server, "MAX_RUN_S", 0.5)

    async def call() -> None:
        async with Client(server.server) as client:
            started = await client.call_tool("render", {})
            name = started.structured_content["job"]
            pid = await _started(name)
            task = server._jobs[name].task
            assert task is not None
            await task
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)

    asyncio.run(call())
    (job,) = server._jobs.values()
    assert (job.state, job.error) == ("failed", "ran past 0.5 s")
    assert not server._gpu.locked()


def test_a_full_queue_refuses_a_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server, "RENDERS", tmp_path)
    monkeypatch.setattr(server, "MAX_QUEUED", 1)
    queued = server.Job(tmp_path / "queued", "baseline", load_variants(BASELINE))
    monkeypatch.setattr(server, "_jobs", {"queued": queued})
    result = _call("render")
    assert result.is_error
    assert "at most 1" in result.content[0].text
    assert len(server._jobs) == 1


def test_a_waiting_result_reports_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _sleeper(tmp_path, monkeypatch)
    monkeypatch.setattr(server, "WAIT_S", 0.3)
    monkeypatch.setattr(server, "PROGRESS_S", 0.1)
    reported: list[tuple[float, float | None]] = []

    async def progress(
        progress: float, total: float | None, message: str | None
    ) -> None:
        reported.append((progress, total))

    async def call() -> None:
        async with Client(server.server) as client:
            started = await client.call_tool("render", {})
            name = started.structured_content["job"]
            await client.call_tool(
                "render_result", {"job": name}, progress_callback=progress
            )
            await client.call_tool("cancel", {"job": name})

    asyncio.run(call())
    assert len(reported) >= 2
    assert set(reported) == {(0, 2)}


def test_an_old_cancelled_job_is_forgotten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server, "RENDERS", tmp_path)
    scenes = load_variants(SCENARIOS / "open-sea.toml")

    async def forget() -> list[str]:
        done = asyncio.create_task(asyncio.sleep(0))
        await done
        old = server.Job(tmp_path / "old", "open-sea", scenes, state="cancelled")
        new = server.Job(tmp_path / "new", "open-sea", scenes, state="cancelled")
        old.created_s -= server.KEEP_S + 1
        old.task = new.task = done
        monkeypatch.setattr(server, "_jobs", {"old": old, "new": new})
        server._prune()
        return list(server._jobs)

    assert asyncio.run(forget()) == ["new"]
