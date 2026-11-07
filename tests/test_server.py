"""The render server, without Blender."""

import asyncio
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx2
import pytest
from mcp import Client
from mcp.types import TextContent
from PIL import Image
from pydantic import ValidationError

from seascape import server
from tests.scenarios import BASELINE, OPEN_SEA, SCENARIOS, load_variants

SECRET = "s" * 32


def _state(renders: Path, **limits: Any) -> server.State:
    return server.State(server.Settings(scenarios=SCENARIOS, **limits), renders)


async def _statuses(served: Any, path: str, *hosts: str) -> list[int]:
    async with served.router.lifespan_context(served):
        transport = httpx2.ASGITransport(app=served)
        async with httpx2.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            return [
                (await client.post(path, json={}, headers={"host": h})).status_code
                for h in hosts
            ]


@pytest.fixture
def state(tmp_path: Path) -> server.State:
    return _state(tmp_path)


def _call(state: server.State, tool: str, **arguments: Any) -> Any:
    async def call() -> Any:
        async with Client(server.mcp(state)) as client:
            return await client.call_tool(tool, arguments)

    return asyncio.run(call())


def test_only_the_tools_that_change_state_ask_a_client_for_approval(
    tmp_path: Path,
) -> None:
    async def listed() -> dict[str, bool]:
        async with Client(server.mcp(_state(tmp_path))) as client:
            tools = (await client.list_tools()).tools
            return {
                t.name: bool(t.annotations and t.annotations.read_only_hint)
                for t in tools
            }

    assert asyncio.run(listed()) == {
        "catalog": True,
        "jobs": True,
        "render_result": True,
        "render": False,
        "cancel": False,
    }


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
def test_a_bad_override_is_refused_before_any_job(
    line: str, state: server.State
) -> None:
    result = _call(state, "render", overrides=[line])
    assert result.is_error
    assert not state.jobs


def test_an_unknown_scenario_is_refused(state: server.State) -> None:
    result = _call(state, "render", scenario="../baseline")
    assert result.is_error
    assert not state.jobs


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
        # The first seeds' clips fit; a seed tried after one whose hulls meet does not.
        (["outputs.duration_s = { uniform = [0.0, 70.0] }"], 2, "a clip lasts"),
        # The first seeds' images fit; two seeds render may try instead do not.
        (
            ["outputs.duration_s = { uniform = [0.0, 60.0] }", "outputs.fps = 50"],
            2,
            "images",
        ),
    ],
)
def test_an_oversized_job_is_refused(
    overrides: list[str], variants: int, refusal: str, state: server.State
) -> None:
    result = _call(state, "render", overrides=overrides, variants=variants)
    assert result.is_error
    assert refusal in result.content[0].text
    assert not state.jobs


def test_a_full_disk_refuses_a_job(tmp_path: Path) -> None:
    state = _state(tmp_path, min_free_gb=2.0**50)
    result = _call(state, "render")
    assert result.is_error
    assert "GB free" in result.content[0].text
    assert not state.jobs


def test_jobs_reports_the_free_disk(state: server.State) -> None:
    listed = _call(state, "jobs").structured_content
    assert listed["jobs"] == []
    assert listed["free_gb"] > 0


def test_a_job_older_than_a_day_is_deleted(tmp_path: Path, state: server.State) -> None:
    old, new = tmp_path / "old", tmp_path / "new"
    for folder in (old, new):
        folder.mkdir()
        (folder / "labels.json").touch()
    (tmp_path / "old.zip").touch()
    a_day_ago = time.time() - server.KEEP_S - 1
    for path in (old, tmp_path / "old.zip"):
        os.utime(path, (a_day_ago, a_day_ago))
    asyncio.run(server._prune(state))
    assert [path.name for path in tmp_path.iterdir()] == ["new"]


def test_a_variant_job_finds_the_seeds_render_kept(tmp_path: Path) -> None:
    scenes = load_variants(OPEN_SEA, count=2)
    first = scenes[0].seed
    # `render` skipped first + 1, whose hulls met.
    for seed in (first, first + 2):
        (tmp_path / str(seed)).mkdir()
    (tmp_path / "labels.json").touch()
    job = server.Job(tmp_path, "open-sea", scenes)
    assert job.folders == [tmp_path / str(first), tmp_path / str(first + 2)]
    assert [s.seed for s in server._written(job, OPEN_SEA, [])] == [
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
    monkeypatch.setenv("SEASCAPE_ALLOWED_HOSTS", allowed)
    served = server.app(_state(tmp_path), "0.0.0.0")

    [status] = asyncio.run(_statuses(served, "/mcp", host))
    assert (status == 421) is refused


def test_a_hung_gpu_probe_still_serves(
    state: server.State,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def hang(*args: Any, **kwargs: Any) -> None:
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    served: list[Any] = []
    monkeypatch.setattr(server.subprocess, "run", hang)
    monkeypatch.setattr(server.uvicorn, "run", lambda app, **_: served.append(app))
    server.serve(state, "127.0.0.1", 0)
    assert served
    assert "no answer" in capsys.readouterr().out


@pytest.mark.parametrize(("secret", "logged"), [(None, True), (SECRET, False)])
def test_a_secret_path_is_not_access_logged(
    secret: str | None,
    logged: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs: dict[str, Any] = {}
    monkeypatch.setattr(
        server.subprocess,
        "run",
        lambda *_, **__: SimpleNamespace(stdout="None"),
    )
    monkeypatch.setattr(server.uvicorn, "run", lambda _, **kw: kwargs.update(kw))
    server.serve(_state(tmp_path, mcp_secret=secret), "127.0.0.1", 0)
    assert kwargs["access_log"] is logged


def test_jobs_lists_a_finished_job_whose_archive_is_gone(
    tmp_path: Path, state: server.State
) -> None:
    scenes = load_variants(OPEN_SEA)
    state.jobs["gone"] = server.Job(tmp_path / "gone", "open-sea", scenes, state="done")
    (listed,) = _call(state, "jobs").structured_content["jobs"]
    assert listed["status"] == "done"
    assert "archive" not in listed


@pytest.mark.parametrize(("value", "expected"), [("", 10.0), ("2.5", 2.5)])
def test_a_limit_reads_the_environment(
    value: str, expected: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SEASCAPE_MIN_FREE_GB", value)
    assert server.Settings(scenarios=SCENARIOS).min_free_gb == expected


@pytest.mark.parametrize("value", ["4k", "inf", "0"])
def test_a_limit_that_is_not_a_positive_number_names_its_field(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SEASCAPE_MIN_FREE_GB", value)
    with pytest.raises(ValidationError, match="min_free_gb"):
        server.Settings(scenarios=SCENARIOS)


def test_a_finishing_job_is_not_ahead_of_the_next(tmp_path: Path) -> None:
    scenes = load_variants(OPEN_SEA)
    job = server.Job(tmp_path, "open-sea", scenes, state="finishing")
    assert not job.waiting


def test_a_finished_job_reports_its_files_and_a_preview_per_variant(
    state: server.State, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def render(*_args: str) -> None:  # Blender's frames, without Blender
        (job,) = state.jobs.values()
        for scene in job.scenes:
            folder = job.folder / str(scene.seed)  # as `render --variants` names it
            folder.mkdir(parents=True, exist_ok=True)
            for mount in scene.mounts:
                frame = folder / f"{mount.name}.{scene.outputs.format}"
                Image.new("RGB", (64, 48)).save(frame)

    monkeypatch.setattr(server, "_seascape", render)

    async def call() -> Any:
        async with Client(server.mcp(state)) as client:
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


def _sleeper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **limits: Any
) -> server.State:
    """`seascape render` as a process that writes into its job's folder and waits."""
    script = tmp_path / "sleeper"
    # $6 is the folder after `-m seascape.cli render <scenario> -o`.
    script.write_text('#!/bin/sh\nmkdir -p "$6"\necho $$ > "$6/pid"\nexec sleep 60\n')
    script.chmod(0o755)
    monkeypatch.setattr(sys, "executable", str(script))
    return _state(tmp_path / "renders", **limits)


async def _started(state: server.State, name: str) -> int:
    pid = state.jobs[name].folder / "pid"
    async with asyncio.timeout(10):
        while not pid.exists() or not pid.read_text():
            await asyncio.sleep(0.01)
    return int(pid.read_text())


def test_cancel_stops_a_job_and_hands_the_gpu_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _sleeper(tmp_path, monkeypatch)

    async def call() -> None:
        async with Client(server.mcp(state)) as client:
            first, queued, last = [
                (await client.call_tool("render", {})).structured_content["job"]
                for _ in range(3)
            ]
            pid = await _started(state, first)

            cancelled = await client.call_tool("cancel", {"job": queued})
            assert cancelled.structured_content["status"] == "cancelled"
            assert state.jobs[first].state == "running"

            await client.call_tool("cancel", {"job": first})
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
            assert not state.jobs[first].folder.exists()
            reported = await client.call_tool("render_result", {"job": first})
            assert isinstance(report := reported.content[0], TextContent)
            assert json.loads(report.text) == {"status": "cancelled"}

            await _started(state, last)
            assert state.jobs[last].state == "running"
            await client.call_tool("cancel", {"job": last})
            again = await client.call_tool("cancel", {"job": last})
            assert again.is_error
            assert isinstance(refusal := again.content[0], TextContent)
            assert "cancelled" in refusal.text

    asyncio.run(call())
    assert not state.gpu.locked()


@pytest.mark.parametrize(("fmt", "refused"), [("jpg", False), ("exr", True)])
def test_a_job_is_refused_when_its_images_would_fill_the_disk(
    fmt: str, refused: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With room for one jpg job, beside one already queued, and the margin."""
    state = _state(tmp_path)
    queued = server.Job(tmp_path / "queued", "baseline", load_variants(BASELINE))
    state.jobs["queued"] = queued
    room = SimpleNamespace(
        free=2 * queued.size_bytes + state.settings.min_free_gb * 1e9
    )
    monkeypatch.setattr(shutil, "disk_usage", lambda _: room)

    async def run(*_: object) -> None:
        pass

    monkeypatch.setattr(server, "_run", run)
    result = _call(state, "render", overrides=[f'outputs.format = "{fmt}"'])
    assert result.is_error is refused
    assert len(state.jobs) == 1 + (not refused)


def test_a_job_past_its_deadline_fails_and_frees_the_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _sleeper(tmp_path, monkeypatch, max_run_s=0.5)

    async def call() -> None:
        async with Client(server.mcp(state)) as client:
            started = await client.call_tool("render", {})
            name = started.structured_content["job"]
            pid = await _started(state, name)
            task = state.jobs[name].task
            assert task is not None
            await task
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)

    asyncio.run(call())
    (job,) = state.jobs.values()
    assert (job.state, job.error) == ("failed", "ran past 0.5 s")
    assert not state.gpu.locked()


def test_a_job_waiting_for_the_gpu_keeps_its_whole_run_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _state(tmp_path, max_run_s=0.2)

    async def seascape(*_: str) -> None:
        await asyncio.sleep(0.05)

    monkeypatch.setattr(server, "_seascape", seascape)
    monkeypatch.setattr(server, "_finish", lambda _: None)

    async def run() -> str:
        job = server.Job(tmp_path / "a", "open-sea", load_variants(OPEN_SEA))
        async with state.gpu:
            task = asyncio.create_task(server._run(state, job, tmp_path, [], 1))
            # Longer than the run limit.
            await asyncio.sleep(0.3)
        await task
        return job.state

    assert asyncio.run(run()) == "done"


def test_a_full_queue_refuses_a_job(tmp_path: Path) -> None:
    state = _state(tmp_path, max_queued=1)
    state.jobs["queued"] = server.Job(
        tmp_path / "queued", "baseline", load_variants(BASELINE)
    )
    result = _call(state, "render")
    assert result.is_error
    assert "at most 1" in result.content[0].text
    assert len(state.jobs) == 1


def test_a_waiting_result_reports_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _sleeper(tmp_path, monkeypatch)
    monkeypatch.setattr(server, "WAIT_S", 0.3)
    monkeypatch.setattr(server, "PROGRESS_S", 0.1)
    reported: list[tuple[float, float | None]] = []

    async def progress(
        progress: float,
        total: float | None,
        message: str | None,  # noqa: ARG001  # `ProgressFnT` names it
    ) -> None:
        reported.append((progress, total))

    async def call() -> None:
        async with Client(server.mcp(state)) as client:
            started = await client.call_tool("render", {})
            name = started.structured_content["job"]
            await client.call_tool(
                "render_result", {"job": name}, progress_callback=progress
            )
            await client.call_tool("cancel", {"job": name})

    asyncio.run(call())
    assert len(reported) >= 2
    assert set(reported) == {(0, 2)}


def test_an_old_cancelled_job_is_forgotten(tmp_path: Path, state: server.State) -> None:
    scenes = load_variants(OPEN_SEA)

    async def forget() -> list[str]:
        done = asyncio.create_task(asyncio.sleep(0))
        await done
        old = server.Job(tmp_path / "old", "open-sea", scenes, state="cancelled")
        new = server.Job(tmp_path / "new", "open-sea", scenes, state="cancelled")
        old.created_s -= server.KEEP_S + 1
        old.task = new.task = done
        state.jobs.update(old=old, new=new)
        await server._prune(state)
        return list(state.jobs)

    assert asyncio.run(forget()) == ["new"]


def test_a_killed_job_leaves_no_temporary_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cancelled job's Blender is killed, which skips its own clean-up."""
    seen = {}

    class Killed:
        async def communicate(self) -> None:
            raise asyncio.CancelledError

        def kill(self) -> None:
            pass

        async def wait(self) -> int:
            return -9

    async def spawn(*_: str, env: dict[str, str], **__: Any) -> Killed:
        seen["tmp"] = Path(env["TMPDIR"])
        (seen["tmp"] / "blender_frames").mkdir()
        return Killed()

    monkeypatch.setattr(server.asyncio, "create_subprocess_exec", spawn)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(server._seascape("render"))
    assert not seen["tmp"].exists()


@pytest.mark.parametrize(
    ("secret", "path", "found"),
    [
        (None, "/mcp", True),
        (SECRET, "/mcp", False),
        (SECRET, "/mcp/wrong", False),
        (SECRET, f"/mcp/{SECRET}", True),
    ],
)
def test_a_secret_is_the_only_path_to_mcp(
    secret: str | None, path: str, found: bool, tmp_path: Path
) -> None:
    served = server.app(_state(tmp_path, mcp_secret=secret), "0.0.0.0")
    [status] = asyncio.run(_statuses(served, path, "localhost:8765"))
    assert (status != 404) is found


def test_the_public_urls_host_is_allowed(tmp_path: Path) -> None:
    state = _state(tmp_path, public_url="https://seascape.ngrok.app", mcp_secret=SECRET)
    served = server.app(state, "0.0.0.0")
    tunnel, other = asyncio.run(
        _statuses(served, f"/mcp/{SECRET}", "seascape.ngrok.app", "evil.example")
    )
    assert tunnel != 421
    assert other == 421


def test_a_public_url_needs_a_secret(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="SEASCAPE_MCP_SECRET"):
        _state(tmp_path, public_url="https://seascape.ngrok.app")


def test_listed_hosts_are_trimmed() -> None:
    settings = server.Settings(scenarios=SCENARIOS, allowed_hosts="a:*, b:* ,")  # type: ignore[arg-type]
    assert settings.allowed_hosts == ("a:*", "b:*")


def test_a_public_url_must_be_https(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="https"):
        _state(tmp_path, public_url="http://seascape.ngrok.app", mcp_secret=SECRET)


@pytest.mark.parametrize(
    "secret", ["short", "a/b" + "s" * 30, "{x}" + "s" * 30, "ä" * 40]
)
def test_a_secret_must_be_long_and_path_safe(secret: str, tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="32 or more"):
        _state(tmp_path, mcp_secret=secret)


@pytest.mark.parametrize(
    ("public", "url"),
    [
        (None, "http://gpu-box:8765/renders"),
        ("https://seascape.ngrok.app", "https://seascape.ngrok.app/renders"),
    ],
)
def test_files_are_linked_at_the_public_url(
    public: str | None, url: str, tmp_path: Path
) -> None:
    state = _state(tmp_path, public_url=public, mcp_secret=SECRET)
    ctx = SimpleNamespace(headers={"host": "gpu-box:8765"})
    assert server._url(state, ctx) == url  # ty: ignore[invalid-argument-type]


def test_a_public_url_with_a_port_is_allowed(tmp_path: Path) -> None:
    state = _state(tmp_path, public_url="https://box.example:8443", mcp_secret=SECRET)
    served = server.app(state, "0.0.0.0")
    [status] = asyncio.run(_statuses(served, f"/mcp/{SECRET}", "box.example:8443"))
    assert status != 421


def test_a_render_is_refused_before_anything_loads_when_the_queue_is_full(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _state(tmp_path, max_queued=1)
    state.jobs["busy"] = server.Job(tmp_path / "busy", "baseline", [], state="running")
    loads: list[object] = []
    monkeypatch.setattr(server, "_tried", lambda *a: loads.append(a))
    result = _call(state, "render")
    assert result.is_error
    assert "queued" in result.content[0].text
    assert not loads


def test_a_burst_of_renders_is_refused_before_it_loads_past_the_queue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _state(tmp_path, max_queued=2)
    loads: list[object] = []

    def slow_load(*args: object) -> tuple[list[object], float]:
        loads.append(args)
        time.sleep(0.3)
        return [], 1.0

    async def idle(*_: object) -> None:
        await asyncio.sleep(0)

    monkeypatch.setattr(server, "_tried", slow_load)
    monkeypatch.setattr(server, "_run", idle)

    async def burst() -> list[bool]:
        async with Client(server.mcp(state)) as client:
            calls = [client.call_tool("render", {}) for _ in range(5)]
            return [bool(r.is_error) for r in await asyncio.gather(*calls)]

    refused = asyncio.run(burst())
    assert refused.count(True) == 3
    assert len(loads) == 2
    assert len(state.jobs) == 2


def test_a_load_keeps_its_place_in_the_queue_after_its_caller_goes_away(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _state(tmp_path)
    started, release = threading.Event(), threading.Event()

    def slow_load(*_: object) -> None:
        started.set()
        release.wait(5)
        raise ValueError("bad scenario")

    monkeypatch.setattr(server, "_tried", slow_load)

    async def attempt() -> tuple[int, int]:
        async with Client(server.mcp(state)) as client:
            call = asyncio.create_task(client.call_tool("render", {}))
            await asyncio.to_thread(started.wait, 5)
            call.cancel()
            await asyncio.gather(call, return_exceptions=True)
            during = len(state.jobs)
            release.set()
            while state.jobs:
                await asyncio.sleep(0.01)
        return during, len(state.jobs)

    assert asyncio.run(attempt()) == (1, 0)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"variants": 101}, "at most 100 variants"),
        (
            {"overrides": ["x = '" + "x" * server.MAX_OVERRIDE_CHARS + "'"]},
            "characters",
        ),
        ({"overrides": ["outputs.samples.eo = 100000"]}, "samples per pixel"),
        ({"overrides": ["objects = [{ count = 100000 }]"]}, "hulls in all"),
        ({"overrides": ["seed = 1"] * (server.MAX_OVERRIDES + 1)}, "items"),
        ({"overrides": ["x = " + "[" * 497 + "]" * 497]}, "nests too deeply"),
        (
            {
                "overrides": [
                    "farms = [{ asset = 'turbine_15mw_floating', range_m = 1000,"
                    " bearing_deg = 0, rows = 3000, columns = 3000 }]"
                ]
            },
            "hulls",
        ),
        (
            {"overrides": ["outputs.duration_s = 60", "outputs.fps = 20000000"]},
            "images; a job writes at most",
        ),
    ],
)
def test_a_costly_request_is_refused(
    arguments: dict[str, Any], message: str, state: server.State
) -> None:
    result = _call(state, "render", **arguments)
    assert result.is_error
    assert message in result.content[0].text
    assert not state.jobs


def test_a_preset_key_cannot_leave_the_presets(state: server.State) -> None:
    result = _call(state, "render", overrides=['"../../.." = { preset = "pyproject" }'])
    assert result.is_error
    assert "not a preset directory" in result.content[0].text
    assert "tool.setuptools" not in result.content[0].text


def test_a_failing_gpu_probe_still_serves(
    state: server.State,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(*args: Any, **_: Any) -> None:
        raise subprocess.CalledProcessError(1, args[0], stderr="no GL\n")

    served: list[Any] = []
    monkeypatch.setattr(server.subprocess, "run", fail)
    monkeypatch.setattr(server.uvicorn, "run", lambda app, **_: served.append(app))
    server.serve(state, "127.0.0.1", 0)
    assert served
    assert "no GL" in capsys.readouterr().out


def test_a_running_job_is_not_deleted_for_its_age(
    tmp_path: Path, state: server.State
) -> None:
    folder = tmp_path / "busy"
    folder.mkdir()
    a_day_ago = time.time() - server.KEEP_S - 1
    os.utime(folder, (a_day_ago, a_day_ago))
    state.jobs["busy"] = server.Job(folder, "baseline", [], state="running")
    asyncio.run(server._prune(state))
    assert folder.exists()
    assert "busy" in state.jobs


def test_a_failed_render_tells_the_client_only_its_last_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def spawn(*_: Any, **__: Any) -> Any:
        async def communicate() -> tuple[bytes, bytes]:
            return b"", b"Traceback\n  File /srv/x.py\nValueError: bad scenario\n"

        return SimpleNamespace(communicate=communicate, returncode=1)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    with pytest.raises(RuntimeError) as error:
        asyncio.run(server._seascape("render"))
    assert str(error.value) == "ValueError: bad scenario"
