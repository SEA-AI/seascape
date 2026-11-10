"""The render server, without Blender."""

import asyncio
import json
import os
import re
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
from PIL import ExifTags, Image
from pydantic import ValidationError

from seascape.config import Scenario, json_schema, load
from seascape.server import Settings, State, jobs, tools, transport
from seascape.server.jobs import Job, JobState
from tests.scenarios import BASELINE, OPEN_SEA, SCENARIOS, load_variants

SECRET = "s" * 32


def _state(renders: Path, **limits: Any) -> State:
    return State(Settings(scenarios=SCENARIOS, **limits), renders)


async def _statuses(
    served: Any, path: str, *hosts: str, auth: str | None = None
) -> list[int]:
    async with served.router.lifespan_context(served):
        transport = httpx2.ASGITransport(app=served)
        async with httpx2.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            return [
                (
                    await client.post(
                        path,
                        json={},
                        headers={"host": h} | ({"authorization": auth} if auth else {}),
                    )
                ).status_code
                for h in hosts
            ]


@pytest.fixture
def state(tmp_path: Path) -> State:
    return _state(tmp_path)


def _call(state: State, tool: str, **arguments: Any) -> Any:
    async def call() -> Any:
        async with Client(transport.mcp(state)) as client:
            return await client.call_tool(tool, arguments)

    return asyncio.run(call())


def test_only_the_tools_that_change_state_ask_a_client_for_approval(
    tmp_path: Path,
) -> None:
    async def listed() -> dict[str, bool]:
        async with Client(transport.mcp(_state(tmp_path))) as client:
            tools = (await client.list_tools()).tools
            return {
                t.name: bool(t.annotations and t.annotations.read_only_hint)
                for t in tools
            }

    assert asyncio.run(listed()) == {
        "catalog": True,
        "scenario": True,
        "schema": True,
        "jobs": True,
        "get_job": True,
        "preview": True,
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
def test_a_bad_override_is_refused_before_any_job(line: str, state: State) -> None:
    result = _call(state, "render", overrides=[line])
    assert result.is_error
    assert not state.jobs


def test_an_unknown_scenario_is_refused(state: State) -> None:
    result = _call(state, "render", scenario="../baseline")
    assert result.is_error
    assert not state.jobs


def test_the_catalog_names_what_render_takes(state: State) -> None:
    listed = _call(state, "catalog").structured_content

    assert listed["scenarios"] == list(state.scenarios())
    assert set(listed["schema_sections"]) == set(json_schema()["properties"])
    assert all(listed["schema_sections"].values())
    assert "schema" not in listed


def test_a_scenario_is_its_file(state: State) -> None:
    result = _call(state, "scenario", name="baseline")
    file = SCENARIOS / "baseline.toml"
    assert result.content[0].text == file.read_text(encoding="utf-8")
    assert _call(state, "scenario", name="../baseline").is_error


def test_a_schema_section_holds_what_it_reaches_and_no_more(state: State) -> None:
    section = _call(state, "schema", section="sea").structured_content

    assert set(section["properties"]) == {"sea"}
    assert section["$defs"]
    full = json_schema()
    assert len(json.dumps(section)) < len(json.dumps(full)) / 2
    refs = set(re.findall(r'"#/\$defs/(\w+)"', json.dumps(section)))
    assert refs <= set(section["$defs"])
    assert _call(state, "schema", section="nope").is_error


def test_a_full_disk_refuses_a_job(tmp_path: Path) -> None:
    state = _state(tmp_path, min_free_gb=2.0**50)
    result = _call(state, "render")
    assert result.is_error
    assert "GB free" in result.content[0].text
    assert not state.jobs


def test_jobs_reports_the_free_disk(state: State) -> None:
    listed = _call(state, "jobs").structured_content
    assert listed["jobs"] == []
    assert listed["free_gb"] > 0


def test_a_job_older_than_a_day_is_deleted(tmp_path: Path, state: State) -> None:
    old, busy, new = tmp_path / "old", tmp_path / "busy", tmp_path / "new"
    for folder in (old, busy, new):
        folder.mkdir()
        (folder / "labels.json").touch()
    (tmp_path / "old.zip").touch()
    state.jobs["busy"] = Job(busy, "baseline", [], state="running")
    a_day_ago = time.time() - jobs.KEEP_S - 1
    for path in (old, busy, tmp_path / "old.zip"):
        os.utime(path, (a_day_ago, a_day_ago))
    asyncio.run(jobs.prune(state))
    # A running job's folder may sit unchanged past the cutoff.
    assert sorted(path.name for path in tmp_path.iterdir()) == ["busy", "new"]
    assert "busy" in state.jobs


def test_an_idle_server_keeps_deleting_what_expires(
    tmp_path: Path, state: State, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(jobs, "SWEEP_S", 0.05)
    first, second = tmp_path / "first", tmp_path / "second"
    for folder in (first, second):
        folder.mkdir()
    a_day_ago = time.time() - jobs.KEEP_S - 1
    os.utime(first, (a_day_ago, a_day_ago))
    application = transport.app(state, "localhost")

    async def until(gone: Path) -> None:
        for _ in range(100):
            if not gone.exists():
                return
            await asyncio.sleep(0.05)

    async def served() -> None:
        async with application.router.lifespan_context(application):
            await until(first)
            # Expires after the startup sweep, so only a later one can delete it.
            os.utime(second, (a_day_ago, a_day_ago))
            await until(second)

    asyncio.run(served())
    assert not first.exists()
    assert not second.exists()


def test_a_folder_lives_as_long_as_its_archive(tmp_path: Path, state: State) -> None:
    folder, archive = tmp_path / "job", tmp_path / "job.zip"
    folder.mkdir()
    archive.touch()
    a_day_ago = time.time() - jobs.KEEP_S - 1
    os.utime(folder, (a_day_ago, a_day_ago))
    asyncio.run(jobs.prune(state))
    assert folder.exists()
    assert archive.exists()


def test_a_variant_job_finds_the_seeds_render_kept(tmp_path: Path) -> None:
    scenes = load_variants(OPEN_SEA, count=2)
    first = scenes[0].seed
    # `render` skipped first + 1, whose hulls met.
    for seed in (first, first + 2):
        (tmp_path / str(seed)).mkdir()
    (tmp_path / "labels.json").touch()
    job = Job(tmp_path, "open-sea", scenes)
    assert job.folders == [tmp_path / str(first), tmp_path / str(first + 2)]
    assert [s.seed for s in jobs._written(job, OPEN_SEA, [])] == [
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
        ("gpu-box:* , ", "gpu-box:8765", False),
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
    served = transport.app(_state(tmp_path), "0.0.0.0")

    [status] = asyncio.run(_statuses(served, "/mcp", host))
    assert (status == 421) is refused


@pytest.mark.parametrize(
    ("error", "said"),
    [
        (subprocess.TimeoutExpired("probe", 60), "no answer"),
        (subprocess.CalledProcessError(1, "probe", stderr="no GL\n"), "no GL"),
    ],
)
def test_a_gpu_probe_that_fails_still_serves(
    error: Exception,
    said: str,
    state: State,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(*_: Any, **__: Any) -> None:
        raise error

    served: list[Any] = []
    monkeypatch.setattr(subprocess, "run", fail)
    monkeypatch.setattr(transport.uvicorn, "run", lambda app, **_: served.append(app))
    transport.serve(state, "127.0.0.1", 0)
    assert served
    assert said in capsys.readouterr().out


def test_jobs_lists_a_finished_job_whose_archive_is_gone(
    tmp_path: Path, state: State
) -> None:
    scenes = load_variants(OPEN_SEA)
    state.jobs["gone"] = Job(tmp_path / "gone", "open-sea", scenes, state="done")
    (listed,) = _call(state, "jobs").structured_content["jobs"]
    assert listed["status"] == "done"
    assert "archive" not in listed


@pytest.mark.parametrize(("value", "expected"), [("", 10.0), ("2.5", 2.5)])
def test_a_limit_reads_the_environment(
    value: str, expected: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SEASCAPE_MIN_FREE_GB", value)
    assert Settings(scenarios=SCENARIOS).min_free_gb == expected


@pytest.mark.parametrize("value", ["4k", "inf", "0"])
def test_a_limit_that_is_not_a_positive_number_names_its_field(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SEASCAPE_MIN_FREE_GB", value)
    with pytest.raises(ValidationError, match="min_free_gb"):
        Settings(scenarios=SCENARIOS)


def test_a_finishing_job_is_not_ahead_of_the_next(tmp_path: Path) -> None:
    scenes = load_variants(OPEN_SEA)
    job = Job(tmp_path, "open-sea", scenes, state="finishing")
    assert not job.waiting


@pytest.fixture
def blenderless(state: State, monkeypatch: pytest.MonkeyPatch) -> None:
    """`render` writes each variant's frames as blank pictures."""

    async def render(*_args: str) -> None:
        (job,) = state.jobs.values()
        for scene in job.scenes:
            folder = job.folder / str(scene.seed)  # as `render --variants` names it
            folder.mkdir(parents=True, exist_ok=True)
            for mount in scene.mounts:
                frame = folder / f"{mount.name}.{scene.outputs.format}"
                Image.new("RGB", (640, 480)).save(frame)

    monkeypatch.setattr(jobs, "_seascape", render)


def _rendered(state: State, *then: tuple[str, dict[str, Any]]) -> list[Any]:
    """Render 2 variants, then call each of `then` on the job."""

    async def call() -> list[Any]:
        async with Client(transport.mcp(state)) as client:
            started = await client.call_tool("render", {"variants": 2})
            job = started.structured_content["job"]
            return [
                await client.call_tool(tool, {"job": job, **arguments})
                for tool, arguments in then
            ]

    return asyncio.run(call())


@pytest.mark.usefixtures("blenderless")
def test_a_finished_job_reports_its_archive_and_a_sheet_per_variant(
    state: State,
) -> None:
    (result,) = _rendered(state, ("get_job", {}))
    report = json.loads(result.content[0].text)
    assert report["status"] == "done", report
    assert report["archive"].endswith(".zip")
    (job,) = state.jobs.values()
    assert report["cameras"] == [m.name for m in job.scenes[0].mounts]
    assert len(report["sheets"]) == len(job.scenes) == 2
    assert [(c.type, c.mime_type) for c in result.content[1:]] == [
        ("image", "image/jpeg")
    ] * 2
    description = Image.open(job.folders[1] / jobs.SHEET).getexif()[
        ExifTags.Base.ImageDescription
    ]
    assert json.loads(description) == {
        "scenario": "baseline",
        "seed": job.scenes[1].seed,
        "overrides": [],
    }
    assert report["sheets_not_shown"] == 0


@pytest.mark.usefixtures("blenderless")
def test_preview_gives_a_variants_sheet_or_one_camera(state: State) -> None:
    mounts = load_variants(BASELINE)[0].mounts
    camera = mounts[0].name
    _, sheet, frame, *refused = _rendered(
        state,
        ("get_job", {}),
        ("preview", {"variant": 1}),
        ("preview", {"camera": camera}),
        ("preview", {"variant": 2}),
        ("preview", {"camera": "nope"}),
    )
    assert [r.content[0].mime_type for r in (sheet, frame)] == ["image/jpeg"] * 2
    assert "variant 2 is not in 0 to 1" in refused[0].content[0].text
    assert "no camera 'nope'" in refused[1].content[0].text


@pytest.mark.usefixtures("blenderless")
def test_a_sheet_that_cannot_be_made_leaves_the_render_done(
    state: State, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*_args: Any) -> None:
        raise FileNotFoundError("/server/path/frame.png")

    monkeypatch.setattr(jobs.montage, "sheet", refuse)
    report, sheet = _rendered(state, ("get_job", {}), ("preview", {}))
    report = json.loads(report.content[0].text)
    assert report["status"] == "done"
    assert report["sheets"] == []
    assert sheet.is_error
    assert "could not be read" in sheet.content[0].text
    assert "/server/path" not in sheet.content[0].text


def _preview_of(state: State, scene: Scenario, job_state: JobState) -> Any:
    """What `preview` answers for a stored job in `job_state`, with no render."""

    async def call() -> Any:
        job = Job(state.renders / "stored", "open-sea", [scene], state=job_state)
        job.task = asyncio.create_task(asyncio.sleep(0))
        state.jobs["stored"] = job
        async with Client(transport.mcp(state)) as client:
            return await client.call_tool("preview", {"job": "stored"})

    return asyncio.run(call())


def test_a_long_request_leaves_its_overrides_out_of_the_exif(tmp_path: Path) -> None:
    job = Job(tmp_path, "open-sea", [], overrides=['"' * 1000] * 50)
    exif = jobs.provenance(job, 7)
    described = json.loads(exif[ExifTags.Base.ImageDescription])
    assert described == {
        "scenario": "open-sea",
        "seed": 7,
        "overrides_omitted": True,
    }
    assert len(exif.tobytes()) < 65_535


def test_preview_refuses_an_exr_job(state: State) -> None:
    refused = _preview_of(state, load(OPEN_SEA, ['outputs.format = "exr"']), "done")
    assert refused.is_error
    assert "exr" in refused.content[0].text


def test_preview_refuses_a_job_that_is_not_done(state: State) -> None:
    refused = _preview_of(state, load(OPEN_SEA), "running")
    assert refused.is_error
    assert "is running" in refused.content[0].text


def _sleeper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **limits: Any) -> State:
    """`seascape render` as a process that writes into its job's folder and waits."""
    script = tmp_path / "sleeper"
    # $6 is the folder after `-m seascape.cli render <scenario> -o`.
    script.write_text('#!/bin/sh\nmkdir -p "$6"\necho $$ > "$6/pid"\nexec sleep 60\n')
    script.chmod(0o755)
    monkeypatch.setattr(sys, "executable", str(script))
    return _state(tmp_path / "renders", **limits)


async def _started(state: State, name: str) -> int:
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
        async with Client(transport.mcp(state)) as client:
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
            reported = await client.call_tool("get_job", {"job": first})
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
    queued = Job(tmp_path / "queued", "baseline", load_variants(BASELINE))
    state.jobs["queued"] = queued
    room = SimpleNamespace(
        free=2 * queued.size_bytes + state.settings.min_free_gb * 1e9
    )
    monkeypatch.setattr(shutil, "disk_usage", lambda _: room)

    async def run(*_: object) -> None:
        pass

    monkeypatch.setattr(tools, "run", run)
    result = _call(state, "render", overrides=[f'outputs.format = "{fmt}"'])
    assert result.is_error is refused
    assert len(state.jobs) == 1 + (not refused)


def test_a_job_past_its_deadline_fails_and_frees_the_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _sleeper(tmp_path, monkeypatch, max_run_s=0.5)

    async def call() -> None:
        async with Client(transport.mcp(state)) as client:
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

    monkeypatch.setattr(jobs, "_seascape", seascape)
    monkeypatch.setattr(jobs, "_finish", lambda _: None)

    async def run() -> str:
        job = Job(tmp_path / "a", "open-sea", load_variants(OPEN_SEA))
        async with state.gpu:
            task = asyncio.create_task(jobs.run(state, job, tmp_path, [], 1))
            # Longer than the run limit.
            await asyncio.sleep(0.3)
        await task
        return job.state

    assert asyncio.run(run()) == "done"


def test_a_waiting_result_reports_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _sleeper(tmp_path, monkeypatch)
    monkeypatch.setattr(tools, "WAIT_S", 0.3)
    monkeypatch.setattr(tools, "PROGRESS_S", 0.1)
    reported: list[tuple[float, float | None]] = []

    async def progress(
        progress: float,
        total: float | None,
        message: str | None,  # noqa: ARG001  # `ProgressFnT` names it
    ) -> None:
        reported.append((progress, total))

    async def call() -> None:
        async with Client(transport.mcp(state)) as client:
            started = await client.call_tool("render", {})
            name = started.structured_content["job"]
            await client.call_tool("get_job", {"job": name}, progress_callback=progress)
            await client.call_tool("cancel", {"job": name})

    asyncio.run(call())
    assert len(reported) >= 2
    assert set(reported) == {(0, 2)}


def test_an_old_cancelled_job_is_forgotten(tmp_path: Path, state: State) -> None:
    scenes = load_variants(OPEN_SEA)

    async def forget() -> list[str]:
        done = asyncio.create_task(asyncio.sleep(0))
        await done
        old = Job(tmp_path / "old", "open-sea", scenes, state="cancelled")
        new = Job(tmp_path / "new", "open-sea", scenes, state="cancelled")
        old.created_s -= jobs.KEEP_S + 1
        old.task = new.task = done
        state.jobs.update(old=old, new=new)
        await jobs.prune(state)
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

    monkeypatch.setattr(jobs.asyncio, "create_subprocess_exec", spawn)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(jobs._seascape("render"))
    assert not seen["tmp"].exists()


@pytest.mark.parametrize(
    ("secret", "auth", "refused"),
    [
        (None, None, False),
        (SECRET, None, True),
        (SECRET, "Bearer wrong", True),
        (SECRET, SECRET, True),
        (SECRET, f"Bearer {SECRET}", False),
    ],
)
def test_a_secret_is_the_bearer_token_mcp_needs(
    secret: str | None, auth: str | None, refused: bool, tmp_path: Path
) -> None:
    served = transport.app(_state(tmp_path, mcp_secret=secret), "0.0.0.0")
    [status] = asyncio.run(_statuses(served, "/mcp", "localhost:8765", auth=auth))
    assert (status == 401) is refused


def test_renders_need_no_token(tmp_path: Path) -> None:
    (tmp_path / "job").mkdir()
    (tmp_path / "job" / "a.txt").write_text("hi")
    served = transport.app(_state(tmp_path, mcp_secret=SECRET), "0.0.0.0")

    async def fetch() -> int:
        transport_ = httpx2.ASGITransport(app=served)
        async with httpx2.AsyncClient(
            transport=transport_, base_url="http://test"
        ) as c:
            return (
                await c.get("/renders/job/a.txt", headers={"host": "localhost"})
            ).status_code

    assert asyncio.run(fetch()) == 200


@pytest.mark.parametrize(
    ("public", "host", "refused"),
    [
        ("https://seascape.ngrok.app", "seascape.ngrok.app", False),
        ("https://seascape.ngrok.app", "evil.example", True),
        ("https://box.example:8443", "box.example:8443", False),
    ],
)
def test_only_the_public_urls_host_is_allowed(
    public: str, host: str, refused: bool, tmp_path: Path
) -> None:
    state = _state(tmp_path, public_url=public, mcp_secret=SECRET)
    served = transport.app(state, "0.0.0.0")
    [status] = asyncio.run(_statuses(served, "/mcp", host, auth=f"Bearer {SECRET}"))
    assert (status == 421) is refused


def test_a_public_url_needs_a_secret(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="SEASCAPE_MCP_SECRET"):
        _state(tmp_path, public_url="https://seascape.ngrok.app")


def test_a_public_url_must_be_https(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="https"):
        _state(tmp_path, public_url="http://seascape.ngrok.app", mcp_secret=SECRET)


@pytest.mark.parametrize(
    "secret", ["short", "s" * 31, "ä" * 40, "has space" + "s" * 30]
)
def test_a_secret_must_be_long_and_header_safe(secret: str, tmp_path: Path) -> None:
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
    assert tools._url(state, ctx) == url  # ty: ignore[invalid-argument-type]


def test_a_render_is_refused_before_anything_loads_when_the_queue_is_full(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _state(tmp_path, max_queued=1)
    state.jobs["busy"] = Job(tmp_path / "busy", "baseline", [], state="running")
    loads: list[object] = []
    monkeypatch.setattr(jobs, "_tried", lambda *a: loads.append(a))
    result = _call(state, "render")
    assert result.is_error
    assert "queued" in result.content[0].text
    assert not loads
    assert len(state.jobs) == 1


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

    monkeypatch.setattr(jobs, "_tried", slow_load)
    monkeypatch.setattr(tools, "run", idle)

    async def burst() -> list[bool]:
        async with Client(transport.mcp(state)) as client:
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

    monkeypatch.setattr(jobs, "_tried", slow_load)

    async def attempt() -> tuple[int, int]:
        async with Client(transport.mcp(state)) as client:
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
        ({"overrides": ["outputs.duration_s = 61.0"]}, "a clip lasts"),
        (
            {
                "overrides": [
                    "rigs.bow.cameras.eo = { width_px = 7680, height_px = 4320 }"
                ]
            },
            "pixels",
        ),
        (
            {
                "overrides": ["outputs.duration_s = 60.0", "outputs.fps = 100"],
                "variants": 2,
            },
            "images",
        ),
        # The first seeds' clips fit; a seed tried after one whose hulls meet does not.
        (
            {
                "overrides": ["outputs.duration_s = { uniform = [0.0, 70.0] }"],
                "variants": 2,
            },
            "a clip lasts",
        ),
        # The first seeds' images fit; two seeds render may try instead do not.
        (
            {
                "overrides": [
                    "outputs.duration_s = { uniform = [0.0, 60.0] }",
                    "outputs.fps = 50",
                ],
                "variants": 2,
            },
            "images",
        ),
        (
            {"overrides": ["x = '" + "x" * tools.MAX_OVERRIDE_CHARS + "'"]},
            "characters",
        ),
        ({"overrides": ["outputs.samples.eo = 100000"]}, "samples per pixel"),
        ({"overrides": ["objects = [{ count = 100000 }]"]}, "hulls in all"),
        ({"overrides": ["seed = 1"] * (tools.MAX_OVERRIDES + 1)}, "items"),
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
    arguments: dict[str, Any], message: str, state: State
) -> None:
    result = _call(state, "render", **arguments)
    assert result.is_error
    assert message in result.content[0].text
    assert not state.jobs


def test_a_preset_key_cannot_leave_the_presets(state: State) -> None:
    result = _call(state, "render", overrides=['"../../.." = { preset = "pyproject" }'])
    assert result.is_error
    assert "not a preset directory" in result.content[0].text
    assert "tool.setuptools" not in result.content[0].text


def test_a_failed_render_tells_the_client_only_its_last_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def spawn(*_: Any, **__: Any) -> Any:
        async def communicate() -> tuple[bytes, bytes]:
            return b"", b"Traceback\n  File /srv/x.py\nValueError: bad scenario\n"

        return SimpleNamespace(communicate=communicate, returncode=1)

    monkeypatch.setattr(jobs.asyncio, "create_subprocess_exec", spawn)
    with pytest.raises(RuntimeError) as error:
        asyncio.run(jobs._seascape("render"))
    assert str(error.value) == "ValueError: bad scenario"


def test_render_says_a_list_override_replaces_the_scenarios_list(
    state: State,
) -> None:
    async def described() -> str:
        async with Client(transport.mcp(state)) as client:
            tools = {tool.name: tool for tool in (await client.list_tools()).tools}
            return tools["render"].description or ""

    assert "replaces the whole list" in " ".join(asyncio.run(described()).split())
