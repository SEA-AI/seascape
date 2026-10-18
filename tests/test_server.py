"""The render server, without Blender."""

import asyncio
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx2
import pytest
from mcp import Client

from seascape import server
from seascape.config import load_variants
from seascape.server import SCENARIOS


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


def test_a_limit_that_is_not_a_number_names_its_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SEASCAPE_PROBE", "4k")
    with pytest.raises(ValueError, match="SEASCAPE_PROBE"):
        server._limit("PROBE", 7.0)
