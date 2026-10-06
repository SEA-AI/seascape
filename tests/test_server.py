"""The render server's checks, before a job starts. No Blender."""

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

import httpx2
import pytest
from mcp import Client
from PIL import Image

from seascape import server


def _call(tool: str, **arguments: Any) -> Any:
    async def call() -> Any:
        async with Client(server.server) as client:
            return await client.call_tool(tool, arguments)

    return asyncio.run(call())


@pytest.mark.parametrize(
    "line",
    [
        'objects = [{ preset = "/etc/hosts.toml" }]',
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
            [
                'rig.pods = [{ name = "bow", yaw_deg = 0.0, cameras = [{ kind = "eo", '
                "hfov_deg = 45.0, width_px = 7680, height_px = 4320 }] }]"
            ],
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


def test_a_finished_job_reports_its_files_and_a_preview_per_variant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server, "RENDERS", tmp_path)
    monkeypatch.setattr(server, "_jobs", {})

    async def render(*args: str) -> None:  # Blender's frames, without Blender
        (job,) = server._jobs.values()
        for scene, folder in zip(job.scenes, job.folders, strict=True):
            folder.mkdir(parents=True, exist_ok=True)
            for mount in scene.rig.mounts:
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


@pytest.mark.parametrize(("secret", "path", "found"), [
    (None, "/mcp", True),
    ("s3cret", "/mcp", False),
    ("s3cret", "/mcp/wrong", False),
    ("s3cret", "/mcp/s3cret", True),
])  # fmt: skip
def test_a_secret_is_the_only_path_to_mcp(
    secret: str | None,
    path: str,
    found: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "RENDERS", tmp_path)
    if secret:
        monkeypatch.setenv("SEASCAPE_MCP_SECRET", secret)
    else:
        monkeypatch.delenv("SEASCAPE_MCP_SECRET", raising=False)
    served = server.app("0.0.0.0")

    async def post() -> int:
        async with served.router.lifespan_context(served):
            transport = httpx2.ASGITransport(app=served)
            async with httpx2.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                response = await client.post(path, json={})
                return response.status_code

    assert (asyncio.run(post()) != 404) is found
