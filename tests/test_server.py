"""The render server's checks, before a job starts. No Blender."""

import asyncio
import os
import time
from pathlib import Path
from typing import Any

import pytest
from mcp import Client

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
