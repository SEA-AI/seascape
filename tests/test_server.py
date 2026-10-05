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
