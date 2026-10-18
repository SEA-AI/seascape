"""The render server, without Blender."""

import asyncio
import os
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
    job = server.Job(tmp_path, scenes)
    assert job.folders == [tmp_path / str(first), tmp_path / str(first + 2)]
    assert [s.seed for s in server._written(job, SCENARIOS / "open-sea.toml", [])] == [
        first,
        first + 2,
    ]


@pytest.mark.parametrize(
    ("allowed", "host", "refused"),
    [
        ("", "evil.example", False),
        ("hector:*", "hector:8765", False),
        ("hector:*", "evil.example", True),
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
