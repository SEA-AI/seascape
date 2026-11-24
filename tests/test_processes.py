"""Functions run side by side, each in a fresh Python."""

import importlib.util
import os
import sys
import types
from collections.abc import Callable
from pathlib import Path

import pytest

from seascape import processes
from tests.jobs import Kind, job


def _gone(pid_file: Path) -> bool:
    try:
        os.kill(int(pid_file.read_text()), 0)
    except ProcessLookupError:
        return True
    return False


def _run(
    tmp_path: Path, kinds: dict[str, Kind], tick: Callable[[], None] = lambda: None
) -> list[str]:
    jobs = {key: (kind, tmp_path / f"{key}.pid") for key, kind in kinds.items()}
    logs = {key: tmp_path / f"{key}.log" for key in kinds}
    return processes.at_once(job, jobs, logs, tick)


def test_the_jobs_load_blender_in_one_temporary_folder_gone_after_them(
    tmp_path: Path,
) -> None:
    assert _run(tmp_path, {"top": "blender", "chase": "blender"}) == []

    folders = {
        (tmp_path / f"{key}.log").read_text().splitlines()[-1]
        for key in ("top", "chase")
    }
    [tmp] = folders
    assert not Path(tmp).exists()


@pytest.mark.parametrize("kind", ["fail", "crash"])
def test_a_job_that_fails_or_crashes_stops_the_rest(tmp_path: Path, kind: Kind) -> None:
    assert _run(tmp_path, {"ir": kind, "eo": "wait"}) == ["ir"]

    assert _gone(tmp_path / "eo.pid")


def test_a_failed_job_logs_its_traceback(tmp_path: Path) -> None:
    _run(tmp_path, {"ir": "fail", "eo": "wait"})

    assert "ValueError: no GPU" in (tmp_path / "ir.log").read_text()


def test_an_interrupt_kills_every_job(tmp_path: Path) -> None:
    pid_file = tmp_path / "eo.pid"

    def interrupt() -> None:
        if pid_file.exists():
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        _run(tmp_path, {"eo": "wait"}, interrupt)

    assert _gone(pid_file)


def test_a_job_from_a_module_run_with_m_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `python -m seascape.cli` names the module of each function it holds `__main__`.
    main = types.ModuleType("__main__")
    main.__spec__ = importlib.util.find_spec("tests.jobs")
    monkeypatch.setitem(sys.modules, "__main__", main)
    monkeypatch.setattr(job, "__module__", "__main__")

    assert _run(tmp_path, {"top": "blender"}) == []
