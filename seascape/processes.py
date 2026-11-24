"""Functions run side by side, each in a fresh Python of its own."""

import os
import pickle
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any


def at_once[K](
    work: Callable[..., object],
    jobs: Mapping[K, tuple[Any, ...]],
    logs: Mapping[K, Path],
    tick: Callable[[], None] = lambda: None,
) -> list[K]:
    """Run `work` on each job's arguments at once, each in a fresh Python, so each
    loads a Blender of its own.

    The first job to fail or crash stops the rest, and an error here or Ctrl-C kills
    them; every one has ended when this returns or raises.

    Args:
        work: A module-level function, which a fresh Python imports by name.
        jobs: Each job's arguments, by its key; picklable.
        logs: Where each job's output and traceback go, by its key.
        tick: Called about once a second while the jobs run.

    Returns:
        The keys of the jobs that failed.
    """
    # Limitation: a SIGTERM or SIGKILL to this process leaves them running; Linux's
    # PR_SET_PDEATHSIG in each child would end them.
    runs: dict[K, subprocess.Popen[bytes]] = {}
    failed: list[K] = []
    # A killed Blender leaves its temporary files, the foam memory's frames among them;
    # this folder is deleted when the jobs end.
    with tempfile.TemporaryDirectory(prefix="seascape-jobs-") as tmp:
        try:
            for key, args in jobs.items():
                with logs[key].open("w") as log, tempfile.TemporaryFile() as job:
                    pickle.dump((work, args), job)
                    job.seek(0)
                    runs[key] = subprocess.Popen(
                        [sys.executable, "-m", __name__],
                        stdin=job,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env={**os.environ, "TMPDIR": tmp},
                    )
            pending = dict(runs)
            while pending and not failed:
                done = {k: run for k, run in pending.items() if run.poll() is not None}
                failed = [k for k, run in done.items() if run.returncode != 0]
                for key in done:
                    del pending[key]
                if pending and not done:
                    tick()
                    time.sleep(1)
        finally:
            for run in runs.values():
                if run.poll() is None:
                    run.kill()
                    run.wait()
    return failed


if __name__ == "__main__":
    # A job from `at_once`.
    work, args = pickle.load(sys.stdin.buffer)
    work(*args)
