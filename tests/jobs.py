"""A job for `processes.at_once`, in a module a fresh Python imports by its name:
pytest imports a test module under its file's stem.
"""

import os
import time
from pathlib import Path
from typing import Literal

import bpy

type Kind = Literal["blender", "wait", "fail", "crash"]


def job(kind: Kind, out: Path) -> None:
    """Write this process's id to `out`, then by `kind`: load Blender and print its
    temporary folder, wait to be killed, or, once every job beside it has written
    its id, raise or abort.
    """
    # Named once whole, so a job that waits on it never reads it half written.
    written = out.with_suffix(".tmp")
    written.write_text(str(os.getpid()))
    written.replace(out)
    if kind in ("fail", "crash"):
        while len(list(out.parent.glob("*.pid"))) < 2:
            time.sleep(0.05)
    match kind:
        case "blender":
            bpy.ops.wm.read_factory_settings(use_empty=True)
            print(os.environ["TMPDIR"])
        case "wait":
            time.sleep(60)
        case "fail":
            raise ValueError("no GPU")
        case "crash":
            os.abort()
