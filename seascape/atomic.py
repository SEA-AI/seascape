"""Files that appear whole or not at all."""

import uuid
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def replacing(path: Path) -> Generator[Path]:
    """A scratch file to write, moved over `path` when the block ends cleanly.

    A killed or failed write never leaves a truncated file under `path`, and
    concurrent writers do not share a scratch file. The scratch file keeps `path`'s
    suffix for writers that choose a format by it.

    Args:
        path: The file to replace.
    """
    part = path.with_name(f"{path.name}.{uuid.uuid4().hex}.part{path.suffix}")
    try:
        yield part
        part.replace(path)
    finally:
        part.unlink(missing_ok=True)
