"""The source checkout an editable install runs from, as PEP 610's direct_url.json
records it."""

from functools import cache
from importlib.metadata import distribution
from pathlib import Path


@cache
def root() -> Path | None:
    """None unless the install is editable."""
    origin = distribution("seascape").origin
    if origin is None or not getattr(getattr(origin, "dir_info", None), "editable", 0):
        return None
    return Path.from_uri(origin.url)
