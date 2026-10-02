"""Asset manifest: where a mesh comes from, and the download cache."""

import hashlib
import os
import shutil
import tomllib
import urllib.request
import uuid
from pathlib import Path, PurePosixPath

from pydantic import Field

from seascape.config import Model

# XDG ignores a relative XDG_CACHE_HOME; honouring one puts meshes in the source tree.
_XDG = os.environ.get("XDG_CACHE_HOME", "")
CACHE = (Path(_XDG) if _XDG.startswith("/") else Path.home() / ".cache") / "seascape"
MANIFEST = Path(__file__).parent / "assets.toml"


class Asset(Model):
    """One fetchable mesh."""

    url: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    length_m: float = Field(gt=0.0)  # bow to stern; the mesh arrives in arbitrary units
    # Required: a default of zero floats the hull and looks almost right.
    draught_m: float = Field(ge=0.0)
    # Bearing of the mesh's bow as authored. The build turns it to +Y.
    bow_deg: float = 0.0
    licence: str = Field(min_length=1)
    attribution: str = Field(min_length=1)


def manifest() -> dict[str, Asset]:
    with MANIFEST.open("rb") as handle:
        return {name: Asset(**body) for name, body in tomllib.load(handle).items()}


def _verify(path: Path, sha256: str) -> None:
    with path.open("rb") as handle:
        got = hashlib.file_digest(handle, "sha256").hexdigest()
    if got != sha256:
        raise ValueError(f"{path}: sha256 {got}, manifest says {sha256}")


def fetch(name: str) -> Path:
    asset = manifest()[name]
    return download(name, asset.url, asset.sha256)


def download(name: str, url: str, sha256: str) -> Path:
    """The cached file for `name`, downloaded once. Verified on every call."""
    path = CACHE / f"{name}{PurePosixPath(url).suffix}"
    if path.exists():
        _verify(path, sha256)
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    # A killed or corrupt transfer must never take the cache name, and concurrent
    # callers must not share a scratch file.
    part = path.with_name(f"{path.name}.{uuid.uuid4().hex}.part")
    # The default socket timeout is None, so a server that stops sending hangs the
    # build forever.
    with (
        urllib.request.urlopen(url, timeout=30) as response,
        part.open("wb") as out,
    ):
        shutil.copyfileobj(response, out)
    _verify(part, sha256)
    part.replace(path)
    return path
