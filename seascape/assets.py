"""Asset manifest: where a mesh comes from, and the download cache."""

import hashlib
import os
import shutil
import tomllib
import urllib.request
import uuid
from abc import abstractmethod
from collections import Counter
from functools import cache
from pathlib import Path, PurePosixPath
from typing import Annotated, ClassVar, Literal

from pydantic import ConfigDict, Field, TypeAdapter, field_validator

from seascape.model import Model

# XDG ignores a relative XDG_CACHE_HOME; honouring one puts meshes in the source tree.
_XDG = os.environ.get("XDG_CACHE_HOME", "")
CACHE = (Path(_XDG) if _XDG.startswith("/") else Path.home() / ".cache") / "seascape"
MANIFEST = Path(__file__).parent / "assets.toml"


SHA256 = r"^[0-9a-f]{64}$"


class File(Model):
    """A file a mesh reads beside itself, such as a glTF's buffer or a texture."""

    model_config = ConfigDict(frozen=True)

    url: str
    sha256: str = Field(pattern=SHA256)


class _Mesh(Model):
    model_config = ConfigDict(frozen=True)

    kind: str
    supercategory: ClassVar[str]

    description: str = Field(min_length=1)
    category: str = Field(min_length=1)
    url: str
    sha256: str = Field(pattern=SHA256)
    # By path relative to the mesh, as it names them.
    files: dict[str, File] = Field(default_factory=dict)
    # Required: a default of zero floats the hull and looks almost right.
    draught_m: float = Field(ge=0.0)
    # As `scene.measure` reads them.
    triangles: int = Field(gt=0)
    texture_px: tuple[int, ...]
    licence: str = Field(min_length=1)
    attribution: str = Field(min_length=1)

    @abstractmethod
    def scale(self, size: tuple[float, float, float]) -> float:
        """Metres per mesh unit, from the mesh's extent along x, y and z."""

    @property
    @abstractmethod
    def size(self) -> str: ...

    def summary(self) -> str:
        sizes = Counter(self.texture_px)
        textures = ", ".join(f"{n} x {px}px" for px, n in sizes.items()) or "none"
        return (
            f"{self.kind:<6} {self.size:>12}  {self.triangles:>9,} triangles  "
            f"textures: {textures}  {self.licence}\n"
            f"    {self.supercategory} / {self.category}: {self.description}"
        )

    @field_validator("files")
    @classmethod
    def _beside_the_mesh(cls, files: dict[str, File]) -> dict[str, File]:
        for path in map(PurePosixPath, files):
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"{path} is outside the mesh's folder")
        return files


class Hull(_Mesh):
    """A vessel, fitted bow to stern."""

    kind: Literal["hull"]
    supercategory: ClassVar[str] = "vessel"
    length_m: float = Field(gt=0.0)  # the mesh arrives in arbitrary units
    # Bearing of the mesh's bow as authored. The build turns it to +Y.
    bow_deg: float = 0.0

    def scale(self, size: tuple[float, float, float]) -> float:
        return self.length_m / size[1]

    @property
    def size(self) -> str:
        return f"{self.length_m:.0f} m long"


class Buoy(_Mesh):
    """A float with no bow, fitted keel to top."""

    kind: Literal["buoy"]
    supercategory: ClassVar[str] = "buoy"
    height_m: float = Field(gt=0.0)
    bow_deg: ClassVar[float] = 0.0

    def scale(self, size: tuple[float, float, float]) -> float:
        return self.height_m / size[2]

    @property
    def size(self) -> str:
        return f"{self.height_m:.1f} m tall"


class Debris(_Mesh):
    """Something adrift with no bow, fitted along its longest side."""

    kind: Literal["debris"]
    supercategory: ClassVar[str] = "debris"
    length_m: float = Field(gt=0.0)
    bow_deg: ClassVar[float] = 0.0

    def scale(self, size: tuple[float, float, float]) -> float:
        return self.length_m / max(size[0], size[1])

    @property
    def size(self) -> str:
        return f"{self.length_m:.2f} m long"


type Asset = Annotated[Hull | Buoy | Debris, Field(discriminator="kind")]


@cache
def manifest() -> dict[str, Asset]:
    with MANIFEST.open("rb") as handle:
        body = tomllib.load(handle)
    return TypeAdapter(dict[str, Asset]).validate_python(body)


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _verify(path: Path, sha256: str) -> None:
    got = digest(path)
    if got != sha256:
        raise ValueError(f"{path}: sha256 {got}, manifest says {sha256}")


def fetch(name: str) -> Path:
    """The cached mesh for `name`, and the files it reads, downloaded once."""
    asset = manifest()[name]
    for path, file in asset.files.items():
        _get(local(name).parent / path, file.url, file.sha256)
    return _get(local(name), asset.url, asset.sha256)


def cached(name: str) -> bool:
    """Whether `fetch` would download nothing."""
    files = [local(name).parent / path for path in manifest()[name].files]
    return all(path.exists() for path in [local(name), *files])


def local(name: str) -> Path:
    """Where `fetch` keeps the mesh for `name`."""
    asset = manifest()[name]
    if asset.files:
        # Its own folder, where the mesh finds the files by the paths it names.
        return CACHE / name / PurePosixPath(asset.url).name
    return cache_path(name, asset.url)


def cache_path(name: str, url: str) -> Path:
    return CACHE / f"{name}{PurePosixPath(url).suffix}"


def download(name: str, url: str, sha256: str) -> Path:
    """The cached file for `name`, downloaded once. Verified on every call."""
    return _get(cache_path(name, url), url, sha256)


def _get(path: Path, url: str, sha256: str) -> Path:
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
