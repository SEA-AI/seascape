"""Asset manifest: where a mesh comes from, and the download cache."""

import hashlib
import math
import os
import re
import shutil
import tomllib
import urllib.request
import uuid
from abc import abstractmethod
from collections import Counter
from functools import cache
from pathlib import Path, PurePosixPath
from typing import Annotated, ClassVar, Literal

from pydantic import ConfigDict, Field, TypeAdapter, field_validator, model_validator

from seascape.model import Model
from seascape.physics.thermal import ZoneName

# XDG ignores a relative XDG_CACHE_HOME; honouring one puts meshes in the source tree.
_XDG = os.environ.get("XDG_CACHE_HOME", "")
CACHE = (Path(_XDG) if _XDG.startswith("/") else Path.home() / ".cache") / "seascape"
MANIFEST = Path(__file__).parent / "assets.toml"


SHA256 = r"^[0-9a-f]{64}$"
ISLAND_ID = r"^(0|[1-9][0-9]*)$"


class File(Model):
    """A file a mesh reads beside itself, such as a glTF's buffer or a texture."""

    model_config = ConfigDict(frozen=True)

    url: str
    sha256: str = Field(pattern=SHA256)


Range = Annotated[tuple[float, float], Field(description="Lowest and highest, 0-1.")]


class Rule(Model):
    """A zone for the islands of one material whose means of its Principled BSDF
    inputs all fall in the given ranges. Scoped to a material, since a map one slot
    gets right another may not."""

    model_config = ConfigDict(frozen=True)

    material: str
    zone: ZoneName
    metallic: Range | None = None
    roughness: Range | None = None
    alpha: Range | None = None

    @model_validator(mode="after")
    def _bounded(self) -> "Rule":
        if not self.ranges:
            raise ValueError("a rule needs a range")
        if any(not 0.0 <= lo <= hi <= 1.0 for lo, hi in self.ranges.values()):
            raise ValueError("a range runs low to high within 0-1")
        return self

    @property
    def ranges(self) -> dict[str, tuple[float, float]]:
        channels = ("metallic", "roughness", "alpha")
        return {k: v for k in channels if (v := getattr(self, k)) is not None}


class Zones(Model):
    """Each material's zone, and the islands that differ from their material's: those
    listed, then those the first matching rule names.

    An island is the faces of one material in one object that touch at a vertex, and at
    a UV where the material is textured, numbered by object name, then first face, as
    `zones.parts` imports them.
    """

    model_config = ConfigDict(frozen=True)

    materials: dict[str, ZoneName] = Field(min_length=1)
    # The import's island count and `Parts.island_digest`: an importer that adds,
    # drops or reorders faces renumbers the islands, and fails the check.
    island_count: int | None = Field(default=None, ge=1)
    island_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{16}$")
    islands: dict[int, ZoneName] = Field(default_factory=dict)
    rules: list[Rule] = Field(default_factory=list)

    @field_validator("islands", mode="before")
    @classmethod
    def _canonical(cls, islands: object) -> object:
        # "00" and "+1" would coerce to ids that read as another key.
        if isinstance(islands, dict) and any(
            not isinstance(k, str) or not re.fullmatch(ISLAND_ID, k) for k in islands
        ):
            raise ValueError(f"island ids are written {ISLAND_ID}")
        return islands

    @model_validator(mode="after")
    def _counted(self) -> "Zones":
        fingerprint = (self.island_count, self.island_digest)
        if self.islands and None in fingerprint:
            raise ValueError("islands need island_count and island_digest")
        if not self.islands and fingerprint != (None, None):
            raise ValueError("island_count and island_digest need islands")
        if self.island_count is not None and any(
            not 0 <= i < self.island_count for i in self.islands
        ):
            raise ValueError(f"an island id is outside 0-{self.island_count - 1}")
        if unknown := {r.material for r in self.rules} - set(self.materials):
            raise ValueError(f"a rule names no material of these: {sorted(unknown)}")
        return self


class _Mesh(Model):
    model_config = ConfigDict(frozen=True)

    kind: str
    supercategory: ClassVar[str]

    description: str = Field(min_length=1)
    category: str = Field(min_length=1)
    # One per category, so labels.json files merge.
    category_id: int = Field(ge=1)
    url: str
    sha256: str = Field(pattern=SHA256)
    # By path relative to the mesh, as it names them.
    files: dict[str, File] = Field(default_factory=dict)
    # Required: a default of zero floats the hull and looks almost right.
    draught_m: float = Field(ge=0.0)
    # As `meshes.measure` reads them.
    triangles: int = Field(gt=0)
    texture_px: tuple[int, ...]
    licence: str = Field(min_length=1)
    attribution: str = Field(min_length=1)
    zones: Zones | None = None

    @abstractmethod
    def scale(self, size: tuple[float, float, float]) -> float:
        """Metres per mesh unit, from the mesh's extent along x, y and z."""

    def centre(
        self, low: tuple[float, float, float], high: tuple[float, float, float]
    ) -> tuple[float, float]:
        """Where the anchor goes in x and y, from the fitted box's corners."""
        return (low[0] + high[0]) / 2, (low[1] + high[1]) / 2

    @property
    @abstractmethod
    def size(self) -> str: ...

    @property
    def textures(self) -> str:
        sizes = Counter(self.texture_px)
        return ", ".join(f"{n} x {px} px" for px, n in sizes.items()) or "-"

    def row(self) -> tuple[str, ...]:
        return (
            self.kind,
            self.category,
            self.size,
            f"{self.triangles:,}",
            self.textures,
            self.licence,
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
        return f"{self.length_m:.3g} m long"


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
        return f"{self.height_m:.3g} m tall"


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
        return f"{self.length_m:.3g} m long"


class Rotor(Model):
    """A turbine's hub and blades, and the speed its controller turns them at."""

    model_config = ConfigDict(frozen=True)

    # An empty at the hub whose +Y is the shaft, pointing upwind; hub and blades under
    # it. Unique across the manifest's files, so an import never renames it.
    node: str = Field(min_length=1)
    diameter_m: float = Field(gt=0.0)
    hub_height_m: float = Field(gt=0.0)  # above the waterline
    tip_speed_ratio: float = Field(gt=0.0)
    min_rpm: float = Field(ge=0.0)
    rated_rpm: float = Field(gt=0.0)
    cut_in_mps: float = Field(ge=0.0)
    cut_out_mps: float = Field(gt=0.0)

    @model_validator(mode="after")
    def _ordered(self) -> "Rotor":
        if self.min_rpm > self.rated_rpm or self.cut_in_mps >= self.cut_out_mps:
            raise ValueError(f"{self.node}: min_rpm > rated_rpm or cut_in >= cut_out")
        return self

    def rpm(self, hub_mps: float) -> float:
        """At the hub's wind: parked outside cut-in and cut-out, else the speed that
        holds the tip-speed ratio, between minimum and rated."""
        if not self.cut_in_mps <= hub_mps <= self.cut_out_mps:
            return 0.0
        tracking = 60.0 * self.tip_speed_ratio * hub_mps / (math.pi * self.diameter_m)
        return min(max(tracking, self.min_rpm), self.rated_rpm)


class Turbine(_Mesh):
    """A wind turbine on its foundation, fitted keel to blade tip, rotor to +Y.

    Stands where a farm puts it, yawed into the wind; only its rotor turns.
    """

    kind: Literal["turbine"]
    supercategory: ClassVar[str] = "structure"
    height_m: float = Field(gt=0.0)
    bow_deg: float = 0.0
    rotor: Rotor

    def scale(self, size: tuple[float, float, float]) -> float:
        return self.height_m / size[2]

    def centre(
        self, low: tuple[float, float, float], high: tuple[float, float, float]
    ) -> tuple[float, float]:
        # The authored origin, on the tower's axis: the rotor's overhang would shift
        # the box's centre.
        return 0.0, 0.0

    @property
    def size(self) -> str:
        return f"{self.rotor.diameter_m:.3g} m rotor"


type Asset = Annotated[Hull | Buoy | Debris | Turbine, Field(discriminator="kind")]


@cache
def manifest() -> dict[str, Asset]:
    with MANIFEST.open("rb") as handle:
        body = tomllib.load(handle)
    return TypeAdapter(dict[str, Asset]).validate_python(body)


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


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
    """The cached file for `name`, downloaded again whenever it misses `sha256`."""
    return _get(cache_path(name, url), url, sha256)


def _get(path: Path, url: str, sha256: str) -> Path:
    # A cached file that misses the digest is stale or corrupt: fetch it again, and
    # keep it until a verified copy replaces it.
    if path.exists() and digest(path) == sha256:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    # A killed or corrupt transfer must never take the cache name, and concurrent
    # callers must not share a scratch file.
    part = path.with_name(f"{path.name}.{uuid.uuid4().hex}.part")
    # The default socket timeout is None, so a server that stops sending hangs the
    # build forever.
    try:
        with (
            urllib.request.urlopen(url, timeout=30) as response,
            part.open("wb") as out,
        ):
            shutil.copyfileobj(response, out)
        if (got := digest(part)) != sha256:
            raise ValueError(f"{url}: sha256 {got}, manifest says {sha256}")
        part.replace(path)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    return path
