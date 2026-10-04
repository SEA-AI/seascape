"""Scenario config: pydantic models, and a TOML loader that never guesses.

A scenario is a TOML file. Two forms of reuse, both resolved here:

- `extends = "baseline.toml"` at the top level, for a variant that is a diff.
- `preset = "twin_pod"` inside a block, next to the overrides it applies to.

Rules:

1. `preset` is a key inside its block, so there are no precedence rules between
   parents.
2. Syntax decides what a name is. A bare name is a preset shipped under `cfg/`,
   anything with `/` or ending in `.toml` is a path relative to the including file.
   Never try one form and fall back to the other.
3. Tables merge, everything else replaces. A list is replaced whole.
"""

import math
import tomllib
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Any, Literal, NamedTuple

from pydantic import (
    AfterValidator,
    Field,
    WithJsonSchema,
    model_validator,
)

from seascape import assets, lwir, skies, waves
from seascape.model import Model

CFG_DIR = Path(__file__).parent / "cfg"

# A camera's kind is the band it sees in.
type Band = Literal["eo", "ir"]

type ImageFormat = Literal["exr", "png", "jpg"]


class Camera(Model):
    """One camera in a pod: its band, its aim relative to the pod, its image."""

    kind: Band = Field(description="The band it sees in: eo visible, ir LWIR.")
    yaw_deg: float = Field(
        default=0.0, description="Relative to the pod axis, positive to starboard."
    )
    pitch_deg: float = Field(
        default=0.0,
        gt=-90.0,
        lt=90.0,
        description="Relative to the pod, negative is down.",
    )
    name: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9_-]+$",
        description="Its image's filename. Derived from its place in the pod when "
        "absent.",
    )
    hfov_deg: float = Field(
        gt=0.0,
        lt=180.0,
        description="Measured across the image width, even in portrait.",
    )
    width_px: int = Field(gt=0, description="Image width.")
    height_px: int = Field(gt=0, description="Image height.")


class Pod(Model):
    """One enclosure: several cameras behind one yaw and one mount point.

    Pods a beam apart overlap in angle before they overlap in space, which leaves a
    blind wedge over the bow; coincident cameras would hide it.
    """

    # Path text would write outside the output directory.
    name: str = Field(
        pattern=r"^[A-Za-z0-9_-]+$",
        description="Prefixes the filenames of its unnamed cameras.",
    )
    yaw_deg: float = Field(
        description="Pod axis relative to the bow, positive to starboard."
    )
    offset_x_m: float = Field(
        default=0.0, description="From the centreline, positive to starboard."
    )
    offset_y_m: float = Field(
        default=0.0, description="From midships, positive forward."
    )
    cameras: list[Camera] = Field(min_length=1, description="The cameras in this pod.")


class Mount(NamedTuple):
    pod: Pod
    camera: Camera
    index: int

    @property
    def name(self) -> str:
        """Never an angle: re-aiming a rig would invalidate every filename it wrote."""
        return self.camera.name or f"{self.pod.name}_{self.camera.kind}_{self.index}"

    @property
    def nominal_bearing_deg(self) -> float:
        """Not the achieved boresight: the rig's pitch sits between the two yaws, so
        an off-axis camera points elsewhere. `scene.boresight_deg` measures the
        built camera."""
        return self.pod.yaw_deg + self.camera.yaw_deg


class Rig(Model):
    """The pods on the ownship."""

    height_m: float = Field(gt=0.0, description="Pod height above the waterline.")
    pitch_deg: float = Field(
        default=0.0,
        gt=-90.0,
        lt=90.0,
        description="Every pod, about its own transverse axis. Negative is down.",
    )
    # Depth precision goes as far / near, so larger is better. The bound is a lens's
    # clearance from its own structure.
    near_clip_m: float = Field(
        default=5.0, gt=0.0, description="Anything nearer a camera is not rendered."
    )
    pods: list[Pod] = Field(
        min_length=1, description="The camera enclosures on the ownship."
    )

    @property
    def mounts(self) -> list[Mount]:
        """Counted, not searched: two cameras with the same fields are equal to
        pydantic, so `index()` would give both the same number."""
        mounts: list[Mount] = []
        for pod in self.pods:
            seen: dict[Band, int] = {}
            for camera in pod.cameras:
                index = seen.get(camera.kind, 0)
                seen[camera.kind] = index + 1
                mounts.append(Mount(pod, camera, index))
        return mounts

    @model_validator(mode="after")
    def _names_are_unique(self) -> "Rig":
        """Cameras are parented by pod name, so two pods of one name send every
        camera to the last one built, with nothing raised."""
        pods = [pod.name for pod in self.pods]
        if len(set(pods)) != len(pods):
            raise ValueError(f"two pods share a name: {sorted(pods)}")
        names = [mount.name for mount in self.mounts]
        if len(set(names)) != len(names):
            raise ValueError(f"two cameras share a name: {sorted(names)}")
        return self


class Swell(Model):
    """Waves from a distant storm, whatever the local wind."""

    height_m: float = Field(gt=0.0, description="Significant wave height.")
    period_s: float = Field(gt=0.0, description="Seconds between crests.")
    from_deg: float = Field(
        default=0.0,
        description="Where it comes from, clockwise from the ownship's bow.",
    )


class Sea(Model):
    """The temperature bound is the span of the shipped optical-constant table. `lwir`
    clamps to it; here it is an error.
    """

    t_sea_k: float = Field(
        default=lwir.T_SEA_K,
        ge=271.0,
        le=311.0,
        description="Sea surface temperature. IR only. Unset, the atmosphere's own.",
    )
    wind_speed_mps: float = Field(
        default=7.0,
        ge=0.0,
        description="Measured 10 m above the sea. Sets the wind sea.",
    )
    wind_from_deg: float = Field(
        default=0.0,
        description="Where the wind blows from, clockwise from the ownship's bow.",
    )
    swell: Swell | None = Field(default=None, description="On top of the wind's sea.")
    # Opt-in: surfactants come from the water.
    slick_cover: float = Field(
        default=0.0,
        ge=0.0,
        lt=1.0,
        description="Fraction of the sea under slicks, gathered in windrows. None in "
        "calm air, where slick and clean water are one.",
    )
    # The atmosphere bends a ray down, so the sea curves at R / (1 - k). 0.13 is the
    # standard survey value for average air (0.13-0.16 usual). At k = 1 the effective
    # radius is infinite.
    refraction_k: float = Field(
        default=0.13,
        ge=0.0,
        lt=1.0,
        description="Coefficient of terrestrial refraction; 0 is none.",
    )


class Sky(Model):
    """Blender's Sky Texture or a photographed sky in EO, and the downwelling radiance
    the sea reflects in IR.

    Haze is `aerosol_density` for the EO sky, the node's own, and `visibility_km` for
    the air between the camera and what it sees, in both bands. In LWIR, `atmosphere`
    adds its water vapour, shapes the sky, and gives `t_air_k` unless it is set.
    """

    sun_elevation_deg: float | None = Field(
        default=30.0,
        ge=-90.0,
        le=90.0,
        description="Above the horizon; negative is below it. With `hdri`, set from "
        "the photo, and None where its sun shows no disc.",
    )
    sun_bearing_deg: float = Field(
        default=0.0, description="Clockwise from the ownship's bow."
    )
    # Steady state, where absorbed sun balances convection and re-radiation:
    #   dT = a E / (h + 4 eps sigma T^3)
    # a = 0.30 for light marine paint, E = 1000 W m^-2 for a clear sky, and
    # h = 10.45 - v + 10 sqrt(v) = 30 W m^-2 K^-1 at 7 m/s. Weakly held: a dark hull
    # absorbs three times what a light one does.
    solar_gain_k: float = Field(
        default=8.5,
        ge=0.0,
        description="How much warmer a sunlit surface is than a shaded one. IR only.",
    )
    aerosol_density: float = Field(
        default=1.0,
        ge=0.0,
        le=10.0,
        description="Haze, as the Sky Texture's own parameter. EO only.",
    )
    # Judgement, after Adams: 42 km, inside the open ocean's measured spread. OPAC's
    # maritime aerosols at 550 nm and 80% humidity (Hess, Koepke & Schult, BAMS 79(5)
    # 831, 1998), plus sea-level Rayleigh's 0.012 km^-1 (the MODTRAN 2/3 report,
    # eq. 26): clean, 0.090 km^-1, is 38 km; tropical, 0.043 km^-1, is 72 km. The
    # ocean's mean optical depth, 0.11 at 500 nm, gives 34 km in OPAC's clean profile
    # (Smirnov et al., JGR 2009, doi:10.1029/2008JD011257).
    visibility_km: float | None = Field(
        default=42.0,
        gt=0.0,
        description="Meteorological range at 550 nm; None is no aerosol.",
    )
    atmosphere: lwir.Atmosphere = Field(
        default=lwir.ATMOSPHERE,
        description="LOWTRAN 7's model atmosphere, or the North Sea's, for the LWIR "
        "sky and air. EO ignores it. subarctic_winter's horizon partly sees space, so "
        "its horizon sky reads warmer than LOWTRAN's.",
    )
    t_air_k: float = Field(
        default=lwir.SURFACE_AIR_K[lwir.ATMOSPHERE],
        ge=250.0,
        le=320.0,
        description="Scales the IR sky. EO ignores it. Unset, the atmosphere's own.",
    )

    hdri: (
        Annotated[
            str, WithJsonSchema({"type": "string", "enum": sorted(skies.library())})
        ]
        | None
    ) = Field(
        default=None,
        description="A photographed sky from seascape/skies.toml, in place of the Sky "
        "Texture, which sets `sun_elevation_deg` and ignores `aerosol_density`. LWIR "
        "keeps its own sky but takes the photo's sun, turned to `sun_bearing_deg`.",
    )

    @model_validator(mode="before")
    @classmethod
    def _sun_follows_the_photo(cls, data: Any) -> Any:
        # Anything but a name is left for the field's own validation to refuse.
        if not isinstance(data, dict) or not isinstance(data.get("hdri"), str):
            return data
        photos = skies.library()
        if data["hdri"] not in photos:
            raise ValueError(f"no sky {data['hdri']!r} in {sorted(photos)}")
        elevation = photos[data["hdri"]].sun_elevation_deg
        # Warned, not refused: `extends` and `--set` cannot remove a key. A dumped
        # scenario holds the photo's own values and is not warned about.
        unread = {
            "sun_elevation_deg": elevation,
            "aerosol_density": cls.model_fields["aerosol_density"].default,
        }
        for key, value in unread.items():
            if key in data and data[key] != value:
                warnings.warn(f"the hdri sets the sky; {key} is ignored", stacklevel=2)
        data = {**data, "sun_elevation_deg": elevation}
        if elevation is None:
            # No disc, no direct beam: nothing warms a sunlit side over a shaded one.
            data.setdefault("solar_gain_k", 0.0)
        return data

    @model_validator(mode="after")
    def _a_sky_texture_has_a_sun(self) -> "Sky":
        if self.hdri is None and self.sun_elevation_deg is None:
            raise ValueError(
                "sun_elevation_deg is None only for an hdri without a disc"
            )
        return self

    @model_validator(mode="before")
    @classmethod
    def _air_follows_the_atmosphere(cls, data: Any) -> Any:
        if isinstance(data, dict) and "t_air_k" not in data:
            air_k = lwir.SURFACE_AIR_K.get(data.get("atmosphere", lwir.ATMOSPHERE))
            if air_k is not None:  # an unknown profile fails its own validation
                data = {**data, "t_air_k": air_k}
        return data

    @property
    def extinction_per_m(self) -> float:
        """Koschmieder's law: over `visibility_km` a dark target keeps 2% of its
        contrast against the horizon sky."""
        if self.visibility_km is None:
            return 0.0
        return math.log(1 / 0.02) / (self.visibility_km * 1000)


def _in_the_manifest(name: str) -> str:
    known = assets.manifest()
    if name not in known:
        raise ValueError(f"no asset {name!r} in {sorted(known)}")
    return name


# The names go in the schema too, so an editor completes them.
AssetName = Annotated[
    str,
    AfterValidator(_in_the_manifest),
    WithJsonSchema({"type": "string", "enum": sorted(assets.manifest())}),
]
_ASSET = "An asset name from the manifest: `seascape assets` lists them."
_T_HULL = "Shaded hull temperature. IR only."
_HEADING = "Where its bow points, clockwise from the ownship's bow."
_RANGE = "Horizontal, from the ownship's origin."
_SPEED = "Along its heading."
_DRIFT = "A figure-eight about its pose."


class Drift(Model):
    """A hull at single anchor fishtails: across its heading once a period, along it
    twice. Starts at its pose; the heading holds."""

    sway_m: float = Field(ge=0.0, description="Peak, across the heading.")
    surge_m: float = Field(ge=0.0, description="Peak, along the heading.")
    period_s: float = Field(gt=0.0, description="One full figure-eight.")


class Orbit(Model):
    """Round the ownship's origin clockwise from the hull's bearing, at its range, bow
    along the circle.

    Identical hulls share the lap evenly, so a loop need only last `period_s / count`.
    """

    period_s: float = Field(gt=0.0, description="One full lap.")
    count: int = Field(default=1, gt=0, description="Hulls spaced evenly on it.")


class Object(Model):
    """Something to detect."""

    asset: AssetName = Field(description=_ASSET)
    range_m: float = Field(gt=0.0, description=_RANGE)
    bearing_deg: float = Field(description="Clockwise from the ownship's bow.")
    heading_deg: float = Field(default=0.0, description=_HEADING)
    speed_mps: float = Field(default=0.0, ge=0.0, description=_SPEED)
    drift: Drift | None = Field(default=None, description=_DRIFT)
    orbit: Orbit | None = Field(default=None, description="Round the ownship.")
    t_k: float = Field(default=293.0, ge=250.0, le=400.0, description=_T_HULL)

    @model_validator(mode="after")
    def _an_orbit_steers(self) -> "Object":
        steered = {"heading_deg", "speed_mps", "drift"} & self.model_fields_set
        if self.orbit is not None and steered:
            raise ValueError(
                f"{self.asset} orbits, which sets its course: drop {sorted(steered)}"
            )
        return self


class Swing(Model):
    """A sinusoid about the mean attitude, starting at the mean."""

    amplitude_deg: float = Field(ge=0.0, lt=90.0, description="Peak, either way.")
    period_s: float = Field(gt=0.0, description="One full cycle.")


class Heave(Model):
    """A sinusoid about the waterline, starting at it."""

    amplitude_m: float = Field(ge=0.0, description="Peak, either way.")
    period_s: float = Field(gt=0.0, description="One full cycle.")


class Ownship(Model):
    """The vessel the rig is bolted to, rolling and pitching about its origin at the
    waterline. Without an asset it is the attitude alone."""

    asset: AssetName | None = Field(default=None, description=_ASSET)
    t_k: float = Field(default=296.0, ge=250.0, le=400.0, description=_T_HULL)
    roll_deg: float = Field(
        default=0.0, gt=-90.0, lt=90.0, description="Positive is starboard down."
    )
    pitch_deg: float = Field(
        default=0.0, gt=-90.0, lt=90.0, description="Positive is bow up."
    )
    # ponytail: one sine per axis, a sum over a wave spectrum when irregular motion
    # matters.
    roll: Swing | None = Field(default=None, description="About roll_deg.")
    pitch: Swing | None = Field(default=None, description="About pitch_deg.")
    heave: Heave | None = Field(default=None, description="About the waterline.")


class Targets(Model):
    """A ring of vessels at one range, spread over a span of bearings.

    Placed in the world, so nothing guarantees a camera sees one.
    """

    asset: AssetName = Field(description=_ASSET)
    count: int = Field(gt=0, description="How many.")
    range_m: float = Field(gt=0.0, description=_RANGE)
    bearing_deg: tuple[float, float] = Field(
        description="First and last, clockwise from the ownship's bow. Both ends get "
        "a target."
    )
    # Spread so aspect varies between targets.
    heading_deg: tuple[float, float] = Field(
        default=(0.0, 315.0),
        description="First and last, spread evenly, clockwise from the ownship's bow.",
    )
    speed_mps: float = Field(default=0.0, ge=0.0, description=_SPEED)
    drift: Drift | None = Field(default=None, description=_DRIFT)
    t_k: float = Field(default=293.0, ge=250.0, le=400.0, description=_T_HULL)

    def _spread(self, span: tuple[float, float], i: int) -> float:
        low, high = span
        return low + (high - low) * i / max(self.count - 1, 1)

    def poses(self) -> list[tuple[float, float]]:
        """Bearing and heading per target, in degrees."""
        return [
            (self._spread(self.bearing_deg, i), self._spread(self.heading_deg, i))
            for i in range(self.count)
        ]


class Samples(Model):
    """Cycles samples per band.

    A model so an override merges field by field; a dict would replace it whole
    and drop the band left out.
    """

    eo: int = Field(default=16, gt=0, description="Per pixel, for EO frames.")
    ir: int = Field(default=64, gt=0, description="Per pixel, for IR frames.")


class Outputs(Model):
    """What a render writes.

    Every camera of a listed band is rendered. An EO png or jpg is 8-bit, as a camera
    with auto-exposure takes it. An LWIR png is 16-bit, a hundredth of a kelvin per
    count; an LWIR jpg is 8-bit grey, auto-contrasted as a thermal camera does. An exr
    keeps the radiance, in W m^-2 sr^-1.
    """

    # uniqueItems for editors validating against the schema; `_bands_are_distinct`
    # enforces it.
    bands: tuple[Band, ...] = Field(
        default=("eo", "ir"),
        min_length=1,
        json_schema_extra={"uniqueItems": True},
        description="Bands to render; one with no camera is skipped.",
    )
    samples: Samples = Field(default_factory=Samples)
    format: ImageFormat = Field(
        default="jpg",
        description="jpg to look at, png lossless or LWIR in kelvin, exr the radiance.",
    )
    # The compositor computes in float32, normal from 2^-126 to 2^127 (IEEE 754).
    exposure_compensation_ev: float = Field(
        default=0.0,
        ge=-126.0,
        le=126.0,
        description="Stops over auto-exposure. Applies to 8-bit EO only.",
    )
    duration_s: float = Field(default=0.0, ge=0.0, description="0 is a still.")
    fps: int = Field(default=10, gt=0, description="Frames per second of a sequence.")
    loop: bool = Field(
        default=False,
        description="Repeats seamlessly: each period rounds to a whole fraction of "
        "duration_s.",
    )

    @property
    def times_s(self) -> list[float]:
        return [f / self.fps for f in range(max(1, round(self.duration_s * self.fps)))]

    @property
    def span_s(self) -> float:
        """From frame 0 to the frame after the last, which a loop makes frame 0."""
        return len(self.times_s) / self.fps

    def period_s(self, period_s: float) -> float:
        """In a loop, the nearest whole fraction of the span, so every cycle closes."""
        if not self.loop:
            return period_s
        return self.span_s / max(1, round(self.span_s / period_s))

    @model_validator(mode="after")
    def _bands_are_distinct(self) -> "Outputs":
        """A repeat renders the same cameras twice, onto the same files."""
        if len(set(self.bands)) != len(self.bands):
            raise ValueError(f"a band is listed twice: {self.bands}")
        return self


# A judgement.
LOOP_SNAP_TOLERANCE = 0.05


class Scenario(Model):
    """One scene: the rig, the world around it, and what a render writes."""

    seed: int = Field(default=0, description="Seeds every random draw.")
    rig: Rig
    ownship: Ownship = Field(default_factory=Ownship)
    targets: Targets | None = Field(
        default=None, description="A ring of identical vessels."
    )
    sea: Sea = Field(default_factory=Sea)
    sky: Sky = Field(default_factory=Sky)
    objects: list[Object] = Field(
        default_factory=list, description="Vessels placed one by one."
    )
    outputs: Outputs = Field(default_factory=Outputs)

    @model_validator(mode="before")
    @classmethod
    def _sea_follows_the_atmosphere(cls, data: Any) -> Any:
        """The sea's own field cannot see the sky's, so the scenario fills it."""
        if not isinstance(data, dict) or not isinstance(data.get("sea", {}), dict):
            return data
        sea = data.get("sea", {})
        atmosphere = data.get("sky", {}).get("atmosphere", lwir.ATMOSPHERE)
        if "t_sea_k" not in sea and atmosphere in lwir.SURFACE_SEA_K:
            data = {**data, "sea": {**sea, "t_sea_k": lwir.SURFACE_SEA_K[atmosphere]}}
        return data

    @model_validator(mode="after")
    def _a_loop_can_close(self) -> "Scenario":
        if not self.outputs.loop:
            return self
        if self.outputs.duration_s == 0.0:
            raise ValueError("a loop needs outputs.duration_s > 0")
        for spec in (*self.objects, self.targets):
            if spec is not None and spec.speed_mps > 0.0:
                raise ValueError(
                    f"{spec.asset} has speed_mps > 0, and a straight run never comes "
                    "back to close a loop: give it a drift instead"
                )
        motions = [
            ("ownship.roll", self.ownship.roll),
            ("ownship.pitch", self.ownship.pitch),
            ("ownship.heave", self.ownship.heave),
            *((f"{spec.asset} drift", spec.drift) for spec in self.objects),
            ("targets.drift", self.targets.drift if self.targets else None),
            ("sea.swell", self.sea.swell),
        ]
        periods = [(n, m.period_s) for n, m in motions if m is not None] + [
            (f"{spec.asset} orbit", spec.orbit.period_s / spec.orbit.count)
            for spec in self.objects
            if spec.orbit is not None
        ]
        span_s = self.outputs.span_s
        for name, period_s in periods:
            # Rounding it to the span would speed the motion up.
            if period_s > span_s:
                raise ValueError(
                    f"a {span_s} s loop is shorter than {name}'s {period_s} s "
                    "period: make outputs.duration_s at least that"
                )
        snap = self.outputs.period_s
        error = waves.snap_error(self.sea.wind_speed_mps, snap)
        if self.sea.swell is not None:
            period_s = self.sea.swell.period_s
            error = max(error, abs(snap(period_s) - period_s) / snap(period_s))
        if error > LOOP_SNAP_TOLERANCE:
            warnings.warn(
                f"a {span_s} s loop shifts the waves' frequencies by {error:.1%}; a "
                "longer outputs.duration_s shifts them less",
                stacklevel=2,
            )
        return self


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """`over` wins. Two tables merge; anything else replaces outright."""
    out = dict(base)
    for key, value in over.items():
        current = out.get(key)
        out[key] = (
            _merge(current, value)
            if isinstance(current, dict) and isinstance(value, dict)
            else value
        )
    return out


def _include_path(name: Any, base: Path, block: str | None = None) -> Path:
    """Resolve rule 2. `block` is the preset directory; None allows only a path."""
    if not isinstance(name, str):
        raise TypeError(f"include must be a string, got {name!r}")
    if "/" in name or name.endswith(".toml"):
        return base / name
    if block is None:
        raise ValueError(f"{name!r} must be a path: it needs a '/' or a '.toml' suffix")
    return CFG_DIR / block / f"{name}.toml"


def _expand(node: Any, block: str | None, base: Path, chain: tuple[Path, ...]) -> Any:
    """Resolve every `preset` key in the tree, innermost first."""
    if isinstance(node, list):
        return [_expand(item, block, base, chain) for item in node]
    if not isinstance(node, dict):
        return node
    out = {key: _expand(value, key, base, chain) for key, value in node.items()}
    if (name := out.pop("preset", None)) is None:
        return out
    return _merge(_read(_include_path(name, base, block), chain), out)


def _read(path: Path, chain: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.resolve()
    if path in chain:
        cycle = " -> ".join(str(p) for p in (*chain, path))
        raise ValueError(f"circular include: {cycle}")
    chain = (*chain, path)

    with path.open("rb") as handle:  # TOML is UTF-8 by spec, so never read_text
        data = tomllib.load(handle)

    # This file's presets resolve before the parent merges in, or an inherited value
    # would outrank a preset this file names explicitly.
    data = _expand(data, None, path.parent, chain)
    if (parent := data.pop("extends", None)) is not None:
        data = _merge(_read(_include_path(parent, path.parent), chain), data)
    return data


def load(path: str | Path, overrides: Iterable[str] = ()) -> Scenario:
    """Read a scenario TOML, resolving `extends` and `preset`, and validate it.

    Each override is a TOML assignment merged over the file, `rig.pitch_deg = -5`,
    and resolved as if it were a line in it.
    """
    path = Path(path)
    data = _read(path)
    for assignment in overrides:
        data = _merge(data, _expand(tomllib.loads(assignment), None, path.parent, ()))
    return Scenario.model_validate(data)
