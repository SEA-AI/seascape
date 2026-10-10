"""Scenario config: pydantic models, and a TOML loader that never guesses.

A scenario is a TOML file. Two forms of reuse, both resolved here:

- `extends = "baseline.toml"` at the top level, for a variant that is a diff.
- `preset = "pod"` inside a block, next to the overrides it applies to.

Rules:

1. `preset` is a key inside its block, so there are no precedence rules between
   parents.
2. Syntax decides what a name is. A bare name is a preset shipped under `cfg/`,
   anything with `/` or ending in `.toml` is a path relative to the including file.
   Never try one form and fall back to the other.
3. Tables merge, everything else replaces. A list, or a table naming a preset, is
   replaced whole.

Any field can instead be a draw, `{ uniform = [lo, hi] }` or `{ choice = [...] }`,
resolved after both forms of reuse and before validation.
"""

import math
import tomllib
import warnings
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Annotated, Any, Literal, NamedTuple

import numpy as np
from pydantic import (
    AfterValidator,
    Field,
    WithJsonSchema,
    model_validator,
)

from seascape import assets, lwir, skies, waves
from seascape.model import Model

CFG_DIR = Path(__file__).parent / "cfg"

type Band = Literal["eo", "ir"]

type ImageFormat = Literal["exr", "png", "jpg"]


def substream(seed: int, name: str) -> np.random.Generator:
    """A named substream, so adding a component cannot perturb an existing one."""
    return np.random.default_rng([seed, *name.encode()])


class Uniform(Model):
    """A float drawn evenly between two bounds."""

    uniform: tuple[float, float] = Field(description="The bounds.")

    def draw(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(*self.uniform))


class Integer(Model):
    """A whole number drawn evenly between two bounds."""

    integer: tuple[int, int] = Field(description="The bounds, both included.")

    def draw(self, rng: np.random.Generator) -> int:
        return int(rng.integers(*self.integer, endpoint=True))


class Choice(Model):
    """One of several values, each as likely unless weighted."""

    choice: list[Any] = Field(min_length=1, description="The values.")
    weights: list[Annotated[float, Field(ge=0.0)]] | None = Field(
        default=None,
        description="How likely each value is, relative to the others: [8, 2] is 80 "
        "and 20 percent.",
    )

    @model_validator(mode="after")
    def _a_weight_per_value(self) -> "Choice":
        if self.weights is None:
            return self
        if len(self.weights) != len(self.choice):
            raise ValueError(
                f"{len(self.weights)} weights for {len(self.choice)} values"
            )
        total = sum(self.weights)
        if not math.isfinite(total):
            raise ValueError(f"the weights sum to {total}")
        if not total:
            raise ValueError("the weights sum to 0")
        return self

    def draw(self, rng: np.random.Generator) -> Any:
        if self.weights is None:
            return self.choice[rng.integers(len(self.choice))]
        p = np.array(self.weights) / sum(self.weights)
        return self.choice[rng.choice(len(self.choice), p=p)]


_DRAWS: dict[str, type[Uniform | Integer | Choice]] = {
    "uniform": Uniform,
    "integer": Integer,
    "choice": Choice,
}


# Path text would write outside the output directory.
type Name = Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]+$")]


class Camera(Model):
    """One camera on a rig."""

    band: Band = Field(description="The band it sees in: eo visible, ir LWIR.")
    yaw_deg: float = Field(
        default=0.0, description="Relative to the rig's axis, positive to starboard."
    )
    pitch_deg: float = Field(
        default=0.0,
        gt=-90.0,
        lt=90.0,
        description="Relative to the rig, negative is down.",
    )
    hfov_deg: float = Field(
        gt=0.0,
        lt=180.0,
        description="Measured across the image width, even in portrait.",
    )
    width_px: int = Field(gt=0, description="Image width.")
    height_px: int = Field(gt=0, description="Image height.")


class Rig(Model):
    """One installed product: the frame its cameras sit in, at one mount point.

    Rigs a beam apart overlap in angle before they overlap in space, which leaves a
    blind wedge over the bow; coincident cameras would hide it.
    """

    model: str | None = Field(
        default=None, description="The product, such as Pod. A recording needs it."
    )
    height_m: float = Field(gt=0.0, description="Above the waterline.")
    yaw_deg: float = Field(
        default=0.0, description="Its axis relative to the bow, positive to starboard."
    )
    pitch_deg: float = Field(
        default=0.0,
        gt=-90.0,
        lt=90.0,
        description="About its own transverse axis, after its yaw and before its "
        "cameras'. Negative is down.",
    )
    offset_x_m: float = Field(
        default=0.0, description="From the centreline, positive to starboard."
    )
    offset_y_m: float = Field(
        default=0.0, description="From midships, positive forward."
    )
    # Depth precision goes as far / near, so larger is better. The bound is a lens's
    # clearance from its own structure.
    near_clip_m: float = Field(
        default=5.0, gt=0.0, description="Anything nearer a camera is not rendered."
    )
    cameras: dict[Name, Camera] = Field(
        min_length=1, description="Its cameras, by name."
    )


class Mount(NamedTuple):
    rig_name: str
    rig: Rig
    camera_name: str
    camera: Camera

    @property
    def name(self) -> str:
        return f"{self.rig_name}_{self.camera_name}"

    @property
    def nominal_bearing_deg(self) -> float:
        """Not the achieved boresight: the rig's pitch sits between the two yaws, so
        an off-axis camera points elsewhere. `scene.boresight_deg` measures the
        built camera."""
        return self.rig.yaw_deg + self.camera.yaw_deg


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
    fetch_km: float | None = Field(
        default=None,
        gt=0.0,
        description="Open water upwind. Unset, a fully developed sea.",
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

    @property
    def fetch_m(self) -> float | None:
        return None if self.fetch_km is None else self.fetch_km * 1e3


# Judgement: 42 km, inside the open ocean's measured spread. OPAC's
# maritime aerosols at 550 nm and 80% humidity (Hess, Koepke & Schult, BAMS 79(5) 831,
# 1998), plus sea-level Rayleigh's 0.012 km^-1 (the MODTRAN 2/3 report, eq. 26): clean,
# 0.090 km^-1, is 38 km; tropical, 0.043 km^-1, is 71 km.
VISIBILITY_KM = 42.0


class Sky(Model):
    """Blender's Sky Texture or a photographed sky in EO, and the downwelling radiance
    the sea reflects in IR.

    Haze is `visibility_km`, in the air between the camera and what it sees, in both
    bands. In LWIR, `atmosphere` adds its water vapour, shapes the sky, and gives
    `t_air_k` unless it is set.
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
    visibility_km: float | None = Field(
        default=VISIBILITY_KM,
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

    # Judgement: low cloud, whose base is below 2 km (WMO International Cloud Atlas).
    cloud_base_m: float = Field(
        default=1000.0,
        ge=0.0,
        le=2000.0,
        description="The photo's cloud base, which sets its clouds' temperature in IR. "
        "EO ignores it.",
    )

    hdri: (
        Annotated[
            str, WithJsonSchema({"type": "string", "enum": sorted(skies.library())})
        ]
        | None
    ) = Field(
        default=None,
        description="A photographed sky from seascape/skies.toml, in place of the Sky "
        "Texture, which sets `sun_elevation_deg`. LWIR keeps its own clear sky and "
        "takes the photo's sun and clouds, turned to `sun_bearing_deg`.",
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
        if data.get("sun_elevation_deg", elevation) != elevation:
            warnings.warn(
                "the hdri sets the sky; sun_elevation_deg is ignored", stacklevel=2
            )
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
    def _aerosol_density_is_gone(cls, data: Any) -> Any:
        if isinstance(data, dict) and "aerosol_density" in data:
            raise ValueError("aerosol_density is gone: visibility_km is the haze")
        return data

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
_ASSET = "An asset name from the manifest: `seascape assets list` shows them."
_T_HULL = "Shaded hull temperature. IR only."
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
    heading_deg: float = Field(
        default=0.0, description="Where its bow points, clockwise."
    )
    heading_from: Literal["ownship", "line_of_sight"] = Field(
        default="ownship",
        description="What `heading_deg` turns from: the ownship's bow, or the line "
        "from the ownship to it, where 0 points away and 180 towards.",
    )
    speed_mps: float = Field(default=0.0, ge=0.0, description=_SPEED)
    drift: Drift | None = Field(default=None, description=_DRIFT)
    orbit: Orbit | None = Field(default=None, description="Round the ownship.")
    t_k: float = Field(default=293.0, ge=250.0, le=400.0, description=_T_HULL)
    count: int = Field(
        default=1,
        ge=1,
        description="How many, each drawn anew. `load` makes the copies, so a "
        "loaded scenario holds only 1.",
    )

    @property
    def course_deg(self) -> float:
        """`heading_deg` from the ownship's bow, whatever it was given from."""
        if self.heading_from == "line_of_sight":
            return self.bearing_deg + self.heading_deg
        return self.heading_deg

    @model_validator(mode="after")
    def _copies_are_made_on_load(self) -> "Object":
        if self.count != 1:
            raise ValueError(
                f"{self.asset}: count {self.count} is expanded when a scenario loads"
            )
        return self

    @model_validator(mode="after")
    def _an_orbit_steers(self) -> "Object":
        steered = {"heading_deg", "heading_from", "speed_mps", "drift"}
        steered &= self.model_fields_set
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

    asset: AssetName | Annotated[list[AssetName], Field(min_length=1)] = Field(
        description=f"{_ASSET} A list is taken in turn round the ring."
    )
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

    def poses(self) -> list[tuple[str, float, float]]:
        """Asset, bearing_deg and heading_deg per target."""
        names = [self.asset] if isinstance(self.asset, str) else self.asset
        return [
            (
                names[i % len(names)],
                self._spread(self.bearing_deg, i),
                self._spread(self.heading_deg, i),
            )
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
    """One scene: the rigs, the world around them, and what a render writes."""

    seed: int = Field(default=0, description="Seeds every random draw.")
    rigs: dict[Name, Rig] = Field(
        min_length=1, description="The products on the ownship, by name."
    )
    ownship: Ownship = Field(default_factory=Ownship)
    targets: Targets | None = Field(default=None, description="A ring of vessels.")
    sea: Sea = Field(default_factory=Sea)
    sky: Sky = Field(default_factory=Sky)
    objects: list[Object] = Field(
        default_factory=list, description="Vessels placed one by one."
    )
    outputs: Outputs = Field(default_factory=Outputs)

    @property
    def mounts(self) -> list[Mount]:
        return [
            Mount(rig_name, rig, camera_name, camera)
            for rig_name, rig in self.rigs.items()
            for camera_name, camera in rig.cameras.items()
        ]

    @property
    def images(self) -> int:
        """How many images a render writes."""
        bands = self.outputs.bands
        mounts = [m for m in self.mounts if m.camera.band in bands]
        return len(mounts) * len(self.outputs.times_s)

    @model_validator(mode="after")
    def _camera_names_are_unique(self) -> "Scenario":
        """A camera's name joins its rig's, so rig `a_b` with camera `c` and rig `a`
        with camera `b_c` would write one file."""
        names = [mount.name for mount in self.mounts]
        if len(set(names)) != len(names):
            raise ValueError(f"two cameras share a name: {sorted(names)}")
        return self

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
        error = waves.snap_error(self.sea.wind_speed_mps, self.sea.fetch_m, snap)
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


def _kind(node: Any) -> str | None:
    """`node`'s draw key, if it holds one and nothing but that draw's fields."""
    if not isinstance(node, dict):
        return None
    kinds = [key for key in node if key in _DRAWS]
    if len(kinds) == 1 and set(node) <= set(_DRAWS[kinds[0]].model_fields):
        return kinds[0]
    return None


def _is_draw(node: Any) -> bool:
    return _kind(node) is not None


def _is_table(node: Any) -> bool:
    return isinstance(node, dict) and not _is_draw(node)


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    return base | {
        key: _merge_value(base.get(key), value) for key, value in over.items()
    }


def _merge_value(base: Any, over: Any) -> Any:
    """Two tables merge, and a table over a choice of tables merges into each option;
    anything else, a draw or a table naming a preset included, replaces outright."""
    if not _is_table(over) or "preset" in over:
        return over
    if _is_table(base):
        return _merge(base, over)
    options = base.get("choice") if _is_draw(base) else None
    if isinstance(options, list) and all(_is_table(option) for option in options):
        # The choice's own fields, its weights, stay with the choice.
        own = {k: v for k, v in over.items() if k in Choice.model_fields}
        rest = {k: v for k, v in over.items() if k not in own}
        return base | own | {"choice": [_merge(option, rest) for option in options]}
    return over


def _include_path(name: Any, base: Path, block: str | None = None) -> Path:
    """Resolve rule 2. `block` is the preset directory; None allows only a path."""
    if not isinstance(name, str):
        raise TypeError(f"include must be a string, got {name!r}")
    if "/" in name or name.endswith(".toml"):
        return base / name
    if block is None:
        raise ValueError(f"{name!r} must be a path: it needs a '/' or a '.toml' suffix")
    return CFG_DIR / block / f"{name}.toml"


# Tables of named entries: an entry's preset is one of its table's, never of its name.
_KEYED = {"rigs", "cameras"}


def _expand(
    node: Any,
    block: str | None,
    base: Path,
    chain: tuple[Path, ...],
    keyed: bool = False,
) -> Any:
    """Resolve every `preset` key in the tree, innermost first."""
    if isinstance(node, list):
        return [_expand(item, block, base, chain, keyed) for item in node]
    if not isinstance(node, dict):
        return node
    # A draw holds a list of options; an entry by a draw's name would read as one.
    if keyed and (
        named := [k for k in node if k in _DRAWS and not isinstance(node[k], list)]
    ):
        raise ValueError(f"{named[0]!r} is a draw, not a name: rename it")
    # A draw's options are presets of the field it sits on, never of `choice`.
    if _is_draw(node):
        return {key: _expand(v, block, base, chain, keyed) for key, v in node.items()}
    if keyed:
        return {key: _expand(v, block, base, chain) for key, v in node.items()}
    out = {
        key: _expand(value, key, base, chain, key in _KEYED)
        for key, value in node.items()
    }
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

    if (parent := data.pop("extends", None)) is not None:
        data = _merge(_read(_include_path(parent, path.parent), chain), data)
    return _expand(data, None, path.parent, chain)


def _draw(node: Any, seed: int, path: str) -> Any:
    """Replace every draw in the tree by a value from a substream named for where it
    sits, `draw/sky/visibility_km`."""
    if isinstance(node, list):
        return [_draw(item, seed, f"{path}/{i}") for i, item in enumerate(node)]
    if not isinstance(node, dict):
        return node
    if (kind := _kind(node)) is not None:
        value = _DRAWS[kind].model_validate(node).draw(substream(seed, path))
        # A chosen draw on the pick's own substream would replay the pick's state.
        return _draw(value, seed, f"{path}/{kind}" if _is_draw(value) else path)
    return {key: _draw(value, seed, f"{path}/{key}") for key, value in node.items()}


def _copies(objects: list[Any], seed: int) -> list[Any]:
    """Each entry `count` times, every copy drawn from a substream of its own."""
    out = []
    for i, spec in enumerate(objects):
        if not isinstance(spec, dict):
            out.append(spec)  # validation names what it should have been
            continue
        path = f"draw/objects/{i}"
        count = _draw(spec.get("count", 1), seed, f"{path}/count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError(
                f"objects[{i}]: count must be a whole number from 1, not {count}"
            )
        if count > 1 and "orbit" in spec:
            raise ValueError(f"objects[{i}]: an orbit spaces its hulls by orbit.count")
        copies = [_draw(spec, seed, f"{path}/{k}") for k in range(count)]
        # A pick's count would come after the copies were made.
        if _is_draw(spec) and any(c.get("count", 1) != 1 for c in copies):
            raise ValueError(f"objects[{i}]: a drawn object cannot carry a count")
        out += [c | {"count": 1} for c in copies]
    return out


def load(
    path: str | Path, overrides: Iterable[str | Mapping[str, Any]] = ()
) -> Scenario:
    """Read a scenario TOML, resolving `extends` and `preset`, draw every random
    field, and validate it.

    Each override, a TOML assignment such as `rigs.bow.pitch_deg = -5` or the table it
    parses to, merges over the file and resolves as if it were a line in it.
    """
    path = Path(path)
    data = _read(path)
    for override in overrides:
        table = tomllib.loads(override) if isinstance(override, str) else dict(override)
        data = _expand(_merge(data, table), None, path.parent, ())
    seed = data.get("seed", Scenario.model_fields["seed"].default)
    if isinstance(seed, dict):
        raise ValueError("seed cannot be drawn: it seeds the draws")
    if isinstance(objects := data.get("objects"), list):
        data["objects"] = _copies(objects, seed)
    return Scenario.model_validate(_draw(data, seed, "draw"))


def json_schema() -> dict[str, Any]:
    """The scenario's JSON schema, in which any field can be a draw."""
    schema = Scenario.model_json_schema()
    draws = [{"$ref": f"#/$defs/{model.__name__}"} for model in _DRAWS.values()]
    for model in (schema, *schema["$defs"].values()):
        for name, spec in model.get("properties", {}).items():
            if model is schema and name == "seed":
                continue  # `load` refuses a drawn seed
            # Outside the anyOf, where an editor finds the hover doc.
            outer = {
                k: spec.pop(k) for k in ("title", "description", "default") if k in spec
            }
            values = spec.pop("anyOf", None) or [spec]
            model["properties"][name] = {**outer, "anyOf": [*values, *draws]}
    for model in _DRAWS.values():
        schema["$defs"][model.__name__] = model.model_json_schema()
    return schema
