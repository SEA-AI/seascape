"""A whole scenario, and what its render writes."""

import warnings
from typing import Any, Literal

from pydantic import (
    Field,
    model_validator,
)

from seascape.config.cameras import Band, Mount, Name, Rig
from seascape.config.draws import substream
from seascape.config.objects import Farm, Object, Ownship, Targets
from seascape.config.world import Sea, Sky
from seascape.model import Model
from seascape.physics import waves
from seascape.physics.atmosphere import ATMOSPHERE, SURFACE_SEA_K

type ImageFormat = Literal["exr", "png", "jpg"]


class Samples(Model):
    """Cycles samples per band.

    A model so an override merges field by field; a dict would replace it whole
    and drop the band left out.
    """

    eo: int = Field(default=16, gt=0, description="Per pixel, for EO frames.")
    # Where the open sea's grain, adaptive sampling off, falls below a Boson's NETD of
    # 50 mK (datasheet 102-2013-40, table 15).
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
        description=(
            "jpg to look at, png lossless or LWIR in kelvin, exr the radiance. "
            "Only an 8-bit frame is judged for visibility: an exr or LWIR png boxes "
            "every target in view."
        ),
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
    keep_if_any_sees: bool = Field(
        default=False,
        description="Boxes a target in every camera of a band that has it in frame, "
        "at an instant when any of them sees it.",
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
    farms: list[Farm] = Field(
        default_factory=list, description="Wind turbines, in rows or alone."
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

    def wind_waves(self) -> tuple[waves.Wave, ...]:
        sea = self.sea
        return waves.wind_sea(
            sea.wind_speed_mps,
            sea.wind_from_deg,
            sea.fetch_m,
            self.outputs.period_s,
            substream(self.seed, "sea/surface"),
        )

    def swell_waves(self) -> tuple[waves.Wave, ...]:
        swell = self.sea.swell
        if swell is None:
            return ()
        return waves.swell(
            swell.height_m,
            swell.period_s,
            swell.from_deg,
            self.outputs.period_s,
            substream(self.seed, "sea/swell"),
        )

    def wave_field(self) -> tuple[waves.Wave, ...]:
        return self.wind_waves() + self.swell_waves()

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
        atmosphere = data.get("sky", {}).get("atmosphere", ATMOSPHERE)
        if (
            "t_sea_k" not in sea
            and isinstance(atmosphere, str)
            and (atmosphere in SURFACE_SEA_K)
        ):
            data = {**data, "sea": {**sea, "t_sea_k": SURFACE_SEA_K[atmosphere]}}
        return data

    @model_validator(mode="after")
    def _a_warm_layer_needs_the_sun(self) -> "Scenario":
        """Fairall et al. 1996: it forms once the trapped sun outweighs the cooling."""
        if self.sea.warm_layer is not None and not self.sky.sun_up:
            raise ValueError("sea.warm_layer needs the sun above the horizon")
        return self

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
        # Three blades look the same every third of a turn.
        periods += [
            (f"{farm.asset} rotor", 20.0 / rpm)
            for farm in self.farms
            if (rpm := farm.rpm(self.sea)) > 0.0
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
