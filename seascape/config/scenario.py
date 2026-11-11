"""A whole scenario, and what its render writes."""

import math
import warnings
from typing import Literal

from pydantic import (
    Field,
    model_validator,
)

from seascape.assets import manifest
from seascape.config.cameras import Band, Mount, Name, Rig
from seascape.config.draws import substream
from seascape.config.objects import MAX_HULLS, Farm, Object, Ownship, Targets, Turn
from seascape.config.world import Sea, Sky
from seascape.model import Model, ScenarioWarning
from seascape.physics import waves
from seascape.physics.atmosphere import SURFACE_SEA_K

type ImageFormat = Literal["exr", "png", "jpg"]


class Samples(Model):
    """Cycles samples per band.

    A model so an override merges field by field; a dict would replace it whole
    and drop the band left out.
    """

    # Judgement: a trade between an EO frame's grain and its render time.
    eo: int = Field(default=16, gt=0, description="Per pixel, for EO frames.")
    # Where the open sea's grain, adaptive sampling off, falls below a Boson's NETD of
    # 50 mK (datasheet 102-2013-40 Rev 340, table 13).
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
    def frames(self) -> int:
        """How many frames; one for a still."""
        return max(1, round(self.duration_s * self.fps))

    @property
    def times_s(self) -> list[float]:
        """Each frame's time, from 0; one frame for a still."""
        return [f / self.fps for f in range(self.frames)]

    @property
    def span_s(self) -> float:
        """From frame 0 to the frame after the last, which a loop makes frame 0."""
        return self.frames / self.fps

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

    seed: int = Field(default=0, ge=0, description="Seeds every random draw.")
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
    turn: Turn | None = Field(default=None, description="A scripted turn.")
    outputs: Outputs = Field(default_factory=Outputs)

    @property
    def mounts(self) -> list[Mount]:
        """Every camera as rendered, with its rig; an exr renders the rig's own."""
        exr = self.outputs.format == "exr"
        return [
            Mount(rig_name, rig, camera_name, camera)
            for rig_name, rig in self.rigs.items()
            for camera_name, camera in (rig.cameras if exr else rig.rendered).items()
        ]

    def band_mounts(self, band: Band) -> list[Mount]:
        """The cameras that see in `band`."""
        return [m for m in self.mounts if m.camera.band == band]

    @property
    def images(self) -> int:
        """How many images a render writes."""
        rendered = sum(m.camera.band in self.outputs.bands for m in self.mounts)
        return rendered * self.outputs.frames

    @property
    def worn(self) -> frozenset[str]:
        """The assets an object, the ring or a farm wears, less the weathered ones,
        whose textures are their wear.
        """
        wearing = [(o.asset, o.wear) for o in self.objects]
        wearing += [(f.asset, f.wear) for f in self.farms]
        if (ring := self.targets) is not None:
            wearing += [(asset, ring.wear) for asset, _, _ in ring.poses()]
        return frozenset(
            a for a, wear in wearing if wear > 0.0 and not manifest()[a].weathered
        )

    def wind_waves(self) -> tuple[waves.Wave, ...]:
        """The wind sea's waves, from the scenario's wind and fetch."""
        sea = self.sea
        return waves.wind_sea(
            sea.wind_speed_mps,
            sea.wind_from_deg,
            sea.fetch_m,
            self.outputs.period_s,
            substream(self.seed, "sea/surface"),
        )

    def swell_waves(self) -> tuple[waves.Wave, ...]:
        """The swell's waves; none without a swell."""
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

    @model_validator(mode="after")
    def _hulls_are_bounded(self) -> "Scenario":
        hulls = (
            sum(o.orbit.count if o.orbit else 1 for o in self.objects)
            + (self.targets.count if self.targets else 0)
            + sum(farm.rows * farm.columns for farm in self.farms)
        )
        if hulls > MAX_HULLS:
            raise ValueError(f"{hulls} hulls; a scenario holds at most {MAX_HULLS}")
        return self

    @model_validator(mode="after")
    def _camera_names_are_unique(self) -> "Scenario":
        """A camera's name joins its rig's, so rig `a_b` with camera `c` and rig `a`
        with camera `b_c` would write one file.
        """
        names = [mount.name for mount in self.mounts]
        if len(set(names)) != len(names):
            raise ValueError(f"two cameras share a name: {sorted(names)}")
        return self

    @model_validator(mode="after")
    def _sea_follows_the_atmosphere(self) -> "Scenario":
        """The sea's own field cannot see the sky's, so the scenario fills it."""
        if "t_sea_k" not in self.sea.model_fields_set:
            # Validated anew, which model_copy would skip; the caller's Sea stays.
            sea = self.sea.model_dump(exclude_unset=True)
            t_sea_k = SURFACE_SEA_K[self.sky.atmosphere]
            self.sea = Sea.model_validate({**sea, "t_sea_k": t_sea_k})
        return self

    @model_validator(mode="after")
    def _a_warm_layer_needs_the_sun(self) -> "Scenario":
        """Fairall et al. 1996: it forms once the trapped sun outweighs the cooling."""
        if self.sea.warm_layer is not None and not self.sky.sun_up:
            raise ValueError("sea.warm_layer needs the sun above the horizon")
        return self

    @model_validator(mode="after")
    def _an_orbit_needs_a_fixed_ownship(self) -> "Scenario":
        if self.ownship.speed_mps > 0.0 and any(o.orbit for o in self.objects):
            raise ValueError("an object orbits the origin: set ownship.speed_mps to 0")
        return self

    @model_validator(mode="after")
    def _a_turn_fits(self) -> "Scenario":
        if self.turn is None:
            return self
        if self.outputs.loop:
            raise ValueError("a turn is a script, not a loop: drop turn or loop")
        if (turn_s := self.turn.duration_s) > self.outputs.times_s[-1]:
            raise ValueError(
                f"the turn takes {turn_s:.2f} s: make outputs.duration_s at least "
                f"{math.ceil((turn_s + 1.0 / self.outputs.fps) * 100) / 100:.2f}"
            )
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
        if self.ownship.speed_mps > 0.0:
            raise ValueError(
                "ownship.speed_mps > 0, and a straight run never comes back to close "
                "a loop: set it to 0"
            )
        motions = [
            ("ownship.yaw", self.ownship.yaw),
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
        # The blades look the same every `symmetry_rad`.
        periods += [
            (f"{farm.asset} rotor", farm.turbine.rotor.symmetry_period_s(rpm))
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
                ScenarioWarning,
                stacklevel=2,
            )
        return self
