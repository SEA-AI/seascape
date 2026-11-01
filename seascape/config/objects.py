"""The ownship and what is around it."""

import math
from typing import Annotated, Literal

from pydantic import (
    AfterValidator,
    Field,
    WithJsonSchema,
    model_validator,
)

from seascape import assets
from seascape.config.world import Sea
from seascape.model import Model
from seascape.physics import thermal, waves


def _names(turbines: bool) -> list[str]:
    return sorted(
        name
        for name, asset in assets.manifest().items()
        if isinstance(asset, assets.Turbine) == turbines
    )


def _a_vessel(name: str) -> str:
    if name not in (names := _names(turbines=False)):
        raise ValueError(f"no asset {name!r} in {names}; a turbine stands in farms")
    return name


def _a_turbine(name: str) -> str:
    if name not in (names := _names(turbines=True)):
        raise ValueError(f"no turbine {name!r} in {names}")
    return name


# The names go in the schema too, so an editor completes them.
VesselName = Annotated[
    str,
    AfterValidator(_a_vessel),
    WithJsonSchema({"type": "string", "enum": _names(turbines=False)}),
]
TurbineName = Annotated[
    str,
    AfterValidator(_a_turbine),
    WithJsonSchema({"type": "string", "enum": _names(turbines=True)}),
]
_ASSET = "An asset name from the manifest: `seascape assets list` shows them."
_T_HULL = (
    "One temperature for the whole hull, warmed by `sky.solar_gain_k` in sun. Unset, "
    "each zone settles at its own heat balance. IR only."
)
_ENGINE_OFF = (
    "Minutes since its engines stopped; 0 is running. The engine room and exhaust "
    "cool toward the air. IR only, with zones."
)
_PAINT = (
    "Solar absorptance of the paint on its hull, superstructure and funnel: white "
    "low, black high. Unset, a mid grey. IR only, with zones."
)
_WEAR = (
    "Grime, rust, waterline stain, soot and faded paint: 0 is as delivered, 1 long "
    "neglected."
)
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

    asset: VesselName = Field(description=_ASSET)
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
    t_k: float | None = Field(default=None, ge=250.0, le=400.0, description=_T_HULL)
    engine_off_min: float = Field(default=0.0, ge=0.0, description=_ENGINE_OFF)
    paint_absorptance: float | None = Field(
        default=None,
        ge=thermal.PAINT_ABSORPTANCE[0],
        le=thermal.PAINT_ABSORPTANCE[1],
        description=_PAINT,
    )
    wear: float = Field(default=0.0, ge=0.0, le=1.0, description=_WEAR)
    count: int = Field(
        default=1,
        ge=0,
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

    asset: VesselName | None = Field(default=None, description=_ASSET)
    t_k: float | None = Field(default=None, ge=250.0, le=400.0, description=_T_HULL)
    roll_deg: float = Field(
        default=0.0, gt=-90.0, lt=90.0, description="Positive is starboard down."
    )
    pitch_deg: float = Field(
        default=0.0, gt=-90.0, lt=90.0, description="Positive is bow up."
    )
    # Limitation: one sine per axis, a sum over a wave spectrum when irregular motion
    # matters.
    roll: Swing | None = Field(default=None, description="About roll_deg.")
    pitch: Swing | None = Field(default=None, description="About pitch_deg.")
    heave: Heave | None = Field(default=None, description="About the waterline.")


class Targets(Model):
    """A ring of vessels at one range, spread over a span of bearings.

    Placed in the world, so nothing guarantees a camera sees one.
    """

    asset: VesselName | Annotated[list[VesselName], Field(min_length=1)] = Field(
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
    t_k: float | None = Field(default=None, ge=250.0, le=400.0, description=_T_HULL)
    engine_off_min: float = Field(default=0.0, ge=0.0, description=_ENGINE_OFF)
    paint_absorptance: float | None = Field(
        default=None,
        ge=thermal.PAINT_ABSORPTANCE[0],
        le=thermal.PAINT_ABSORPTANCE[1],
        description=_PAINT,
    )
    wear: float = Field(default=0.0, ge=0.0, le=1.0, description=_WEAR)

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


class Farm(Model):
    """Turbines in rows of equal length, each to starboard of the last along
    `row_bearing_deg`. One row of one is a lone turbine."""

    asset: TurbineName = Field(description=f"{_ASSET} A turbine.")
    range_m: float = Field(gt=0.0, description="To the first turbine of the first row.")
    bearing_deg: float = Field(
        description="Of the first turbine, clockwise from the ownship's bow."
    )
    rows: int = Field(default=1, gt=0, description="How many rows.")
    columns: int = Field(default=1, gt=0, description="Turbines in a row.")
    # Horns Rev 1: 560 m both ways, 7 rotor diameters (Vattenfall, "Horns Rev").
    spacing_d: tuple[float, float] = Field(
        default=(7.0, 7.0),
        description="Along a row, then between rows, in rotor diameters.",
    )
    row_bearing_deg: float = Field(
        default=0.0,
        description="Where a row runs from its first turbine, clockwise from the "
        "ownship's bow.",
    )
    staggered: bool = Field(
        default=False, description="Every other row shifted half a spacing along."
    )
    paint_absorptance: float | None = Field(
        default=None,
        ge=thermal.PAINT_ABSORPTANCE[0],
        le=thermal.PAINT_ABSORPTANCE[1],
        description=_PAINT,
    )
    wear: float = Field(default=0.0, ge=0.0, le=1.0, description=_WEAR)

    @property
    def turbine(self) -> assets.Turbine:
        turbine = assets.manifest()[self.asset]
        assert isinstance(turbine, assets.Turbine)  # TurbineName checked it
        return turbine

    def rpm(self, sea: "Sea") -> float:
        """Its rotors' speed in the wind at their hub."""
        rotor = self.turbine.rotor
        return rotor.rpm(waves.wind_at_m(sea.wind_speed_mps, rotor.hub_height_m))

    def positions(self) -> list[tuple[float, float]]:
        """Range and bearing, row by row."""
        along_m, across_m = (d * self.turbine.rotor.diameter_m for d in self.spacing_d)
        bearing = math.radians(self.bearing_deg)
        row = math.radians(self.row_bearing_deg)
        placed = []
        for r in range(self.rows):
            shift = 0.5 if self.staggered and r % 2 else 0.0
            for c in range(self.columns):
                ahead_m = along_m * (c + shift)
                # Starboard of a row's bearing is (cos, -sin).
                east = (
                    self.range_m * math.sin(bearing)
                    + ahead_m * math.sin(row)
                    + r * across_m * math.cos(row)
                )
                north = (
                    self.range_m * math.cos(bearing)
                    + ahead_m * math.cos(row)
                    - r * across_m * math.sin(row)
                )
                placed.append(
                    (math.hypot(east, north), math.degrees(math.atan2(east, north)))
                )
        return placed
