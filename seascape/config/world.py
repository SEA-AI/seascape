"""The sea and the sky."""

import math
import warnings
from typing import Annotated, Any

from pydantic import (
    Field,
    WithJsonSchema,
    model_validator,
)

from seascape.assets import skies
from seascape.model import Model
from seascape.physics import aerosol, lwir, skin
from seascape.physics.atmosphere import ATMOSPHERE, SURFACE_AIR_K, T_SEA_K, Atmosphere


class Swell(Model):
    """Waves from a distant storm, whatever the local wind."""

    height_m: float = Field(gt=0.0, description="Significant wave height.")
    period_s: float = Field(gt=0.0, description="Seconds between crests.")
    from_deg: float = Field(
        default=0.0,
        description="Where it comes from, clockwise from the ownship's bow.",
    )


class WarmLayer(Model):
    """The sun's heat trapped near the surface on a calm day, Fairall et al.'s warm
    layer. Each value is a mean since it formed; unset, their clear COARE day's."""

    hours: float = Field(
        default=skin.CLEAR_DAY_HOURS,
        gt=0.0,
        le=24.0,
        description="How long the sun has heated it, from just after sunrise.",
    )
    solar_w_m2: float = Field(
        default=skin.CLEAR_DAY_SOLAR_W_M2,
        ge=0.0,
        description="Net solar at the surface.",
    )
    cooling_w_m2: float | None = Field(
        default=None,
        description="Net cooling at the surface, longwave, sensible and latent. Unset, "
        "a clear COARE day's at the sea's wind.",
    )

    def rise_k(self, wind_speed_mps: float, t_sea_k: float) -> float:
        cooling = self.cooling_w_m2
        if cooling is None:
            cooling = skin.clear_day_cooling_w_m2(wind_speed_mps)
        rise, _ = skin.warm_layer(
            self.solar_w_m2,
            cooling,
            skin.stress_m2_s2(wind_speed_mps, t_sea_k),
            self.hours,
            t_sea_k,
        )
        return rise


class Sea(Model):
    """The temperature bound is the span of the shipped optical-constant table. `lwir`
    clamps to it; here it is an error.
    """

    t_sea_k: float = Field(
        default=T_SEA_K,
        ge=271.0,
        le=311.0,
        description="Bulk sea temperature, under the skin the camera sees. IR only. "
        "Unset, the atmosphere's own.",
    )
    warm_layer: WarmLayer | None = Field(
        default=None,
        description="A day's sun trapped near the surface, warming the skin. IR only. "
        "Opt-in: the sun's elevation does not say how long it has shone.",
    )
    # Elfouhaily et al. 1997 fitted their spectrum up to 21 m/s.
    wind_speed_mps: float = Field(
        default=7.0,
        ge=0.0,
        le=21.0,
        description="Measured 10 m above the sea. Sets the wind sea.",
    )
    wind_from_deg: float = Field(
        default=0.0,
        description="Where the wind blows from, clockwise from the ownship's bow.",
    )
    fetch_km: float | None = Field(
        default=None,
        gt=0.0,
        description="How far the wind has blown over open water: the shorter, the "
        "smaller and shorter its waves. Unset, the open ocean, with no limit.",
    )
    swell: Swell | None = Field(default=None, description="On top of the wind's sea.")
    # Morel & Maritorena 2001: eq. 14 holds from 0.02 mg/m^3, their data reach 30.
    chlorophyll_mg_m3: float = Field(
        default=0.2,
        ge=0.02,
        le=30.0,
        description="Chlorophyll in the water. Sets its colour, from clear blue to a "
        "green bloom.",
    )
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

    def t_skin_k(self, sun_up: bool) -> float:
        """What the camera sees: the bulk under a cool skin, warmed by its warm layer
        if it has one."""
        warm = self.warm_layer
        rise = 0.0 if warm is None else warm.rise_k(self.wind_speed_mps, self.t_sea_k)
        return skin.skin_k(self.t_sea_k, rise, sun_up)


# Judgement: 42 km, inside the open ocean's measured spread. OPAC's
# maritime aerosols at 550 nm and 80% humidity (Hess, Koepke & Schult, BAMS 79(5) 831,
# 1998), plus sea-level Rayleigh's 0.012 km^-1 (the MODTRAN 2/3 report, eq. 26): clean,
# 0.090 km^-1, is 38 km; tropical, 0.043 km^-1, is 71 km.
VISIBILITY_KM = 42.0


class Sky(Model):
    """Blender's Sky Texture or a photographed sky in EO, and the downwelling radiance
    the sea reflects in IR.

    Haze is `visibility_km`, in the air between the camera and what it sees, in both
    bands, and overhead. In LWIR, `atmosphere` adds its water vapour, shapes the sky,
    and gives `t_air_k` unless it is set.
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
        description="How much warmer a sunlit surface is than a shaded one, on a hull "
        "given one `t_k`. IR only.",
    )
    visibility_km: float | None = Field(
        default=VISIBILITY_KM,
        gt=0.0,
        description="Meteorological range at 550 nm; None is no aerosol. Under "
        f"{aerosol.FOG_KM:g} km it is fog, in both bands.",
    )
    atmosphere: Atmosphere = Field(
        default=ATMOSPHERE,
        description="LOWTRAN 7's model atmosphere, or the North Sea's, for the LWIR "
        "sky and air. EO ignores it. subarctic_winter's horizon partly sees space, so "
        "its horizon sky reads warmer than LOWTRAN's.",
    )
    t_air_k: float = Field(
        default=SURFACE_AIR_K[ATMOSPHERE],
        ge=250.0,
        le=320.0,
        description="Scales the IR sky. EO ignores it. Unset, the atmosphere's own.",
    )

    # Judgement: low cloud, whose base is below 2 km (WMO International Cloud Atlas).
    cloud_base_m: float = Field(
        default=1000.0,
        ge=0.0,
        le=3000.0,
        description="The photo's cloud base, which sets its clouds' temperature in IR. "
        "Unset, its cloud genus's. Cirrus takes its own. EO ignores it.",
    )

    hdri: (
        Annotated[
            str, WithJsonSchema({"type": "string", "enum": sorted(skies.library())})
        ]
        | None
    ) = Field(
        default=None,
        description="A photographed sky from seascape/assets/skies.toml, in place of "
        "the Sky Texture, which sets `sun_elevation_deg`. LWIR keeps its own clear sky "
        "and takes the photo's sun and clouds, turned to `sun_bearing_deg`.",
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
        photo = photos[data["hdri"]]
        elevation = photo.sun_elevation_deg
        # Warned, not refused: `extends` and `--set` cannot remove a key. A dumped
        # scenario holds the photo's own values and is not warned about.
        if data.get("sun_elevation_deg", elevation) != elevation:
            warnings.warn(
                "the hdri sets the sky; sun_elevation_deg is ignored", stacklevel=2
            )
        data = {**data, "sun_elevation_deg": elevation}
        if layer := lwir.CLOUD_LAYER_M.get(photo.cloud_genus):
            data.setdefault("cloud_base_m", layer[0])
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
            air_k = SURFACE_AIR_K.get(data.get("atmosphere", ATMOSPHERE))
            if air_k is not None:  # an unknown profile fails its own validation
                data = {**data, "t_air_k": air_k}
        return data

    @property
    def extinction_per_m(self) -> float:
        return aerosol.extinction_per_m(self.visibility_km)

    @property
    def fog_weight(self) -> float:
        return aerosol.fog_weight(self.visibility_km)

    @property
    def sun_transmittance(self) -> float:
        """What fog leaves of the sun's direct beam; 1 with no sun."""
        if self.sun_elevation_deg is None:
            return 1.0
        return aerosol.sun_transmittance(
            math.radians(self.sun_elevation_deg), self.visibility_km
        )

    @property
    def sun_up(self) -> bool:
        """A photo's sun with no disc is behind cloud, not set."""
        return self.sun_elevation_deg is None or self.sun_elevation_deg > 0.0

    @property
    def sun_vector(self) -> tuple[float, float, float]:
        """Unit vector towards the sun: a direction, so its bearing is not negated."""
        if self.sun_elevation_deg is None:
            raise ValueError("a photographed sun without a disc has no direction")
        elevation = math.radians(self.sun_elevation_deg)
        bearing = math.radians(self.sun_bearing_deg)
        return (
            math.cos(elevation) * math.sin(bearing),
            math.cos(elevation) * math.cos(bearing),
            math.sin(elevation),
        )
