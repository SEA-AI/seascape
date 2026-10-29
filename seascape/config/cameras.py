"""Cameras, and the rigs that carry them."""

import math
from typing import Annotated, Literal, NamedTuple

from pydantic import (
    Field,
    model_validator,
)

from seascape.model import Model

type Band = Literal["eo", "ir"]


# Path text would write outside the output directory.
type Name = Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]+$")]


class Sensor(Model):
    """What a pixel does with the light that reaches it, in EMVA 1288's terms. Without
    a full well the camera is ideal."""

    full_well_e: float | None = Field(
        default=None, gt=0.0, description="Saturation capacity."
    )
    dark_noise_e: float = Field(default=0.0, ge=0.0, description="Temporal dark noise.")
    gain_db: float = Field(
        default=0.0, ge=0.0, description="Analogue gain: fewer electrons fill the ADC."
    )
    adc_bits: int = Field(default=12, ge=8, le=16, description="Bits per read.")
    width_px: int | None = Field(
        default=None,
        gt=0,
        description="Its columns, if more than the image's. At twice or more, the "
        "ISP demosaics and scales down, so each pixel averages its sites; below, the "
        "sensor reads at the image's pitch.",
    )
    hdr_ratio: float = Field(
        default=1.0,
        ge=1.0,
        description="1 takes one exposure, which clips; above it, a second one this "
        "many times shorter, merged and tone-mapped.",
    )

    @model_validator(mode="after")
    def _a_pixel_needs_a_full_well(self) -> "Sensor":
        changed = [
            name
            for name, field in Sensor.model_fields.items()
            if getattr(self, name) != field.default
        ]
        if self.full_well_e is None and changed:
            raise ValueError("a sensor's settings need its full_well_e")
        return self


class Lens(Model):
    """What the optics do beyond the pinhole."""

    vignetting: bool = Field(
        default=False, description="Natural falloff, cos^4 of the field angle."
    )


class Microbolometer(Model):
    """An uncooled LWIR camera's focal plane and lens aperture. Unset, it is ideal."""

    netd_k: float = Field(
        default=0.0,
        ge=0.0,
        description="Temporal noise: one standard deviation against a 30 C scene.",
    )
    pitch_um: float | None = Field(default=None, gt=0.0, description="Pixel pitch.")
    f_number: float | None = Field(
        default=None, gt=0.0, description="The lens's f-number, for its diffraction."
    )

    @model_validator(mode="after")
    def _f_number_and_pitch_go_together(self) -> "Microbolometer":
        if (self.f_number is None) != (self.pitch_um is None):
            raise ValueError("f_number and pitch_um go together")
        return self


class Agc(Model):
    """FLIR's AGC: one equalisation, with or without its information-based mode, and
    a share of linear ramp. Defaults are a Boson+'s factory settings, Product
    Datasheet 102-2013-45 Rev 100 §6.8; the presets in `presets/agc` are FLIR's for its
    Boson-based M364C LR."""

    # §6.8.1, AGC Mode.
    information_based: bool = Field(
        default=True,
        description="Weight detail up in the histogram and add it back on top, so a "
        "hull's edges take greys a smooth sea or sky would; off, plain plateau "
        "equalisation.",
    )
    # §6.8.2. Judgement: a bin is one count, as the datasheet gives no bin width.
    plateau: float = Field(
        default=0.07,
        gt=0.0,
        le=1.0,
        description="The most pixels one histogram bin counts, as a fraction of the "
        "frame.",
    )
    # Rev 114 §6.8.6. Judgement: Figure 28's curves read as a power 1 / ace.
    ace: float = Field(
        default=0.97,
        ge=0.5,
        le=4.0,
        description="Below 1 darkens the frame for more contrast in hot content, "
        "above 1 brightens it for the cold.",
    )
    # §6.8.5, Linear Percent.
    linear_fraction: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="The share of the grey range spread evenly over the frame's "
        "temperatures; the rest follows its histogram. 1 is a linear stretch.",
    )
    # §6.8.4; FLIR Camera Adjustments 102-2013-100-01 §2.2 gives the unit: 200 counts
    # need a gain of about 1.25 to fill 255. Judgement: a count is a centikelvin, the
    # unit `agc.counts` writes; a Boson's counts per kelvin are not published.
    max_gain: float = Field(
        default=1.38,
        gt=0.0,
        description="The steepest the transfer curve may be before ACE bends it, in "
        "grey levels per centikelvin, so noise on a bland frame stays grain.",
    )
    # §6.8.10 keeps 85 % of the curve each frame, and §5.1 runs at 60 Hz.
    tau_s: float = Field(
        default=-1 / 60 / math.log(0.85),
        gt=0.0,
        description="How fast the curve follows the frame.",
    )
    # §6.8.8, Smoothing Factor; a higher value puts more in the detail. Judgement:
    # Camera Adjustments' parameter table names it SigmaR, read here as an
    # edge-preserving filter's range sigma, in counts.
    smoothing: float = Field(
        default=1250.0,
        gt=0.0,
        description="Information-based only: the edge-preserving filter's range "
        "sigma, in centikelvin; a step well past it stays background rather than "
        "detail.",
    )
    # FLIR Camera Adjustments 102-2013-100-01 §2.5 defines it; the datasheet gives no
    # default, so this is the M364C LR's Day preset, the one nearest the factory.
    detail_headroom: int = Field(
        default=25,
        ge=0,
        le=127,
        description="Information-based only: grey levels kept free at each end for "
        "the detail, so a sharpened edge does not clip.",
    )
    # §6.8.7.
    dde: float = Field(
        default=0.95,
        ge=0.0,
        description="Information-based only: gain on the detail, below 1 smoothing, "
        "above 1 sharpening.",
    )


class Processing(Model):
    """The camera's ISP. White balance and sharpening need a sensor."""

    # 2^±126 stays a normal float32 (IEEE 754), as the compositor computes.
    exposure_ev: float = Field(
        default=0.0,
        ge=-126.0,
        le=126.0,
        description="Stops over auto-exposure, on top of "
        "outputs.exposure_compensation_ev.",
    )
    # D65 plus this stays inside Kim et al.'s cubic, 1667-25000 K (US 7,024,034), and,
    # a judgement, above 2500 K, where the locus's sRGB blue nears 0.
    white_balance_k: float = Field(
        default=0.0,
        ge=-4000.0,
        le=18000.0,
        description="How far the camera's guess of the light misses D65's 6504 K, "
        "along the Planckian locus. Positive guesses bluer light, so the frame "
        "comes out warmer.",
    )
    sharpen: float = Field(
        default=0.0, ge=0.0, le=4.0, description="Unsharp mask amount."
    )


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
    sensor: Sensor = Field(default_factory=Sensor)
    lens: Lens = Field(default_factory=Lens)
    processing: Processing = Field(default_factory=Processing)
    microbolometer: Microbolometer = Field(default_factory=Microbolometer)
    agc: Agc = Field(
        default_factory=Agc,
        description="How an ir jpg, and the montage, panorama and video of an ir "
        "png, turn temperature into grey.",
    )

    @model_validator(mode="after")
    def _each_band_takes_its_own_camera(self) -> "Camera":
        if self.band == "ir" and (
            (self.sensor, self.processing) != (Sensor(), Processing())
        ):
            raise ValueError("an ir camera takes no sensor or processing")
        if self.band == "eo" and (self.microbolometer, self.agc) != (
            Microbolometer(),
            Agc(),
        ):
            raise ValueError("an eo camera takes no microbolometer or agc")
        isp = self.processing.model_copy(update={"exposure_ev": 0.0})
        if self.sensor.width_px is not None and self.sensor.width_px < self.width_px:
            raise ValueError("sensor.width_px is narrower than the image")
        if self.sensor.full_well_e is None and isp != Processing():
            raise ValueError("white balance and sharpening need a sensor.full_well_e")
        return self


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
        an off-axis camera points elsewhere. `rigs.boresight_deg` measures the
        built camera."""
        return self.rig.yaw_deg + self.camera.yaw_deg
