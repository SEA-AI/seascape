"""Cameras, and the rigs that carry them."""

import math
from typing import Annotated, Literal, NamedTuple

from pydantic import (
    Field,
    model_validator,
)

from seascape.model import Model
from seascape.physics import fisheye
from seascape.physics.colour import Tone

type Band = Literal["eo", "ir"]


# Path text would write outside the output directory.
type Name = Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]+$")]


class Sensor(Model):
    """What a pixel does with the light that reaches it, in EMVA 1288's terms. Without
    a full well the camera is ideal.
    """

    full_well_e: float | None = Field(
        default=None, gt=0.0, description="Saturation capacity."
    )
    dark_noise_e: float = Field(default=0.0, ge=0.0, description="Temporal dark noise.")
    gain_db: float = Field(
        default=0.0, ge=0.0, description="Analogue gain: fewer electrons fill the ADC."
    )
    adc_bits: int = Field(default=12, ge=8, le=16, description="Bits per read.")
    pitch_um: float | None = Field(
        default=None, gt=0.0, description="Pixel pitch, for the lens's diffraction."
    )
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
    # After ISO 9358's veiling glare index. Judgement: the default, for a clean lens.
    veiling_glare: float = Field(
        default=0.02,
        ge=0.0,
        lt=1.0,
        description="The share of the light the lens scatters across the frame. Only "
        "an 8-bit EO frame takes it.",
    )
    f_number: float | None = Field(
        default=None,
        gt=0.0,
        description="The lens's f-number, which blurs each colour by its diffraction "
        "pattern. It needs sensor.pitch_um. Only png and jpg output use it. An LWIR "
        "camera sets microbolometer.f_number instead.",
    )
    fisheye: tuple[float, float, float, float] | None = Field(
        default=None,
        description="OpenCV's fisheye k1..k4, all 0 for an equidistant lens. Unset, "
        "a pinhole. hfov_deg is then the field angle across the image width.",
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
    Boson-based M364C LR.
    """

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
    """The camera's ISP. All but exposure_ev and tone need a sensor."""

    # 2^±126 stays a normal float32 (IEEE 754), as the compositor computes.
    exposure_ev: float = Field(
        default=0.0,
        ge=-126.0,
        le=126.0,
        description="Stops over auto-exposure, on top of "
        "outputs.exposure_compensation_ev.",
    )
    exposure_target: float | None = Field(
        default=None,
        gt=0.0,
        lt=1.0,
        description="The mean, before tone, that auto-exposure brings the "
        "sensor's output to, clipped or, with HDR, tone-mapped. Unset, it puts the "
        "frame's log-average at ISO 12232's mid-grey.",
    )
    white_balance: Literal["fixed", "grey_world"] = Field(
        default="fixed",
        description="fixed balances by white_balance_k. grey_world has the ISP "
        "balance each frame so the mean of its unsaturated pixels is grey "
        "(Buchsbaum 1980).",
    )
    # D65 plus this stays inside Kim et al.'s cubic, 1667-25000 K (US 7,024,034), and,
    # a judgement, above 2500 K, where the locus's sRGB blue nears 0.
    white_balance_k: float = Field(
        default=0.0,
        ge=-4000.0,
        le=18000.0,
        description="How far the camera's guess of the light misses D65's 6504 K, "
        "along the Planckian locus. Positive guesses bluer light, so the frame "
        "comes out warmer. Only the fixed white balance uses it.",
    )
    sharpen: float = Field(
        default=0.0, ge=0.0, le=4.0, description="Unsharp mask amount."
    )
    tone: Tone | Literal["none"] = Field(
        default="none",
        description="The curve the ISP encodes with, in place of sRGB's: bt709 is "
        "BT.709's OETF, emor the mean response of 201 cameras and films (Grossberg "
        "& Nayar 2004). Both darken shadows and midtones. Only png and jpg output "
        "use it.",
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
            or self.lens.f_number is not None
        ):
            raise ValueError(
                "an ir camera takes no sensor, processing or lens.f_number"
            )
        if self.band == "eo" and (self.microbolometer, self.agc) != (
            Microbolometer(),
            Agc(),
        ):
            raise ValueError("an eo camera takes no microbolometer or agc")
        isp = self.processing.model_copy(update={"exposure_ev": 0.0, "tone": "none"})
        if self.sensor.width_px is not None and self.sensor.width_px < self.width_px:
            raise ValueError("sensor.width_px is narrower than the image")
        if self.lens.f_number is not None and self.sensor.pitch_um is None:
            raise ValueError("lens.f_number needs a sensor.pitch_um")
        if self.sensor.full_well_e is None and isp != Processing():
            raise ValueError(
                "processing but exposure_ev and tone need a sensor.full_well_e"
            )
        if (k := self.lens.fisheye) is not None:
            # Limitation: the falloff is the pinhole's cos^4 of tan(theta) = r / f.
            if self.lens.vignetting:
                raise ValueError("lens.vignetting needs a pinhole camera")
            fisheye.polynomial(
                k, self.fisheye_focal_px, self.width_px, self.height_px, 1.0
            )
        return self

    @property
    def pitch_um(self) -> float | None:
        """An image pixel's pitch where the camera sets one: a microbolometer's, or a
        sensor's times the sites each image pixel averages.
        """
        if self.band == "ir":
            return self.microbolometer.pitch_um
        if self.sensor.pitch_um is None:
            return None
        sites_per_px = (self.sensor.width_px or self.width_px) / self.width_px
        return self.sensor.pitch_um * sites_per_px

    @property
    def fisheye_focal_px(self) -> float:
        """A fisheye's `fx`, the scale of its distorted angle in pixels."""
        assert self.lens.fisheye is not None
        return fisheye.focal_px(
            math.radians(self.hfov_deg), self.width_px, self.lens.fisheye
        )


class Identity(Model):
    """A camera's frames as they are."""

    kind: Literal["identity"] = Field(description="The video is the camera's frames.")
    camera: Name = Field(description="The rig's camera it records.")


class Crop(Model):
    """A camera's frames less rows and columns at their edges."""

    kind: Literal["crop"] = Field(description="The video is a part of each frame.")
    camera: Name = Field(description="The rig's camera it records.")
    left_px: int = Field(default=0, ge=0, description="Columns cut at the left.")
    right_px: int = Field(default=0, ge=0, description="Columns cut at the right.")
    top_px: int = Field(default=0, ge=0, description="Rows cut at the top.")
    bottom_px: int = Field(default=0, ge=0, description="Rows cut at the bottom.")


class Undistortion(Model):
    """A fisheye's frames as the pinhole its recorder dewarps them to."""

    kind: Literal["undistortion"] = Field(
        description="The video is what a pinhole in the camera's place sees."
    )
    camera: Name = Field(description="The rig's camera it records.")
    hfov_deg: float = Field(
        gt=0.0, lt=180.0, description="The pinhole's, edge to edge."
    )
    vfov_deg: float = Field(
        gt=0.0, lt=180.0, description="The pinhole's; with hfov_deg, its frame's shape."
    )
    downscale: int = Field(
        default=1, ge=1, description="The camera's rows per video row."
    )
    method: Literal["render"] = Field(
        default="render",
        description="render: Blender draws the pinhole, so no dewarp resamples it.",
    )

    def pinhole(self, camera: Camera) -> Camera:
        """The pinhole rendered in `camera`'s place, sized as the recorder's
        `Camera.undistort` sizes it: square pixels, rows over `downscale`, each side
        rounded down to even.
        """
        ratio = math.tan(math.radians(self.hfov_deg) / 2) / math.tan(
            math.radians(self.vfov_deg) / 2
        )
        width = int(int(camera.height_px * ratio) / self.downscale)
        height = int(camera.height_px / self.downscale)
        lens = camera.lens.model_dump() | {"fisheye": None}
        return Camera.model_validate(
            camera.model_dump()
            | {
                "hfov_deg": self.hfov_deg,
                "width_px": width - width % 2,
                "height_px": height - height % 2,
                "lens": lens,
            }
        )


class Stitch(Model):
    """Two cameras' frames side by side, as the pinhole between them sees them."""

    kind: Literal["horizontal_stitching"] = Field(
        description="The video is two pinholes' frames stitched into one."
    )
    cameras: tuple[Name, Name] = Field(
        description="The rig's cameras it records, left then right: pinholes of one "
        "band and size."
    )


type Video = Annotated[
    Identity | Crop | Undistortion | Stitch, Field(discriminator="kind")
]


def sources(video: Video) -> tuple[Name, ...]:
    """The rig's cameras `video` records."""
    return video.cameras if isinstance(video, Stitch) else (video.camera,)


class Ptu(Model):
    """A pan-tilt unit: it holds its head's pitch, and pans in a turn by the ptu."""

    # Core-Backend's `boat_setup.target_inclination`, bounded there.
    inclination_deg: float = Field(
        default=0.0,
        ge=-20.0,
        le=20.0,
        description="The pitch it holds its head at, to the horizon, its mount "
        "included. Negative is down.",
    )


class Rig(Model):
    """One installed product: the frame its cameras sit in, at one mount point.

    Rigs a beam apart overlap in angle before they overlap in space, which leaves a
    blind wedge over the bow; coincident cameras would hide it.
    """

    model: str | None = Field(
        default=None, description="The product, such as Pod. A recording needs it."
    )
    hardware_version: str | None = Field(
        default=None,
        description="The unit's hardware version, such as Oscar_V10_1280_Gen_1, which "
        "picks the recorder's profile when a recording is replayed.",
    )
    height_m: float = Field(gt=0.0, description="Above the asset's own waterline.")
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
    ptu: Ptu | None = Field(default=None, description="What it sits on, if anything.")
    offset_x_m: float = Field(
        default=0.0, description="From the centreline, positive to starboard."
    )
    offset_y_m: float = Field(
        default=0.0, description="From midships, positive forward."
    )
    cameras: dict[Name, Camera] = Field(
        min_length=1, description="Its cameras, by name."
    )
    videos: dict[Name, Video] = Field(
        default_factory=dict,
        description="What it records, by name, each from one camera or a stitch's "
        "two, which then go unlabelled. Unset, it records each camera whole. An exr "
        "ignores it.",
    )

    @model_validator(mode="after")
    def _each_video_records_a_camera(self) -> "Rig":
        # A frame is written under its video's name, beside its camera's own.
        if clash := sorted(self.videos.keys() & self.cameras.keys()):
            raise ValueError(f"videos {clash} are named as cameras")
        recorded = [c for video in self.videos.values() for c in sources(video)]
        if missing := sorted(set(recorded) - self.cameras.keys()):
            raise ValueError(f"videos record no camera {missing}")
        # One render per camera, so it takes the one crop.
        if twice := sorted({c for c in recorded if recorded.count(c) > 1}):
            raise ValueError(f"cameras {twice} are in more than one video")
        # A recording holds the videos alone.
        if self.videos and (idle := sorted(self.cameras.keys() - set(recorded))):
            raise ValueError(f"cameras {idle} are in no video, so are not recorded")
        for name, video in self.videos.items():
            if isinstance(video, Stitch):
                left, right = (self.cameras[c] for c in video.cameras)
                shape = [(c.band, c.width_px, c.height_px) for c in (left, right)]
                if shape[0] != shape[1] or left.lens.fisheye or right.lens.fisheye:
                    raise ValueError(f"video {name} stitches unlike or fisheye cameras")
                continue
            camera = self.cameras[video.camera]
            # Its focal plane's vignetting and noise are the whole sensor's.
            if isinstance(video, Crop) and camera.band == "ir":
                raise ValueError(f"video {name} crops ir camera {video.camera}")
            if isinstance(video, Crop) and (
                video.left_px + video.right_px >= camera.width_px
                or video.top_px + video.bottom_px >= camera.height_px
            ):
                raise ValueError(f"video {name} crops all of {video.camera}")
        return self

    @property
    def rendered(self) -> dict[Name, Camera]:
        """Its cameras, an undistortion's as its pinhole."""
        cameras = dict(self.cameras)
        for video in self.videos.values():
            if isinstance(video, Undistortion):
                cameras[video.camera] = video.pinhole(cameras[video.camera])
        return cameras


class Mount(NamedTuple):
    """A camera and the rig carrying it."""

    rig_name: str
    rig: Rig
    camera_name: str
    camera: Camera

    @property
    def name(self) -> str:
        """The camera's name, unique in the scenario."""
        return f"{self.rig_name}_{self.camera_name}"

    @property
    def nominal_bearing_deg(self) -> float:
        """Not the achieved boresight: the rig's pitch sits between the two yaws, so
        an off-axis camera points elsewhere. `rigs.boresight_deg` measures the
        built camera.
        """
        return self.rig.yaw_deg + self.camera.yaw_deg
