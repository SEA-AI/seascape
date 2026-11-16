"""The LWIR camera's lens and focal plane."""

import math

import numpy as np
import pytest

from seascape.config import Camera, Microbolometer, substream
from seascape.outputs import microbolometer
from seascape.physics import lwir, optics

PITCH_M = 12e-6


def goodman(x: float) -> float:
    """A clear circular pupil's MTF at `x` of its cutoff."""
    return 2 / math.pi * (math.acos(x) - x * math.sqrt(1 - x * x))


def test_the_band_s_mtf_lies_between_its_edges() -> None:
    nyquist = 1 / (2 * PITCH_M)
    band = microbolometer.diffraction_mtf(nyquist, 1.0, 290.0)
    short, long = (goodman(nyquist * lam) for lam in lwir.BAND_M)
    assert long < band < short


def test_nothing_passes_beyond_the_shortest_wavelength_s_cutoff() -> None:
    cutoff = 1 / (lwir.BAND_M[0] * 1.2)
    assert microbolometer.diffraction_mtf([0.0, cutoff], 1.2, 290.0) == pytest.approx(
        [1.0, 0.0]
    )


def test_the_lens_spreads_a_point_and_keeps_its_power() -> None:
    radiance = np.full((64, 64), 3.0)
    radiance[32, 32] = 30.0
    blurred = microbolometer.diffract(radiance, 1.0, PITCH_M, 290.0)
    # The Airy disc's far rings fall outside the frame's mirrored margin.
    assert (blurred - 3.0).sum() == pytest.approx(27.0, rel=0.01)
    assert blurred[32, 32] < 30.0
    assert blurred[32, 33] > 3.0


def test_vignetting_falls_as_the_fourth_power_of_the_cosine() -> None:
    hfov = math.radians(32.0)
    shade = microbolometer.falloff((512, 640), hfov)
    focal_px = 320 / math.tan(hfov / 2)
    corner = math.atan(math.hypot(319.5, 255.5) / focal_px)
    assert shade[0, 0] == pytest.approx(math.cos(corner) ** 4)
    assert shade.max() == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("t_k", [240.0, microbolometer.NETD_BACKGROUND_K])
def test_the_noise_is_the_netd_s_radiance(t_k: float) -> None:
    netd_k = 0.05
    camera = Camera(
        band="ir",
        hfov_deg=32.0,
        width_px=256,
        height_px=256,
        microbolometer=Microbolometer(netd_k=netd_k),
    )
    radiance = np.full((256, 256), lwir.band_radiance(t_k))
    seen = microbolometer.seen(radiance, camera, substream(0, "test"))
    background = microbolometer.NETD_BACKGROUND_K
    sigma = lwir.band_radiance(background + netd_k) - lwir.band_radiance(background)
    slope = lwir.band_radiance(t_k + netd_k) - lwir.band_radiance(t_k)
    assert seen.mean() == pytest.approx(t_k, abs=1e-3)
    assert seen.std() == pytest.approx(netd_k * sigma / slope, rel=0.02)


def _filtered(
    radiance: np.ndarray, f_number: float, pitch_m: float, t_k: float
) -> np.ndarray:
    """`radiance` through `diffraction_mtf` at its padded frame's frequencies."""
    pad = optics.airy_reach_px(lwir.BAND_M[1], f_number, pitch_m)
    padded = np.pad(radiance, pad, mode="reflect")
    fy = np.fft.fftfreq(padded.shape[0])[:, None]
    fx = np.fft.rfftfreq(padded.shape[1])[None, :]
    mtf = microbolometer.diffraction_mtf(np.hypot(fx, fy) / pitch_m, f_number, t_k)
    return np.fft.irfft2(np.fft.rfft2(padded) * mtf, s=padded.shape)[pad:-pad, pad:-pad]


def test_each_frame_is_filtered_by_its_own_camera_s_band_mtf() -> None:
    """Frames in turn, as cameras and scenes change: none reads another's MTF. 11.5 um
    and f/1.05 pad a frame as 12 um at f/1 does, so only the key tells their MTFs
    apart.
    """
    frames = [
        ((40, 72), 1.0, PITCH_M, 290.0),
        ((40, 72), 1.0, PITCH_M, 290.0),
        ((40, 72), 1.0, PITCH_M, 240.0),
        ((40, 72), 1.0, 11.5e-6, 240.0),
        ((40, 72), 1.05, PITCH_M, 240.0),
        ((72, 40), 1.0, PITCH_M, 290.0),
    ]
    draws = substream(1, "test/radiance")
    radiances = [draws.normal(3.0, 1.0, shape) for shape, *_ in frames]

    seen = [
        microbolometer.diffract(r, *camera)
        for r, (_, *camera) in zip(radiances, frames, strict=True)
    ]

    expected = [
        _filtered(r, *camera) for r, (_, *camera) in zip(radiances, frames, strict=True)
    ]
    assert [f.tobytes() for f in seen] == [f.tobytes() for f in expected]
