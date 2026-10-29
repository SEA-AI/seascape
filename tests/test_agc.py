"""The thermal camera's two outputs: 16-bit counts, and 8-bit grey after AGC."""

import math
import tomllib

import cv2
import numpy as np
import pytest

from seascape.config import PRESETS, Agc, Camera
from seascape.outputs import agc

# What a scenario gets: the factory default and every preset.
SHIPPED = {"factory": Agc()} | {
    path.stem: Agc.model_validate(tomllib.loads(path.read_text()))
    for path in sorted((PRESETS / "agc").glob("*.toml"))
}
shipped = pytest.mark.parametrize("setting", SHIPPED.values(), ids=SHIPPED.keys())


def column(*t_k: float) -> np.ndarray:
    return np.array(t_k)[:, None]


def linear() -> agc.Agc:
    return agc.Agc(Agc(information_based=False, linear_fraction=1.0, ace=1.0))


def test_a_frame_is_stretched_to_the_full_range() -> None:
    assert linear()(column(272.0, 295.0)).ravel().tolist() == [0, 255]


def test_the_middle_temperature_is_mid_grey() -> None:
    """Catches a gamma encode, which puts 0.5 at 0.74."""
    assert linear()(column(270.0, 285.0, 300.0))[1, 0] == 128


@shipped
def test_a_target_is_not_flattened_to_white(setting: Agc) -> None:
    """A percentile stretch would trim the few rows a distant hull occupies."""
    grey = agc.Agc(setting)(column(*[285.0] * 40, 300.0)).ravel().astype(int)
    # Information-based keeps the flat background above its headroom.
    black = setting.detail_headroom if setting.information_based else 0
    assert (grey[0], grey[-1]) == pytest.approx((black, 255), abs=1)


@shipped
def test_a_frame_of_one_temperature_is_mid_grey_bent_by_ace(setting: Agc) -> None:
    mid = 255 * 0.5 ** (1 / setting.ace)
    if setting.information_based:
        mid = setting.detail_headroom + (255 - 2 * setting.detail_headroom) / 255 * mid
    mid = round(mid)
    assert agc.Agc(setting)(column(290.0, 290.0)).ravel().tolist() == [mid, mid]


def test_ace_below_1_darkens() -> None:
    t_k = np.linspace(280.0, 300.0, 256)[:, None] * np.ones((1, 8))
    flat, dark = (agc.Agc(Agc(ace=ace))(t_k).mean() for ace in (1.0, 0.9))
    assert dark < flat


def test_a_hot_exhaust_leaves_the_scene_its_contrast() -> None:
    """A sea and hull 10 K apart, and a hundredth of the frame at 230 C: linear gives
    the scene a few greys, equalisation most of them."""
    rng = np.random.default_rng(0)
    scene = rng.uniform(283.0, 293.0, (99, 100))
    t_k = np.vstack([scene, np.full((1, 100), 503.15)])
    assert np.ptp(linear()(t_k)[:-1]) < 15
    assert np.ptp(agc.Agc(Agc())(t_k)[:-1]) > 180


def textured() -> np.ndarray:
    """A smooth 285-290 K sea and, in it, a 1 K checkerboard at 288.5 K."""
    t_k = np.linspace(285.0, 290.0, 256)[:, None] * np.ones((1, 256))
    t_k[120:136, 120:136] = 288.5 + np.indices((16, 16)).sum(axis=0) % 2 - 0.5
    return t_k


def test_detail_takes_greys_from_a_smooth_background() -> None:
    """A textured target among the sea's own temperatures: information-based gives
    it more greys than plateau does, and the smooth sea fewer."""
    t_k = textured()
    target, sea = np.s_[124:132, 124:132], np.s_[:100]
    plateau = agc.Agc(Agc(information_based=False))(t_k)
    boson = agc.Agc(Agc(detail_headroom=0))(t_k)
    assert np.ptp(boson[target]) > np.ptp(plateau[target])
    assert np.ptp(boson[sea]) < np.ptp(plateau[sea])


def test_the_headroom_keeps_the_background_off_the_ends() -> None:
    """A smooth ramp has no detail: information-based, it spans the curve squeezed
    inside the headroom."""
    t_k = np.linspace(280.0, 300.0, 256)[:, None] * np.ones((1, 8))
    grey = agc.Agc(Agc(detail_headroom=40))(t_k).astype(int)
    assert (grey.min(), grey.max()) == pytest.approx((40, 215), abs=1)


def test_dde_gains_the_detail() -> None:
    t_k = textured()
    spread = [
        np.ptp(agc.Agc(Agc(dde=dde))(t_k)[124:132, 124:132]) for dde in (0.5, 1.0, 2.0)
    ]
    assert spread == sorted(spread)
    assert spread[0] < spread[-1]


@shipped
def test_noise_on_a_uniform_frame_gains_at_most_the_cap(setting: Agc) -> None:
    """A linear stretch would spread 50 mK of noise across all 256 greys. Noise is
    detail, so DDE above 1 gains it past the cap."""
    t_k = np.random.default_rng(0).normal(290.0, 0.05, (64, 64))
    grey = agc.Agc(setting)(t_k)
    dde = max(setting.dde, 1.0) if setting.information_based else 1.0
    assert abs(grey.mean() - 128) < 5
    assert grey.std() < 1.05 * dde * setting.max_gain * t_k.std() * agc.CENTIKELVIN


def test_the_slope_never_exceeds_the_cap() -> None:
    """A 1 K ramp is 101 counts: the cap leaves it short of the full range."""
    t_k = np.linspace(290.0, 291.0, 101)[:, None]
    plateau = Agc(information_based=False)
    grey = agc.Agc(plateau)(t_k).astype(int)
    assert np.diff(grey, axis=0).max() <= 2
    assert np.ptp(grey) <= plateau.max_gain * 101


def test_without_a_linear_share_two_temperatures_take_two_greys() -> None:
    """The counts between them hold no slope, and must not make the curve NaN."""
    t_k = np.repeat([[290.0, 310.0]], 10, axis=0)
    grey = agc.Agc(Agc(information_based=False, linear_fraction=0.0))(t_k)
    assert (grey[:, 0] < grey[:, 1]).all()


@shipped
def test_without_a_step_each_frame_takes_its_own_curve(setting: Agc) -> None:
    tone = agc.Agc(setting)
    tone(column(280.0, 290.0))
    after = column(290.0, 300.0)
    assert tone(after).tolist() == agc.Agc(setting)(after).tolist()


@shipped
def test_the_curve_follows_a_step_change_over_the_time_constant(setting: Agc) -> None:
    """A hot target entering must not flash the background: after one time constant
    the curve has moved 1 - 1/e of the way, and after many it has arrived."""
    before, after = column(280.0, 290.0), column(280.0, 287.0, 300.0)
    tone = agc.Agc(setting, setting.tau_s / 10)
    tone(before)
    for _ in range(9):
        tone(after)
    moved = 1 - math.exp(-1)
    curve = (1 - moved) * agc.transfer([before], setting) + moved * agc.transfer(
        [after], setting
    )
    assert tone(after).tolist() == agc.grey(after, curve, setting).tolist()
    for _ in range(200):
        tone(after)
    assert tone(after).tolist() == agc.Agc(setting)(after).tolist()


def test_an_eo_camera_takes_no_agc() -> None:
    with pytest.raises(ValueError, match="agc"):
        Camera(
            band="eo",
            hfov_deg=45.0,
            width_px=64,
            height_px=48,
            agc={"information_based": False},
        )


def test_counts_survive_a_png(tmp_path) -> None:
    t_k = column(271.234, 300.987)
    path = tmp_path / "frame.png"
    cv2.imwrite(str(path), agc.counts(t_k))
    assert agc.kelvin(path) == pytest.approx(t_k, abs=0.005)


def test_an_8_bit_image_has_no_temperatures(tmp_path) -> None:
    path = tmp_path / "frame.png"
    cv2.imwrite(str(path), np.zeros((2, 2), np.uint8))
    assert agc.kelvin(path) is None


def test_a_temperature_past_16_bits_raises() -> None:
    with pytest.raises(ValueError, match="overflows"):
        agc.counts(column(700.0))
