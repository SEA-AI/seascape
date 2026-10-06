"""The primer's charts, each in a light and a dark PNG, from seascape's own physics.

uv run python -m docs.primer.charts
"""

import math
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.axes import Axes

from docs.brand import FOCUS_RED, FOG_WHITE, FONT, NIGHT_BLUE, OCEAN_TEAL, SKY_GREY
from docs.primer.figures import CROP_PX, CROP_SCALE, FULL_HD, GAP_PX, HERE, HULL_DOWN_KM
from seascape import lwir, waves
from seascape.config import load

OUT = HERE / "charts"
# The crops' scenario, so the rays chart describes the camera that took them.
SCENARIO = load(HERE / "primer.toml", [FULL_HD])
(CAMERA,) = (mount.camera for mount in SCENARIO.rig.mounts)
REFRACTION_K = SCENARIO.sea.refraction_k
# IAU 2015 Resolution B3: the sun's nominal effective temperature.
SUN_K = 5772.0

THEMES = {
    "light": {
        "series": (FOCUS_RED, OCEAN_TEAL, SKY_GREY),
        "surface": "#FFFFFF",
        "ink": "#000000",
        "muted": SKY_GREY,
        "grid": FOG_WHITE,
    },
    "dark": {
        "series": (FOCUS_RED, FOG_WHITE, SKY_GREY),
        "surface": NIGHT_BLUE,
        "ink": FOG_WHITE,
        "muted": SKY_GREY,
        "grid": OCEAN_TEAL,
    },
}


def note(ax: Axes, t: dict, xy: tuple[float, float], text: str, **kw: Any) -> None:
    style: dict[str, Any] = {"color": t["ink"], "fontsize": 10}
    ax.annotate(text, xy, **style | kw)


def glow(ax: Axes, t: dict) -> None:
    lam_um = np.geomspace(0.2, 60, 600)
    lwir_um = tuple(b * 1e6 for b in lwir.BAND_M)
    for (low, high), name in (((0.4, 0.7), "EO"), (lwir_um, "LWIR")):
        ax.axvspan(low, high, color=t["grid"], zorder=0)
        ax.text(math.sqrt(low * high), 1.12, name, ha="center", color=t["muted"])
    for colour, t_k, name in zip(
        t["series"], (SUN_K, lwir.T_SEA_K), ("the sun", "the sea"), strict=False
    ):
        b = lwir.planck(lam_um * 1e-6, t_k)
        ax.plot(lam_um, b / b.max(), color=colour, lw=2)
        note(ax, t, (lam_um[b.argmax()], 1.0), f"{name}, {t_k:.0f} K",
             xytext=(8, 0), textcoords="offset points", va="center")  # fmt: skip
    ax.set_xscale("log")
    ax.set_xlim(0.2, 60)
    ax.set_ylim(0, 1.2)
    ticks = [0.2, 0.5, 1, 2, 5, 10, 20, 50]
    ax.set_xticks(ticks, [f"{x:g}" for x in ticks])
    ax.set_xlabel("wavelength, µm")
    ax.set_yticks([])


def rays(ax: Axes, t: dict) -> None:
    """Not to scale: the angles are opened up so the rays can be seen at all."""
    radius_m = waves.earth_radius_m(REFRACTION_K)
    height_m = SCENARIO.rig.height_m

    def apart(d_m: float) -> str:
        grazing = math.atan(height_m / d_m) - d_m / (2 * radius_m)
        m = d_m * math.radians(CAMERA.hfov_deg) / CAMERA.width_px / math.sin(grazing)
        return f"{m:.0f} m" if m >= 1 else f"{m * 100:.0f} cm"

    ax.plot([-0.3, 10], [0, 0], color=t["series"][1], lw=2)
    ax.plot([0, 0], [0, 1], color=t["muted"], lw=3)
    ax.plot(0, 1, "o", ms=10, color=t["ink"])
    note(ax, t, (0.15, 1.03), "camera")
    for land, spread, d_m, name in ((1.6, 0.12, 100, "100 m"), (8.0, 1.6, 1e3, "1 km")):
        for x in (land, land + spread):
            ax.plot([0, x], [1, 0], color=t["series"][0], lw=1.2)
        arrow = {"arrowstyle": "<->", "color": t["ink"], "lw": 1}
        ax.annotate("", (land, -0.08), (land + spread, -0.08), arrowprops=arrow)
        label = f"{name} out: {apart(d_m)} apart"
        note(ax, t, (land + spread / 2, -0.18), label, ha="center", va="top")
    ax.set_xlim(-0.5, 10.2)
    ax.set_ylim(-0.5, 1.25)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.spines["bottom"].set_visible(False)


def hidden(ax: Axes, t: dict) -> None:
    radius_m = waves.earth_radius_m(REFRACTION_K)
    r_km = np.linspace(0, 50, 500)
    for colour, h in zip(t["series"], (3.0, 12.0, 30.0), strict=True):
        horizon_km = waves.horizon_m(h, REFRACTION_K) / 1000
        x = r_km[r_km >= horizon_km]
        hidden_m = [-waves.sea_z_m((r - horizon_km) * 1000, 0.0, radius_m) for r in x]
        ax.plot(x, hidden_m, color=colour, lw=2)
        ax.plot(horizon_km, 0, "o", ms=8, color=colour, mec=t["surface"], mew=2)
        at_km = horizon_km + waves.horizon_m(30.0, REFRACTION_K) / 1000
        note(ax, t, (at_km, 30.0), f"{h:.0f} m up", xytext=(-8, 0),
             textcoords="offset points", va="center", ha="right")  # fmt: skip
    far_km = HULL_DOWN_KM[-1]
    behind_km = far_km - waves.horizon_m(30.0, REFRACTION_K) / 1000
    seen = -waves.sea_z_m(behind_km * 1000, 0.0, radius_m)
    ax.plot(far_km, seen, "o", ms=8, color=t["series"][2], mec=t["surface"], mew=2)
    arrow = {"arrowstyle": "-", "color": t["muted"], "lw": 1}
    text = f"the {far_km} km ship above:\n{seen:.0f} m hidden"
    note(ax, t, (far_km, seen), text, xytext=(49.5, 6), ha="right", arrowprops=arrow)
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 80)
    ax.set_xlabel("range to the target, km")
    ax.set_ylabel("height hidden by the sea, m")


# Beside crops.jpg at half width: drawn smaller to keep its text legible, and at the
# crops' aspect so the two sit at one height.
CROPS_ASPECT = CROP_PX[0] * CROP_SCALE / (2 * CROP_PX[1] * CROP_SCALE + GAP_PX)
FIGSIZE = {"rays": (4.2, 4.2 / CROPS_ASPECT)}
CHARTS = {
    "glow": (
        "The sun glows in the visible, the sea in the thermal",
        "Planck's law, each curve scaled to its peak. Shaded: what each camera sees.",
        glow,
    ),
    "rays": (
        "Far out, one pixel covers a lot of sea",
        f"Side view, not to scale. {CAMERA.width_px} px, {CAMERA.hfov_deg:.0f}° "
        f"camera, {SCENARIO.rig.height_m:.0f} m up.",
        rays,
    ),
    "hidden": (
        "Raise the camera and the horizon moves out",
        "How much of a target the curve hides. Each line starts at its horizon.",
        hidden,
    ),
}


def draw(name: str, theme: str) -> None:
    t = THEMES[theme]
    title, subtitle, chart = CHARTS[name]
    with mpl.rc_context(
        {
            "font.family": FONT,
            "font.size": 10.5,
            "figure.facecolor": t["surface"],
            "axes.facecolor": t["surface"],
            "axes.edgecolor": t["muted"],
            "axes.labelcolor": t["ink"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.axisbelow": True,
            "grid.color": t["grid"],
            "grid.linewidth": 0.8,
            "xtick.color": t["muted"],
            "xtick.labelcolor": t["ink"],
            "ytick.labelcolor": t["ink"],
            "ytick.major.size": 0,
        }
    ):
        size = FIGSIZE.get(name, (5.4, 3.0))
        fig, ax = plt.subplots(figsize=size, layout="constrained")
        chart(ax, t)
        ax.minorticks_off()
        fig.suptitle(title, x=0.01, ha="left", color=t["ink"], size=14, weight="bold")
        ax.set_title(subtitle, loc="left", color=t["muted"], fontsize=10, pad=10)
        fig.savefig(OUT / f"{name}-{theme}.png", dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    font_manager.findfont(FONT, fallback_to_default=False)
    OUT.mkdir(exist_ok=True)
    for name in CHARTS:
        for theme in THEMES:
            draw(name, theme)
