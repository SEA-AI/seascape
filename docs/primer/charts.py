"""The primer's charts, each in a light and a dark PNG, from seascape's own physics.

uv run --with matplotlib python docs/primer/charts.py
"""

import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.axes import Axes

from seascape import lwir, waves
from seascape.config import Sea

OUT = Path(__file__).parent / "charts"
FONT = "Barlow Semi Condensed"
REFRACTION_K = Sea().refraction_k
# The footprint chart's camera.
HEIGHT_M, WIDTH_PX, HFOV_DEG = 12.0, 1920, 45.0
# IAU 2015 Resolution B3: the sun's nominal effective temperature.
SUN_K = 5772.0
# The shortest gravity wave, where capillarity takes over: lambda = 2 pi sqrt(sigma /
# rho g), 1.7 cm for seawater (Lamb, Hydrodynamics, section 267).
CAPILLARY_M = 0.017

# SEA.AI brand colours.
THEMES = {
    "light": {
        "series": ("#CB0D00", "#06404C", "#7B9194"),
        "surface": "#FFFFFF",
        "text": "#000000",
        "secondary": "#000000",
        "muted": "#7B9194",
        "grid": "#DFDED9",
    },
    "dark": {
        "series": ("#CB0D00", "#DFDED9", "#7B9194"),
        "surface": "#0B1731",
        "text": "#FFFFFF",
        "secondary": "#DFDED9",
        "muted": "#7B9194",
        "grid": "#06404C",
    },
}


def label_end(
    ax: Axes, t: dict, x: float, y: float, text: str, offset: tuple = (8, 0)
) -> None:
    ax.annotate(
        text,
        (x, y),
        xytext=offset,
        textcoords="offset points",
        va="center",
        ha="left",
        color=t["secondary"],
        fontsize=10,
    )


def reference(ax: Axes, t: dict, y: float, text: str, x: float = 0.01) -> None:
    ax.axhline(y, color=t["muted"], lw=1, ls=(0, (4, 3)), zorder=1)
    ax.annotate(
        text,
        (x, y),
        ha="left" if x < 0.5 else "right",
        xycoords=("axes fraction", "data"),
        xytext=(0, 4),
        textcoords="offset points",
        color=t["muted"],
        fontsize=9.5,
    )


def glow(ax: Axes, t: dict) -> None:
    lam_um = np.geomspace(0.2, 60, 600)
    lwir_um = tuple(b * 1e6 for b in lwir.BAND_M)
    for (low, high), name in (((0.4, 0.7), "EO"), (lwir_um, "LWIR")):
        ax.axvspan(low, high, color=t["grid"], zorder=0)
        ax.text(
            math.sqrt(low * high), 1.12, name, ha="center", color=t["muted"], size=10
        )
    sea_k = lwir.T_SEA_K
    bodies = ((f"the sun, {SUN_K:.0f} K", SUN_K), (f"the sea, {sea_k:.0f} K", sea_k))
    for colour, (name, t_k) in zip(t["series"], bodies, strict=False):
        b = lwir.planck(lam_um * 1e-6, t_k)
        b /= b.max()
        ax.plot(lam_um, b, color=colour, lw=2, label=name)
        label_end(ax, t, lam_um[b.argmax()], 1.0, name)
    ax.set_xscale("log")
    ax.set_xlim(0.2, 60)
    ax.set_ylim(0, 1.2)
    ticks = [0.2, 0.5, 1, 2, 5, 10, 20, 50]
    ax.set_xticks(ticks, [f"{x:g}" for x in ticks])
    ax.set_xlabel("wavelength, µm")
    ax.set_yticks([])


def footprint(ax: Axes, t: dict) -> None:
    radius_m = waves.earth_radius_m(REFRACTION_K)
    d_m = np.geomspace(20, 8000, 300)
    across = d_m * math.radians(HFOV_DEG) / WIDTH_PX
    along = across / np.sin(np.arctan(HEIGHT_M / d_m) - d_m / (2 * radius_m))
    peak_m = 2 * math.pi * waves.GRAVITY_MS2 / waves.peak_omega_rad_s(7.0) ** 2
    reference(ax, t, peak_m, f"a 7 m/s sea's peak wave, {peak_m:.0f} m")
    reference(
        ax,
        t,
        CAPILLARY_M,
        f"the shortest gravity wave, {CAPILLARY_M * 100:.1f} cm",
        x=0.99,
    )
    for colour, y, name in zip(
        t["series"], (along, across), ("along the view", "across"), strict=False
    ):
        ax.plot(d_m, y, color=colour, lw=2, label=name)
        label_end(ax, t, d_m[-1], y[-1], name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(20, 8000)
    ax.set_xticks([20, 100, 1000, 8000], ["20 m", "100 m", "1 km", "8 km"])
    ticks = [0.01, 0.1, 1, 10, 100, 1000]
    ax.set_yticks(ticks, ["1 cm", "10 cm", "1 m", "10 m", "100 m", "1 km"])
    ax.set_xlabel("distance from the camera")
    ax.set_ylabel("one pixel's footprint")


def hidden(ax: Axes, t: dict) -> None:
    radius_m = waves.earth_radius_m(REFRACTION_K)
    r_km = np.linspace(0, 50, 500)
    for colour, h in zip(t["series"], (3.0, 12.0, 30.0), strict=True):
        horizon_km = waves.horizon_m(h, REFRACTION_K) / 1000
        beyond = r_km >= horizon_km
        x = r_km[beyond]
        y = (x - horizon_km) ** 2 * 1e6 / (2 * radius_m)
        ax.plot(x, y, color=colour, lw=2, label=f"camera {h:.0f} m up")
        ax.plot(horizon_km, 0, "o", ms=8, color=colour, mec=t["surface"], mew=2)
        at_km = horizon_km + math.sqrt(2 * radius_m * 30.0) / 1000
        ax.annotate(
            f"{h:.0f} m up",
            (at_km, 30.0),
            xytext=(-8, 0),
            textcoords="offset points",
            va="center",
            ha="right",
            color=t["secondary"],
            fontsize=10,
        )
    seen = (45 - waves.horizon_m(30.0, REFRACTION_K) / 1000) ** 2 * 1e6 / (2 * radius_m)
    ax.plot(45, seen, "o", ms=8, color=t["series"][2], mec=t["surface"], mew=2)
    ax.annotate(
        f"the 45 km ship above:\n{seen:.0f} m hidden",
        (45, seen),
        xytext=(49.5, 6),
        textcoords="data",
        ha="right",
        color=t["secondary"],
        fontsize=9,
        arrowprops={"arrowstyle": "-", "color": t["muted"], "lw": 1},
    )
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 80)
    ax.set_xlabel("range to the target, km")
    ax.set_ylabel("height hidden by the sea, m")


CHARTS = {
    "glow": (
        "The sun glows in the visible, the sea in the thermal",
        "Planck's law, each curve scaled to its peak. Shaded: what each camera sees.",
        glow,
    ),
    "footprint": (
        "Around 1 km, one pixel is as long as the biggest waves",
        f"{WIDTH_PX} px across {HFOV_DEG:.0f}°, {HEIGHT_M:.0f} m up. "
        "A wave shorter than the footprint is drawn as roughness.",
        footprint,
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
            "axes.labelcolor": t["secondary"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.axisbelow": True,
            "grid.color": t["grid"],
            "grid.linewidth": 0.8,
            "xtick.color": t["muted"],
            "ytick.color": t["muted"],
            "xtick.labelcolor": t["secondary"],
            "ytick.labelcolor": t["secondary"],
        }
    ):
        fig, ax = plt.subplots(figsize=(5.4, 3.0), layout="constrained")
        chart(ax, t)
        ax.minorticks_off()
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        fig.suptitle(
            title, x=0.01, ha="left", color=t["text"], fontsize=14, fontweight="bold"
        )
        ax.set_title(subtitle, loc="left", color=t["muted"], fontsize=10, pad=10)
        fig.savefig(OUT / f"{name}-{theme}.png", dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    font_manager.findfont(FONT, fallback_to_default=False)
    OUT.mkdir(exist_ok=True)
    for name in CHARTS:
        for theme in THEMES:
            draw(name, theme)
