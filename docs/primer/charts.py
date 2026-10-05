"""The primer's charts, each in a light and a dark SVG, from seascape's own physics.

uv run --with matplotlib python docs/primer/charts.py
"""

import math
from collections.abc import Callable
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from seascape import lwir, waves

OUT = Path(__file__).parent / "charts"
ATMOSPHERE = "north_sea"
T_K = 288.0
# IAU 2015 Resolution B3: the sun's nominal effective temperature.
SUN_K = 5772.0
# The shortest gravity wave, where capillarity takes over: lambda = 2 pi sqrt(sigma /
# rho g), 1.7 cm for seawater (Lamb, Hydrodynamics, section 267).
CAPILLARY_M = 0.017

# Light and dark steps of one palette: categorical slots 1-3, then text and chrome.
THEMES = {
    "light": {
        "series": ("#2a78d6", "#eb6834", "#1baf7a"),
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "band": "#f0efec",
    },
    "dark": {
        "series": ("#3987e5", "#d95926", "#199e70"),
        "surface": "#1a1a19",
        "text": "#ffffff",
        "secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "band": "#2c2c2a",
    },
}

type Theme = dict
# A chart draws on one Axes, or on a list of them when it has `PANELS`.
type Chart = Callable[..., None]


def label_end(
    ax: Axes, t: Theme, x: float, y: float, text: str, offset: tuple = (6, 0)
) -> None:
    ax.annotate(
        text,
        (x, y),
        xytext=offset,
        textcoords="offset points",
        va="center",
        ha="right" if offset[0] < 0 else "center" if offset[0] == 0 else "left",
        color=t["secondary"],
        fontsize=9,
    )


def reference(ax: Axes, t: Theme, y: float, text: str, x: float = 0.01) -> None:
    ax.axhline(y, color=t["muted"], lw=1, ls=(0, (4, 3)), zorder=1)
    ax.annotate(
        text,
        (x, y),
        ha="left" if x < 0.5 else "right",
        xycoords=("axes fraction", "data"),
        xytext=(0, 4),
        textcoords="offset points",
        color=t["muted"],
        fontsize=8.5,
    )


def glow(ax: Axes, t: Theme) -> None:
    lam_um = np.geomspace(0.2, 60, 600)
    sky_k = float(
        lwir.brightness_temperature(
            lwir.sky_radiance(np.array([math.pi / 2]), T_K, ATMOSPHERE)
        )[0]
    )
    for (low, high), name in (((0.4, 0.7), "EO"), ((8, 14), "LWIR")):
        ax.axvspan(low, high, color=t["band"], zorder=0)
        ax.text(
            math.sqrt(low * high), 1.12, name, ha="center", color=t["muted"], size=9
        )
    bodies = (
        (f"sea {T_K:.0f} K", T_K),
        (f"sun {SUN_K:.0f} K", SUN_K),
        (f"zenith sky {sky_k:.0f} K", sky_k),
    )
    sides = ((-8, 0), (8, 0), (8, 0))
    for colour, (name, t_k), side in zip(t["series"], bodies, sides, strict=True):
        b = lwir.planck(lam_um * 1e-6, t_k)
        b /= b.max()
        ax.plot(lam_um, b, color=colour, lw=2, label=name)
        label_end(ax, t, lam_um[b.argmax()], 1.0, name, side)
    ax.set_xscale("log")
    ax.set_xlim(0.2, 60)
    ax.set_ylim(0, 1.2)
    ticks = [0.2, 0.5, 1, 2, 5, 10, 20, 50]
    ax.set_xticks(ticks, [f"{x:g}" for x in ticks])
    ax.set_xlabel("wavelength, µm")
    ax.set_ylabel("radiance, scaled to each peak")


def sea_temperature(ax: Axes, t: Theme) -> None:
    height_m = 12.0
    radius_m = waves.earth_radius_m(0.13)
    d_m = np.geomspace(10, 0.99 * waves.horizon_m(height_m, 0.13), 400)
    grazing = np.arctan(height_m / d_m) - d_m / (2 * radius_m)
    theta, eps = lwir.emissivity_curve(t_sea_k=T_K)
    e = np.interp(math.pi / 2 - grazing, theta, eps)
    sky = lwir.sky_radiance(grazing, T_K, ATMOSPHERE)
    seen = lwir.brightness_temperature(e * lwir.band_radiance(T_K) + (1 - e) * sky)
    reference(ax, t, T_K, f"the sea's and the air's real temperature, {T_K:.0f} K")
    ax.plot(d_m, seen, color=t["series"][0], lw=2)
    i = seen.argmin()
    ax.plot(d_m[i], seen[i], "o", ms=8, color=t["series"][0], mec=t["surface"], mew=2)
    ax.annotate(
        f"{seen[i]:.0f} K at {d_m[i]:.0f} m",
        (d_m[i], seen[i]),
        xytext=(10, -4),
        textcoords="offset points",
        color=t["secondary"],
        fontsize=9,
    )
    for x, y, text in (
        (11, 281.6, "close in, it emits\nits own temperature"),
        (1500, 284, "far out, it mirrors the\nhorizon sky, at air temperature"),
    ):
        ax.text(x, y, text, color=t["muted"], fontsize=8.5, va="top")
    ax.set_xscale("log")
    ax.set_xlim(10, d_m[-1])
    ax.set_ylim(seen.min() - 2, T_K + 2)
    ax.set_xticks([10, 100, 1000, 10000], ["10 m", "100 m", "1 km", "10 km"])
    ax.set_xlabel("distance from the camera")
    ax.set_ylabel("brightness temperature, K")


def spectrum(ax: Axes, t: Theme) -> None:
    period_s = np.geomspace(0.5, 20, 500)
    omega = 2 * math.pi / period_s
    g = waves.GRAVITY_MS2
    for colour, u in zip(t["series"], (2.0, 7.0, 14.0), strict=True):
        wp = waves.peak_omega_rad_s(u)
        s = waves.PM_ALPHA * g**2 * omega**-5 * np.exp(-1.25 * (wp / omega) ** 4)
        ax.plot(period_s, s, color=colour, lw=2, label=f"{u:.0f} m/s")
        i = s.argmax()
        label_end(ax, t, period_s[i], s[i], f"{u:.0f} m/s", (0, -14))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 1e2)
    ax.set_xlim(0.5, 20)
    ticks = [0.5, 1, 2, 5, 10, 20]
    ax.set_xticks(ticks, [f"{x:g}" for x in ticks])
    ax.set_xlabel("wave period, s")
    ax.set_ylabel("energy, m² s")


def footprint(ax: Axes, t: Theme) -> None:
    height_m, width_px, hfov = 12.0, 1920, math.radians(45)
    radius_m = waves.earth_radius_m(0.13)
    d_m = np.geomspace(20, 8000, 300)
    across = d_m * hfov / width_px
    along = across / np.sin(np.arctan(height_m / d_m) - d_m / (2 * radius_m))
    peak_m = 2 * math.pi * waves.GRAVITY_MS2 / waves.peak_omega_rad_s(7.0) ** 2
    reference(ax, t, peak_m, f"a 7 m/s sea's peak wave, {peak_m:.0f} m")
    reference(ax, t, CAPILLARY_M, "the shortest gravity wave, 1.7 cm", x=0.99)
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


def slope(ax: Axes, t: Theme) -> None:
    u = np.linspace(0, 20, 200)
    fits = (
        ("clean sea", waves.cox_munk_slope),
        ("under a slick", waves.cox_munk_slick_slope),
    )
    for colour, (name, fit) in zip(t["series"], fits, strict=False):
        deg = np.degrees(np.arctan([fit(x) for x in u]))
        ax.plot(u, deg, color=colour, lw=2, label=name)
        label_end(ax, t, u[-1], deg[-1], name)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.set_xticks(range(0, 21, 5))
    ax.set_xlabel("wind at 10 m, m/s")
    ax.set_ylabel("RMS slope of the sea, degrees")


def haze(axes: list[Axes], t: Theme) -> None:
    blue, orange = t["series"][:2]
    # Koschmieder: at the visibility a dark target keeps 2 % of its contrast.
    for ax, v_km, far_km, air in zip(
        axes, (42.0, 3.0), (20.0, 5.0), ("clear air", "haze"), strict=True
    ):
        r_m = np.linspace(10, far_km * 1000, 400)
        eo = np.exp(-math.log(50) * r_m / (v_km * 1000))
        ir = np.exp(-lwir.path_optical_depth(r_m, v_km, ATMOSPHERE))
        for colour, y, band in ((blue, eo, "EO"), (orange, ir, "LWIR")):
            ax.plot(r_m / 1000, y, color=colour, lw=2, label=band)
            label_end(ax, t, far_km, y[-1], band)
        ax.text(
            0.97,
            0.95,
            f"{air}, visibility {v_km:.0f} km",
            transform=ax.transAxes,
            ha="right",
            va="top",
            color=t["text"],
            fontsize=9.5,
        )
        ax.set_xlim(0, far_km)
        ax.set_xlabel("range, km")
    axes[0].set_ylim(0, 1.02)
    axes[0].set_ylabel("share of the target's light kept")


def hidden(ax: Axes, t: Theme) -> None:
    radius_m = waves.earth_radius_m(0.13)
    r_km = np.linspace(0, 50, 500)
    for colour, h in zip(t["series"], (3.0, 12.0, 30.0), strict=True):
        horizon_km = waves.horizon_m(h, 0.13) / 1000
        beyond = r_km >= horizon_km
        x = r_km[beyond]
        y = (x - horizon_km) ** 2 * 1e6 / (2 * radius_m)
        ax.plot(x, y, color=colour, lw=2, label=f"camera {h:.0f} m up")
        ax.plot(horizon_km, 0, "o", ms=8, color=colour, mec=t["surface"], mew=2)
        at_km = horizon_km + math.sqrt(2 * radius_m * 25.0) / 1000
        label_end(ax, t, at_km, 25.0, f"{h:.0f} m up", (8, -6))
    seen = (45 - waves.horizon_m(30.0, 0.13) / 1000) ** 2 * 1e6 / (2 * radius_m)
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
    ax.set_xlabel("range to the target, km (dots: the horizon)")
    ax.set_ylabel("height hidden by the sea, m")


PANELS = {"haze": 2}
CHARTS: dict[str, tuple[str, str, Chart]] = {
    "glow": (
        "Sun, sea and sky each glow at their own wavelength",
        "Planck's law, each curve scaled to its own peak. Shaded, each camera's band.",
        glow,
    ),
    "sea-temperature": (
        "A 288 K sea looks 10 K colder 100 m out",
        "A flat sea seen from 12 m, sea and air both 288 K, North Sea sky, no haze.",
        sea_temperature,
    ),
    "spectrum": (
        "Double the wind: twice the period, about 16 times the energy",
        "Pierson-Moskowitz spectrum of a fully developed sea, wind at 10 m.",
        spectrum,
    ),
    "footprint": (
        "By 1 km, one pixel is as long as the biggest waves",
        "1920 px across 45°, 12 m up. A wave shorter than the footprint is "
        "drawn as roughness.",
        footprint,
    ),
    "slope": (
        "The windier the sea, the wider the glitter",
        "Cox & Munk 1954: RMS surface slope, measured from photographs of sun glitter.",
        slope,
    ),
    "haze": (
        "Clear air favours EO, haze favours LWIR",
        "Koschmieder for EO, LOWTRAN 7 North Sea for LWIR, along the sea surface.",
        haze,
    ),
    "hidden": (
        "Raise the camera and the horizon moves out",
        "How much of a target the earth's curve hides, refraction k = 0.13.",
        hidden,
    ),
}


def draw(name: str, theme: str) -> None:
    t = THEMES[theme]
    title, subtitle, chart = CHARTS[name]
    with mpl.rc_context(
        {
            "font.family": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 9.5,
            "svg.fonttype": "none",
            "figure.facecolor": t["surface"],
            "axes.facecolor": t["surface"],
            "axes.edgecolor": t["axis"],
            "axes.labelcolor": t["secondary"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": t["grid"],
            "grid.linewidth": 0.8,
            "xtick.color": t["axis"],
            "ytick.color": t["axis"],
            "xtick.labelcolor": t["secondary"],
            "ytick.labelcolor": t["secondary"],
            "legend.frameon": False,
            "legend.labelcolor": t["secondary"],
        }
    ):
        panels = PANELS.get(name, 1)
        fig, axes = plt.subplots(
            1, panels, figsize=(7.2, 3.8), layout="constrained", sharey=True
        )
        axes = list(np.atleast_1d(axes))
        ax = axes[0]
        chart(axes if panels > 1 else ax, t)
        if len(ax.get_legend_handles_labels()[0]) > 1:
            ax.legend(loc="upper left", bbox_to_anchor=(0, -0.2), ncols=4, fontsize=8.5)
        fig.suptitle(
            title, x=0.01, ha="left", color=t["text"], fontsize=12, fontweight="bold"
        )
        ax.set_title(subtitle, loc="left", color=t["secondary"], fontsize=9, pad=10)
        fig.savefig(OUT / f"{name}-{theme}.svg", metadata={"Date": None})
        plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for name in CHARTS:
        for theme in THEMES:
            draw(name, theme)
