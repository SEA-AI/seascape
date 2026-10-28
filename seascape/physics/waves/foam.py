"""Whitecaps: where the sea breaks and how long its foam stays.

Whitecaps: Monahan & O'Muircheartaigh, "Optimal power-law description of oceanic
whitecap coverage dependence on wind speed", Journal of Physical Oceanography 10(12)
2094, 1980, for how much; Snyder & Kennedy, "On the formation of whitecaps by a
threshold mechanism. Part I: Basic formalism", Journal of Physical Oceanography 13(8)
1482, 1983, for where: the downward acceleration past a threshold, set from the
coverage (Tse, McGill & Kelly, SPIE Ocean Optics X 1302, 505, 1990; Tessendorf, Reinhard
& Gao, "Whitecap phenomenology for ocean surface simulation", Clemson University report,
2023). Dupuy & Bruneton, "Real-time animation and rendering of ocean whitecaps",
SIGGRAPH Asia 2012 Technical Briefs 15 (doi:10.1145/2407746.2407761), whose Jacobian
criterion is this acceleration to first order in the choppiness. The foam is remembered
where it was laid, on the water, and its area e-folds in 3.85 s in seawater: Monahan,
"Oceanic whitecaps", Journal of Physical Oceanography 1(2) 139, 1971. A memory reads
every wave its grid resolves; at 0.25 m cells its whitecaps are as wide and as brief as
field video measures: Korinenko & Malinovsky, Physical Oceanography 30(6) 776, 2023;
Korinenko, Malinovsky, Dulov & Kudryavtsev, Fundamentalnaya i Prikladnaya
Gidrofizika 15(1) 61, 2022 (doi:10.48612/fpg/5g5t-4mzd-94ab). Koepke, "Effective
reflectance of oceanic whitecaps", Applied Optics 23(11) 1816, 1984
(doi:10.1364/AO.23.001816), for the 0.22 its reflectance averages over its life, from
fresh foam's (Whitlock et al., in `wakes`).
"""

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import cache
from statistics import NormalDist

import numpy as np

from seascape.physics.waves.slopes import _lattice, slope_variance
from seascape.physics.waves.spectrum import Wave
from seascape.physics.waves.surface import _phase, tallest
from seascape.physics.waves.wind import (
    SLICK_DRIFT,
    _smooth_field,
    gaussian_ranks,
    von_karman_field,
)

# Monahan 1971: a seawater whitecap's area e-folds in 3.85 s.
WHITECAP_AREA_EFOLD_S = 3.85
# Dierssen 2019: thin foam reflects 0.18 of stage-A foam's 0.50 in the visible.
# A judgement: foam dims to that share once its crest stops breaking.
THIN_FOAM_SHARE = 0.18 / 0.50
# A judgement: how much the ragged openings weigh against the cell walls.
FOAM_OPENING_WEIGHT = 0.8
# A judgement: how far, in cells, a cell wall wanders off straight.
FOAM_WALL_WARP = 0.25
# A judgement: the bends' scale, in cells.
FOAM_WALL_BEND = 0.3
# Koepke 1984: a whitecap's effective reflectance in the visible, over its life.
WHITECAP_EFFECTIVE_REFLECTANCE = 0.22
# A judgement: a crest breaks over this share of its acceleration's sigma, which
# frays the foam's edge.
WHITECAP_EDGE = 0.35
# A judgement: the waves that carry the sea's height are the ones that break.
BREAKING_HEIGHT_SHARE = 0.999
# A judgement: the sea's extent, in longest waves and periods, a memory's threshold is
# set over.
SAMPLE_SPAN = 64.0
# The least cover the sea draws, and how far under the threshold a crest lays it.
FOAM_VISIBLE_COVER = 1e-3
FOAM_VISIBLE_Z = NormalDist().inv_cdf(FOAM_VISIBLE_COVER)
# A judgement: a grid draws a wave four of its cells long; under two it aliases.
FOAM_CELLS_PER_WAVE = 4
# A foam memory starts this many area e-folds early, so its start has faded to 1 %.
FOAM_WARM_EFOLDS = math.log(100)
# Numerics: before its last seconds a memory steps coarser, which misses brief crests
# whose foam has since faded.
FOAM_FINE_WARM_S = 6.0
FOAM_COARSE_STEP_S = 0.25
# The steps each point remembers when a memory's threshold is set.
FOAM_REFERENCE_STEPS = 120
# Steps closing in on a threshold: halvings first, then interpolation on log cover,
# until its cover is within 2 %.
FOAM_REFERENCE_ITERATIONS = 12
FOAM_REFERENCE_HALVINGS = 4
FOAM_REFERENCE_TOLERANCE = 0.02
# A judgement: the points a memory's threshold is set over, `SAMPLE_SPAN` across.
FOAM_REFERENCE_SAMPLES = 8192
# Monahan & O'Muircheartaigh 1980, robust biweight fit, U at 10 m.
WHITECAP_COEFFICIENT = 3.84e-6
WHITECAP_EXPONENT = 3.41


def _cell_walls(rng: np.random.Generator, cells: int, period: int) -> np.ndarray:
    """A periodic tile, (cells, cells), of each texel's distance, in cells, from the
    wall between its two nearest of `period`^2 random points: 0 on a wall."""
    points = rng.uniform(0.0, 1.0, (period, period, 2))
    at = (np.arange(cells) + 0.5) * period / cells
    x, y = np.meshgrid(at, at, indexing="xy")
    # Warped, so the walls bend as a film's do instead of running straight.
    x = x + FOAM_WALL_WARP * _smooth_field(rng, cells, FOAM_WALL_BEND * cells / period)
    y = y + FOAM_WALL_WARP * _smooth_field(rng, cells, FOAM_WALL_BEND * cells / period)
    i, j = np.floor(x).astype(int), np.floor(y).astype(int)
    distances = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            ci, cj = (i + di) % period, (j + dj) % period
            dx = i + di + points[cj, ci, 0] - x
            dy = j + dj + points[cj, ci, 1] - y
            distances.append(np.hypot(dx, dy))
    nearest, second = np.sort(np.stack(distances), axis=0)[:2]
    return second - nearest


def foam_lattice_field(
    rng: np.random.Generator,
    cells: int,
    tile_m: float,
    cell_sizes_m: tuple[float, ...],
    opening_length_m: float,
) -> np.ndarray:
    """A periodic tile of aged foam's lattice, ranked onto a unit Gaussian: highest on
    the walls between bubbles, one Voronoi field per entry of `cell_sizes_m`, lowest in
    ragged openings of von Karman's spectrum at `opening_length_m`. A rising threshold
    thins the walls and widens the openings, and what is left stays a network."""
    walls = np.minimum.reduce(
        [_cell_walls(rng, cells, round(tile_m / size_m)) for size_m in cell_sizes_m]
    )
    openings = von_karman_field(rng, cells, tile_m / cells, opening_length_m)
    near = -(walls - walls.mean()) / walls.std()
    return gaussian_ranks(near + FOAM_OPENING_WEIGHT * openings)


def whitecap_fraction(wind_speed_mps: float) -> float:
    return min(WHITECAP_COEFFICIENT * wind_speed_mps**WHITECAP_EXPONENT, 1.0)


def breakers(wind: tuple[Wave, ...]) -> tuple[Wave, ...]:
    return tallest(wind, BREAKING_HEIGHT_SHARE) if wind else ()


def _normal_cdf(z: np.ndarray) -> np.ndarray:
    """The standard normal CDF, within 1.5e-7: erf by Abramowitz & Stegun, Handbook of
    Mathematical Functions, 7.1.26."""
    x = np.abs(z) / math.sqrt(2)
    t = 1 / (1 + 0.3275911 * x)
    poly = t * (
        0.254829592
        + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))
    )
    erf = 1 - poly * np.exp(-x * x)
    return 0.5 * (1 + np.sign(z) * erf)


def _laid(z: np.ndarray) -> np.ndarray:
    """The normal CDF of `z`, zero where it lays less than `FOAM_VISIBLE_COVER`."""
    out = np.zeros(z.shape, np.float32)
    near = z > FOAM_VISIBLE_Z
    out[near] = _normal_cdf(z[near])
    return out


def _memory_edge_g(wind: tuple[Wave, ...]) -> float:
    """`foam_edge_g`, never 0, which a memory divides by."""
    return max(foam_edge_g(wind), 1e-9)


def foam_edge_g(wind: tuple[Wave, ...]) -> float:
    """The spread in g over which a crest goes from whole to broken foam."""
    return WHITECAP_EDGE * math.sqrt(slope_variance(breakers(wind)))


@dataclass(frozen=True)
class FoamGrid:
    """Square cells over the sea, south-west corner first; rows run north."""

    west_m: float
    south_m: float
    cell_m: float
    columns: int
    rows: int


def _resolved(wind: tuple[Wave, ...], cell_m: float) -> tuple[Wave, ...]:
    """The waves a grid of `cell_m` draws: breakers are shorter than the sea's peak
    (Gemmrich et al. 2008), so every one long enough. A judgement: the edge a crest
    breaks over, `foam_edge_g`, is the same on every grid; no source gives its spread.
    """
    return tuple(
        w for w in wind if 2 * math.pi / w.k_rad_m >= FOAM_CELLS_PER_WAVE * cell_m
    )


def _drifting_rad_s(
    field: tuple[Wave, ...], drift_mps: tuple[float, float]
) -> np.ndarray:
    """Each wave's phase rate seen from water drifting at `drift_mps`."""
    return np.array(
        [
            w.k_east_rad_m * drift_mps[0]
            + w.k_north_rad_m * drift_mps[1]
            - w.omega_rad_s
            for w in field
        ]
    )


class _Crests:
    """The downward acceleration of `use`, in g, on a grid that drifts with the
    surface: the water at grid point x is at x + v t."""

    def __init__(
        self, use: tuple[Wave, ...], grid: FoamGrid, drift_mps: tuple[float, float]
    ) -> None:
        x = grid.west_m + (np.arange(grid.columns) + 0.5) * grid.cell_m
        y = grid.south_m + (np.arange(grid.rows) + 0.5) * grid.cell_m
        kx = np.array([w.k_east_rad_m for w in use])
        ky = np.array([w.k_north_rad_m for w in use])
        self.east = np.exp(1j * np.outer(x, kx)).astype(np.complex64)
        self.north = np.exp(1j * np.outer(y, ky)).astype(np.complex64)
        self.ak = np.array([w.amplitude_m * w.k_rad_m for w in use])
        self.phase = np.array([w.phase_rad for w in use])
        self.turn = _drifting_rad_s(use, drift_mps)

    def at(self, t_s: float) -> np.ndarray:
        # Separable in east and north: one matrix product, (rows, columns).
        c = (self.ak * np.exp(1j * (self.phase + self.turn * t_s))).astype(np.complex64)
        return ((self.north * c) @ self.east.T).real


def foam_drift_mps(wind_speed_mps: float, downwind_rad: float) -> tuple[float, float]:
    """(east, north): the surface's drift, which carries foam, as it carries slicks."""
    v = SLICK_DRIFT * wind_speed_mps
    return v * math.sin(downwind_rad), v * math.cos(downwind_rad)


def _memory_times(start_s: float, step_s: float, frames: int) -> list[float]:
    """When a memory steps: coarse until its last fine seconds, then every `step_s`,
    its frames from `start_s`."""
    fine = math.ceil(FOAM_FINE_WARM_S / step_s)
    early = start_s - FOAM_WARM_EFOLDS * WHITECAP_AREA_EFOLD_S
    return [
        *np.arange(early, start_s - fine * step_s, FOAM_COARSE_STEP_S),
        *(start_s + i * step_s for i in range(-fine, frames)),
    ]


def foam_memory(
    wind: tuple[Wave, ...],
    drift_mps: tuple[float, float],
    grid: FoamGrid,
    start_s: float,
    step_s: float,
    frames: int,
    threshold_g: float,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Per frame at start_s + i step_s, (rows, columns): the share of each cell foam
    covers and the share breaking now, on `grid` drifting at `drift_mps`. A crest the
    grid draws lays foam by the normal CDF of its acceleration over `threshold_g`, in
    units of the edge, and the foam fades at Monahan's area e-fold.
    """
    use = _resolved(wind, grid.cell_m)
    crests = _Crests(use, grid, drift_mps)
    edge = _memory_edge_g(wind)
    times = _memory_times(start_s, step_s, frames)
    cover = np.zeros((grid.rows, grid.columns), np.float32)
    last = times[0]
    for t_s in times:
        acceleration = crests.at(t_s)
        rows, columns = np.nonzero(acceleration > threshold_g + FOAM_VISIBLE_Z * edge)
        breaking = np.zeros_like(cover)
        breaking[rows, columns] = _normal_cdf(
            (acceleration[rows, columns] - threshold_g) / edge
        )
        cover = np.maximum(
            cover * math.exp(-(t_s - last) / WHITECAP_AREA_EFOLD_S), breaking
        )
        last = t_s
        if t_s >= start_s - step_s / 2:
            yield cover, breaking


def _reference_accelerations(
    use: tuple[Wave, ...], drift_mps: tuple[float, float], step_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """`use`'s downward acceleration, (steps, points), at points spread `SAMPLE_SPAN`
    longest waves across, each drifting at `drift_mps` from a start of its own; and the
    seconds before each step."""
    longest_m = max(2 * math.pi / w.k_rad_m for w in use)
    slowest_s = max(2 * math.pi / w.omega_rad_s for w in use)
    x, y, t = (SAMPLE_SPAN * _lattice(FOAM_REFERENCE_SAMPLES)).T
    phase = _phase(use, x * longest_m, y * longest_m, t * slowest_s)
    amplitude = np.array([w.amplitude_m * w.k_rad_m for w in use])
    state = (amplitude[:, None] * np.exp(1j * phase)).astype(np.complex64)
    rate = _drifting_rad_s(use, drift_mps)[:, None]
    times = np.array(_memory_times(0.0, step_s, FOAM_REFERENCE_STEPS))
    gaps = np.diff(times, prepend=times[0])
    accelerations = np.empty((len(times), FOAM_REFERENCE_SAMPLES), np.float32)
    for i, gap in enumerate(gaps):
        state *= np.exp(1j * rate * gap).astype(np.complex64)
        accelerations[i] = state.real.sum(axis=0)
    return accelerations, gaps


def _closed_in(
    covers: Callable[[np.ndarray], np.ndarray], wanted: np.ndarray, lo: float, hi: float
) -> np.ndarray:
    """The thresholds between `lo` and `hi` at which `covers` gives `wanted`. Halved
    first; then, where the cover at both ends is known, closed in on by the line
    through log cover at its ends, which falls near straight with the threshold."""
    target = np.log(wanted)
    low, high = np.full(len(wanted), lo), np.full(len(wanted), hi)
    low_cover = np.full(len(wanted), np.nan)
    high_cover = np.full(len(wanted), np.nan)
    for step in range(FOAM_REFERENCE_ITERATIONS):
        mid = (low + high) / 2
        if step >= FOAM_REFERENCE_HALVINGS:
            ends = np.log(low_cover), np.log(high_cover)
            share = np.clip(
                (ends[0] - target) / np.maximum(ends[0] - ends[1], 1e-9), 0.1, 0.9
            )
            known = np.isfinite(share)
            mid = np.where(known, low + np.where(known, share, 0.0) * (high - low), mid)
        cover = covers(mid)
        over = cover > wanted
        low, high = np.where(over, mid, low), np.where(over, high, mid)
        low_cover = np.where(over, cover, low_cover)
        high_cover = np.where(over, high_cover, cover)
        if step + 1 >= FOAM_REFERENCE_HALVINGS and np.all(
            np.abs(np.log(cover) - target) < FOAM_REFERENCE_TOLERANCE
        ):
            return mid
    return (low + high) / 2


@cache
def foam_memory_thresholds_g(
    wind: tuple[Wave, ...],
    drift_mps: tuple[float, float],
    cell_m: float,
    step_s: float,
    fractions: tuple[float, ...],
) -> np.ndarray:
    """The breaking thresholds at which a foam memory at `cell_m` covers each of
    `fractions` of the sea: over points spread `SAMPLE_SPAN` longest waves across, each
    drifting at `drift_mps` and remembering its own foam over `FOAM_REFERENCE_STEPS`
    from a start of its own."""
    wanted = np.array(fractions, dtype=float)
    inside = (wanted > 0.0) & (wanted < 1.0)
    out = np.where(wanted <= 0.0, math.inf, -math.inf)
    use = _resolved(wind, cell_m)
    if not use:
        out[:] = math.inf
    elif inside.any():
        accelerations, gaps = _reference_accelerations(use, drift_mps, step_s)
        edge = _memory_edge_g(wind)
        decays = np.exp(-gaps / WHITECAP_AREA_EFOLD_S)
        steps = len(gaps)

        def covers(thresholds: np.ndarray) -> np.ndarray:
            """Each threshold's mean remembered cover, one pass for them all."""
            remembered = np.zeros((len(thresholds), FOAM_REFERENCE_SAMPLES), np.float32)
            total = np.zeros(len(thresholds))
            levels = thresholds[:, None].astype(np.float32)
            for i, a in enumerate(accelerations):
                laid = _laid((a[None, :] - levels) / edge)
                remembered = np.maximum(remembered * decays[i], laid)
                if i >= steps - FOAM_REFERENCE_STEPS:
                    total += remembered.mean(axis=1)
            return total / FOAM_REFERENCE_STEPS

        # At the low end every point lays all but `FOAM_VISIBLE_COVER` of its foam; at
        # the high end none lays any. Every fraction between is bracketed.
        out[inside] = _closed_in(
            covers,
            wanted[inside],
            float(accelerations.min()) + FOAM_VISIBLE_Z * edge,
            float(accelerations.max()) - FOAM_VISIBLE_Z * edge,
        )
    # A cached array is shared by every caller.
    out.flags.writeable = False
    return out
