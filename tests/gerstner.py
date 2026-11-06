"""Reference surfaces the shader's waves are checked against."""

import math
from dataclasses import replace

import numpy as np

from seascape.physics.waves import (
    Wave,
    gerstner_offset_m,
    gerstner_rest_m,
    visibility,
)
from seascape.physics.waves.spectrum import columns
from seascape.physics.waves.surface import phases


def height_m(
    field: tuple[Wave, ...], east_m: np.ndarray, north_m: np.ndarray, t_s: float
) -> np.ndarray:
    """The surface's height at each point."""
    phase = phases(field, np.asarray(east_m), np.asarray(north_m), t_s)
    return np.tensordot(columns(field)[0], np.cos(phase), axes=1)


def slope(
    field: tuple[Wave, ...], east_m: np.ndarray, north_m: np.ndarray, t_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """The sea's slope: d height / d east and d height / d north."""
    sine = np.sin(phases(field, np.asarray(east_m), np.asarray(north_m), t_s))
    a, _, kx, ky, *_ = columns(field)
    return np.tensordot(-a * kx, sine, axes=1), np.tensordot(-a * ky, sine, axes=1)


def normal(
    field: tuple[Wave, ...], east_m: np.ndarray, north_m: np.ndarray, t_s: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The unit normal of a Gerstner surface at the water at rest: the cross product
    of the surface's derivatives along east and north at rest.

    Args:
        field: Gerstner waves, whose offsets add.
        east_m: Where the water sits on a calm sea, east; a Gerstner wave carries it
            round a circle about there.
        north_m: The same, north.
        t_s: Time; each wave's phase advances at its angular frequency.
    """
    phase = phases(field, np.asarray(east_m), np.asarray(north_m), t_s)
    sine, cosine = np.sin(phase), np.cos(phase)

    a, k, kx, ky, *_ = columns(field)
    # d(offset)/d(rest): -a k_i k_j / k cos along, -a k_j sin up.
    ee = 1 - np.tensordot(a * kx**2 / k, cosine, axes=1)
    nn = 1 - np.tensordot(a * ky**2 / k, cosine, axes=1)
    en = -np.tensordot(a * kx * ky / k, cosine, axes=1)
    ue = -np.tensordot(a * kx, sine, axes=1)
    un = -np.tensordot(a * ky, sine, axes=1)
    # (ee, en, ue) x (en, nn, un)
    normal = np.array([en * un - ue * nn, ue * en - ee * un, ee * nn - en * en])
    normal /= np.linalg.norm(normal, axis=0)
    return normal[0], normal[1], normal[2]


def hit(
    field: tuple[Wave, ...],
    origin_m: np.ndarray,
    direction: np.ndarray,
    t_s: float,
    step_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Where rays first meet a Gerstner surface: the water's rest there, and whether
    they meet it at all before rising clear of the crests. Marched along the sea by
    the ray's height over the water, over the steepest the surface can rise, or by
    the shortest step where that is less; then halved to a micrometre.

    Args:
        field: Gerstner waves, whose offsets add.
        origin_m: Where the rays start, 3 x n: east, north, up.
        direction: The rays' unit directions, 3 x n: east, north, up.
        t_s: Time; each wave's phase advances at its angular frequency.
        step_m: The shortest step along the sea. A crest thinner than it is
            missed.
    """
    origin, direction = np.asarray(origin_m, float), np.asarray(direction, float)
    crest = sum(w.amplitude_m for w in field)
    # The surface's slope is the offset's over its stretch, 1 - sum(a k) at least. A
    # field steeper than 1 has no such bound, and marches at `step_m`.
    steepness = sum(w.amplitude_m * w.k_rad_m for w in field)
    steepest = steepness / (1 - steepness) if steepness < 1 else math.inf
    run = np.hypot(direction[0], direction[1])
    # From where the ray is first as low as a crest.
    start = np.where(
        direction[2] < 0, np.maximum((origin[2] - crest) / -direction[2], 0.0), 0.0
    )
    near, far = start.copy(), start.copy()
    rest_east = origin[0] + start * direction[0]
    rest_north = origin[1] + start * direction[1]
    met = np.zeros(origin.shape[1], bool)
    gap = np.zeros(origin.shape[1])
    # Rising no faster than `steepest` along the sea, the water cannot close the gap
    # within a step.
    closing = np.maximum(steepest * run - direction[2], 1e-6)
    marching = np.arange(origin.shape[1])
    while marching.size:
        i = marching
        step = np.maximum(step_m / np.maximum(run[i], 1e-6), gap[i] / closing[i])
        far[i] = near[i] + step
        at = origin[:, i] + far[i] * direction[:, i]
        # The water under a point moves with it, near enough to start from.
        rest_east[i], rest_north[i] = gerstner_rest_m(
            field,
            at[0],
            at[1],
            t_s,
            (
                rest_east[i] + step * direction[0, i],
                rest_north[i] + step * direction[1, i],
            ),
        )
        up = gerstner_offset_m(field, rest_east[i], rest_north[i], t_s)[2]
        gap[i] = at[2] - up
        below = gap[i] < 0
        met[i] = below
        clear = (at[2] > crest) & (direction[2, i] >= 0)
        sunk = at[2] < -crest
        going = ~below & ~clear & ~sunk
        near[i[going]] = far[i[going]]
        marching = i[going]
    # The surface lies between `near` and `far` on every ray that met it.
    i = np.flatnonzero(met)
    widest = float((far[i] - near[i]).max(initial=1e-6))
    for _ in range(int(np.ceil(np.log2(widest / 1e-6)))):
        middle = (near[i] + far[i]) / 2
        at = origin[:, i] + middle * direction[:, i]
        rest_east[i], rest_north[i] = gerstner_rest_m(
            field, at[0], at[1], t_s, (rest_east[i], rest_north[i])
        )
        up = gerstner_offset_m(field, rest_east[i], rest_north[i], t_s)[2]
        below = at[2] < up
        far[i] = np.where(below, middle, far[i])
        near[i] = np.where(below, near[i], middle)
    return rest_east, rest_north, met


def filtered(field: tuple[Wave, ...], footprint_m: float) -> tuple[Wave, ...]:
    """The waves a pixel of `footprint_m` draws, each scaled by its visibility."""
    shown = visibility(2 * math.pi / columns(field)[1], footprint_m)
    return tuple(
        replace(w, amplitude_m=w.amplitude_m * v)
        for w, v in zip(field, shown.tolist(), strict=True)
    )
