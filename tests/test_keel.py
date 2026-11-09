import numpy as np
import pytest

from seascape.blender import meshes


def plane(*corners: tuple[float, float, float]) -> np.ndarray:
    """Two triangles over four corners in order."""
    a, b, c, d = corners
    return np.array([[a, b, c], [a, c, d]], dtype=float)


def v_hull(length_m: float, half_beam_m: float, draught_m: float) -> np.ndarray:
    """A V section, keel at z = -draught, sides rising to the waterline at the half
    beam and on, upright, to a metre above it; open at the ends and the deck.
    """
    y0, y1 = -length_m / 2, length_m / 2
    sides = []
    for x in (-half_beam_m, half_beam_m):
        sides.append(
            plane((0, y0, -draught_m), (x, y0, 0), (x, y1, 0), (0, y1, -draught_m))
        )
        sides.append(plane((x, y0, 0), (x, y0, 1), (x, y1, 1), (x, y1, 0)))
    return np.concatenate(sides)


def box(facets: np.ndarray, rise_m: float = 0.0) -> tuple[float, float]:
    keel, cell_m2 = meshes.keel_m(facets)
    return meshes.box_m(keel, cell_m2, float(np.ptp(facets[:, :, 1])), rise_m)


def test_a_v_hull_s_box_has_its_waterline_beam_and_half_its_draught() -> None:
    """Volume 8 m2 a metre over a waterplane 8 m wide: a box 8 m by 1 m."""
    assert box(v_hull(40.0, 4.0, 2.0)) == pytest.approx((8.0, 1.0))


def test_a_lighter_hull_s_box_is_its_hull_below_the_lower_waterline() -> None:
    """Risen a metre, the V is 4 m wide at the water and 1 m deep."""
    assert box(v_hull(40.0, 4.0, 2.0), rise_m=1.0) == pytest.approx((4.0, 0.5))


def test_an_inner_skin_and_a_boom_over_the_side_leave_the_box_alone() -> None:
    hull = v_hull(40.0, 4.0, 2.0)
    inner = v_hull(40.0, 4.0, 2.0) + np.array([0.0, 0.0, 0.3])
    boom = plane((-8, -1, 2), (8, -1, 2), (8, 1, 2), (-8, 1, 2))

    facets = np.concatenate([hull, inner, boom])

    assert box(facets) == pytest.approx((8.0, 1.0))


def test_a_hull_with_nothing_below_the_waterline_has_no_box() -> None:
    deck = plane((-1, -5, 1), (1, -5, 1), (1, 5, 1), (-1, 5, 1))

    assert box(deck) == (0.0, 0.0)
