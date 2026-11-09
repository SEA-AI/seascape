import bpy
import numpy as np
import pytest

from seascape.blender import meshes


def sheet(
    y_m: tuple[float, float], z_m: tuple[float, float], x_m: float = 1.0
) -> np.ndarray:
    """Two triangles from x = -`x_m` to `x_m`, rising from z_m[0] at y_m[0] to z_m[1]
    at y_m[1]: no corner between the two ends.
    """
    (y0, y1), (z0, z1) = y_m, z_m
    a, b = (-x_m, y0, z0), (x_m, y0, z0)
    c, d = (x_m, y1, z1), (-x_m, y1, z1)
    return np.array([[a, b, c], [a, c, d]], dtype=float)


def test_a_broad_roof_with_no_corner_in_a_station_still_tops_it() -> None:
    hull = np.concatenate([sheet((-4, 4), (0, 0)), sheet((-2, 2), (2, 2))])

    rows = meshes.top_per_station(hull, step_m=1.0)

    tops = {y: z for y, z, _ in rows}

    assert tops == {-3.5: 0, -2.5: 2, -1.5: 2, -0.5: 2, 0.5: 2, 1.5: 2, 2.5: 2, 3.5: 0}
    assert all(x == 0 for _, _, x in rows)


def test_a_ramp_tops_where_it_leaves_the_station_not_at_its_middle() -> None:
    ramp = sheet((0, 4), (0, 4))

    tops = [z for _, z, _ in meshes.top_per_station(ramp, step_m=1.0)]

    assert tops == pytest.approx([1, 2, 3, 4])


def test_a_tube_thinner_than_a_station_is_found_with_its_x() -> None:
    deck = sheet((0, 2), (0, 0))
    tube = np.array([[[0.3, 0.5, 3.0], [0.301, 0.5, 3.0], [0.3, 0.501, 2.9]]])

    rows = meshes.top_per_station(np.concatenate([deck, tube]), step_m=1.0)

    assert rows[0] == pytest.approx((0.5, 3.0, 0.3), abs=1e-3)


@pytest.mark.usefixtures("empty_blend")
def test_triangles_are_in_world_metres() -> None:
    bpy.ops.mesh.primitive_cube_add(location=(0, 5, 1), scale=(1, 2, 0.5))
    bpy.context.view_layer.update()

    found = meshes.triangles([bpy.context.object])

    assert found.shape == (12, 3, 3)
    assert found.min(axis=(0, 1)) == pytest.approx([-1, 3, 0.5])
    assert found.max(axis=(0, 1)) == pytest.approx([1, 7, 1.5])


def test_a_post_at_the_stern_or_bow_is_inside_the_last_station() -> None:
    """0.9 / 0.3 rounds to just under 3 stations."""
    deck = sheet((0, 0.9), (0, 0))
    post = np.array([[[0, 0.9, 0], [0.1, 0.9, 0], [0, 0.9, 3]]], dtype=float)

    rows = meshes.top_per_station(np.concatenate([deck, post]), step_m=0.3)

    assert rows[-1][1] == 3


def test_a_station_longer_than_the_hull_stays_on_it() -> None:
    rows = meshes.top_per_station(sheet((0, 1), (0, 0)), step_m=5.0)

    assert [y for y, _, _ in rows] == [0.5]
