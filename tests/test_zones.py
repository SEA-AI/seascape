"""Thermal zones in the manifest, and the parts they name."""

from typing import get_args

import bpy
import numpy as np
import pytest
from pydantic import ValidationError

from seascape import faces, zones
from seascape.assets import Zones, manifest
from seascape.blender.meshes import measured
from seascape.physics.thermal import ZoneName


def test_every_asset_has_zones_from_the_vocabulary() -> None:
    vocabulary = set(get_args(ZoneName.__value__))
    for name, asset in manifest().items():
        assert asset.zones is not None, name
        given = {*asset.zones.materials.values(), *asset.zones.islands.values()}
        assert given <= vocabulary, name


def test_every_zone_has_a_contact_sheet_colour() -> None:
    assert set(zones.ZONE_COLOURS) == set(get_args(ZoneName.__value__))


DIGEST = "0123456789abcdef"
NUMBERED = {"island_count": 3, "island_digest": DIGEST}


@pytest.mark.parametrize(
    "body",
    [
        {"materials": {"hull": "paint"}},
        {"materials": {"hull": "mixed"}},
        {"materials": {"hull": "deck"}, "islands": {"0": "glass"}},
        {"materials": {"hull": "deck"}, **NUMBERED, "islands": {"3": "glass"}},
        {"materials": {"hull": "deck"}, "island_count": 2, "islands": {"0": "glass"}},
        {"materials": {"hull": "deck"}, **NUMBERED, "islands": {"00": "glass"}},
        {"materials": {"hull": "deck"}, **NUMBERED, "islands": {"+1": "glass"}},
        {"materials": {"hull": "deck"}, **NUMBERED},
        {"materials": {"hull": "deck"}, "island_count": 3},
        {"materials": {"hull": "deck"}, "island_digest": DIGEST},
        {"materials": {"hull": "deck"}, **NUMBERED, "islands": {"0": "glass"}}
        | {"island_digest": "not hex"},
        {"materials": {}},
    ],
)
def test_zones_outside_the_rules_are_refused(body: dict) -> None:
    with pytest.raises(ValidationError):
        Zones.model_validate(body)


def test_an_island_id_reads_from_a_toml_key() -> None:
    body = {"materials": {"hull": "deck"}, **NUMBERED, "islands": {"2": "glass"}}
    assert Zones.model_validate(body).islands == {2: "glass"}


def test_islands_split_at_materials_and_gaps_not_at_split_vertices() -> None:
    """Two quads sharing an edge through duplicated vertices are one island; a third
    of another material on the same edge is its own, and so is a loose one."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    vertices = [
        (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),  # quad a
        (1, 0, 0), (2, 0, 0), (2, 1, 0), (1, 1, 0),  # quad b, its own copy of the edge
        (1, 0, 0), (1, 0, 1), (1, 1, 1), (1, 1, 0),  # quad c, upright on the edge
        (5, 5, 5), (6, 5, 5), (6, 6, 5),  # loose
    ]  # fmt: skip
    polygons = [(0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11), (12, 13, 14)]
    mesh = bpy.data.meshes.new("probe")
    mesh.from_pydata(vertices, [], polygons)
    material = np.array([0, 0, 1, 0])
    label = faces._face_components(mesh, material, np.array([False, False]))
    assert label[0] == label[1]
    assert len({label[0], label[2], label[3]}) == 3


@pytest.mark.render
@pytest.mark.parametrize("name", manifest())
def test_the_import_is_the_mesh_its_zones_were_given_on(name: str) -> None:
    """The triangles and textures the manifest commits, the materials its zones name
    and the islands they number."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    asset = manifest()[name]
    found = zones.parts(name)
    assert measured(name, found.objects) == (asset.triangles, asset.texture_px)
    assert asset.zones is not None
    faces.check(name, found, asset.zones)


@pytest.mark.render
def test_a_view_names_only_parts_and_shows_them_all() -> None:
    """The flat pass, unblended, decodes to part numbers and nothing between, past
    the 2048 where a half float stops counting whole numbers."""
    name = min(manifest(), key=lambda n: manifest()[n].triangles)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    found = zones.parts(name)
    # Top digits 255, the largest a channel holds.
    high = 16_777_000
    views = zones._views(found, [m + high for m in found.face_material])
    seen = set().union(*(np.unique(v.part).tolist() for v in views))
    assert seen == {-1, *range(high, high + len(found.materials))}


def test_a_skinned_part_is_measured_in_its_pose(skinned: bpy.types.Object) -> None:
    heights = zones._heights(faces.found([skinned]), [np.array([0])], 1)

    assert heights[0] == pytest.approx([5.0, 5.0])
