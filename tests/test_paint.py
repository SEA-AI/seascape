"""A hull painted another colour its asset is sold in."""

import bpy
import pytest

from seascape.assets import manifest
from seascape.blender import faces
from seascape.blender.meshes import mesh_objects
from seascape.blender.paint import recolourable
from seascape.blender.vessels import Hulls, add_vessels
from seascape.config import load
from seascape.physics import earth
from seascape.physics.colour import srgb_hex_decode
from tests.scenarios import BASELINE


def test_an_object_takes_its_colour_and_a_copy_of_its_import_keeps_its_own() -> None:
    buoy = 'asset = "mooring_buoy", bearing_deg = 0.0'
    painted = f'{{ {buoy}, range_m = 30.0, colour = "orange" }}'
    own = f"{{ {buoy}, range_m = 40.0 }}"
    scenario = load(BASELINE, [f"objects = [{painted}, {own}]"])
    bpy.ops.wm.read_factory_settings(use_empty=True)
    vessels, _ = add_vessels(
        scenario,
        earth.earth_radius_m(scenario.sea.refraction_k),
        Hulls(scenario.sky, None, scenario.worn),
        [],
        [],
    )
    orange, white = (mesh_objects([a])[0] for a in vessels["mooring_buoy"])

    sold = manifest()["mooring_buoy"].colours["orange"]
    assert tuple(orange["paint"]) == pytest.approx(srgb_hex_decode(sold))
    assert orange["painted"] == 1.0
    assert "painted" not in white  # reads 0: its own Base Color
    # One material between them: the colour is each copy's own.
    assert orange.active_material == white.active_material
    mix = faces.bsdf(orange.active_material).inputs["Base Color"].links[0].from_node
    assert (mix.type, mix.data_type) == ("MIX", "RGBA")
    reads = {
        name: (link.from_node.attribute_type, link.from_node.attribute_name)
        for name in ("Factor", "B")
        for link in mix.inputs[name].links
    }
    assert reads == {"Factor": ("OBJECT", "painted"), "B": ("OBJECT", "paint")}


def test_a_material_to_paint_that_no_mesh_uses_is_refused() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    hull = bpy.data.objects.new("hull", bpy.data.meshes.new("hull"))
    hull.data.materials.append(bpy.data.materials.new("deck"))

    with pytest.raises(ValueError, match=r"materials to paint: \['hull_paint'\]"):
        recolourable([hull], {"hull_paint"})
