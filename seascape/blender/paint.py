"""A hull's paint in another colour its asset is sold in, per object, so copies of
one import share its materials and each keeps its own colour.
"""

from collections.abc import Iterable, Set

import bpy

from seascape.blender import faces
from seascape.blender.blend import attribute, feed, mesh_objects, node, source
from seascape.physics.colour import srgb_hex_decode


def recolourable(meshes: Iterable[bpy.types.Object], materials: Set[str]) -> None:
    """Let `paint` set the Base Color of `meshes`' `materials`, per object and copy.

    Limitation: the colour replaces a textured Base Color, its detail too.
    """
    used = {s.material for o in meshes for s in o.material_slots} - {None}
    if missing := materials - {m.name for m in used}:
        raise ValueError(f"no mesh uses the materials to paint: {sorted(missing)}")
    for material in (m for m in used if m.name in materials):
        if (shader := faces.bsdf(material)) is None:
            raise ValueError(
                f"material {material.name} has no Principled BSDF to paint"
            )
        t = material.node_tree
        mix = node(t, "ShaderNodeMix", data_type="RGBA")
        feed(
            t, mix.inputs["Factor"], attribute(t, "painted", "OBJECT").outputs["Factor"]
        )
        feed(t, mix.inputs["A"], source(shader.inputs["Base Color"]))
        feed(t, mix.inputs["B"], attribute(t, "paint", "OBJECT").outputs["Color"])
        t.links.new(mix.outputs["Result"], shader.inputs["Base Color"])


def paint(anchor: bpy.types.Object, srgb_hex: str) -> None:
    """Paint a recolourable hull's meshes `srgb_hex`."""
    linear = list(srgb_hex_decode(srgb_hex))
    for part in mesh_objects([anchor]):
        part["paint"] = linear
        # An object without it reads 0: its own Base Color.
        part["painted"] = 1.0
