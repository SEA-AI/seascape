import math

import bpy


def probe(
    lens: bpy.types.Camera | None = None,
    at: tuple[float, float, float] = (0.0, 0.0, 0.0),
    pitch_deg: float = 0.0,
    samples: int = 1,
) -> bpy.types.Object:
    """A camera at `at`, `pitch_deg` from straight down (90 is level)."""
    sc = bpy.context.scene
    camera = bpy.data.objects.new("probe", lens or bpy.data.cameras.new("probe"))
    sc.collection.objects.link(camera)
    camera.location = at
    camera.rotation_euler = (math.radians(pitch_deg), 0.0, 0.0)
    sc.camera = camera
    sc.cycles.samples = samples
    return camera


def show_sea(socket: bpy.types.NodeSocket) -> bpy.types.Node:
    """The sea's Surface swapped for an Emission of `socket`, which is returned."""
    tree = bpy.data.materials["sea"].node_tree
    emission = tree.nodes.new("ShaderNodeEmission")
    tree.links.new(socket, emission.inputs["Color"])
    output = tree.nodes["Material Output"]
    tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return emission
