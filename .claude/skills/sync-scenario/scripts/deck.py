"""A hull's top per station along it, in the vessel frame a rig's `offset_y_m`,
`offset_x_m` and `height_m` use: y forward of midships, z above the waterline.
"""

import argparse

import bpy
import numpy as np

from seascape.blender import meshes


def tops(name: str, step_m: float) -> list[tuple[float, float, float]]:
    """(y, top z, x at the top) per `step_m` of length, stern first: the highest
    vertex in it, or face that a ray cast straight down through its middle meets.
    Vertices catch thin rails and tubes that rays slip past; rays catch a broad roof
    with no vertex inside a station.
    """
    bpy.ops.wm.read_factory_settings(use_empty=True)
    parts = meshes.import_asset(name, "eo")
    bpy.context.view_layer.update()
    found = []
    for mesh in meshes.mesh_objects(parts):
        local = np.empty(len(mesh.data.vertices) * 3)
        mesh.data.vertices.foreach_get("co", local)
        world = np.array(mesh.matrix_world)
        found.append(local.reshape(-1, 3) @ world[:3, :3].T + world[:3, 3])
    x, y, z = np.concatenate(found).T
    depsgraph = bpy.context.evaluated_depsgraph_get()
    rows = []
    for start in np.arange(y.min(), y.max(), step_m):
        middle_m = start + step_m / 2
        inside = (y >= start) & (y < start + step_m)
        hits = (
            [(z[inside].max(), x[inside][z[inside].argmax()])] if inside.any() else []
        )
        for across_m in np.linspace(x.min(), x.max(), 41):
            hit, at, *_ = bpy.context.scene.ray_cast(
                depsgraph, (across_m, middle_m, z.max() + 1.0), (0.0, 0.0, -1.0)
            )
            if hit:
                hits.append((at.z, across_m))
        if hits:
            top_m, at_m = max(hits)
            rows.append((middle_m, top_m, at_m))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Pick the station where the product is installed (a hardtop, a radar "
        "arch, a mast): offset_y_m = y_m, offset_x_m = x_m, height_m a little above "
        "top_z_m.",
    )
    parser.add_argument(
        "asset", help="A hull in assets.toml; `seascape assets list` names them."
    )
    parser.add_argument(
        "--step-m",
        type=float,
        default=0.5,
        help="Length of each station along the hull, in metres (default: %(default)s).",
    )
    args = parser.parse_args()
    print("y_m     top_z_m  x_m")
    for y_m, z_m, x_m in tops(args.asset, args.step_m):
        print(f"{y_m:6.2f}  {z_m:6.2f}  {x_m:6.2f}")
