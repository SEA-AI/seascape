"""One EO build, renders at several wear amounts: does a per-object change apply between renders?"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import bpy, weather
from seascape.blender import scene, sea
from seascape.config import load
sc = load(sys.argv[1]); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
weather.install(sc, {})
built = scene.build(sc, "eo", cull=True); sea.remember_foam()
mount = [m for m in sc.mounts if m.camera.band == "eo"][0]; cam = mount.camera
s = bpy.context.scene; s.camera = built.cameras[mount.name]
s.render.resolution_x, s.render.resolution_y = cam.width_px, cam.height_px
for a in [float(x) for x in sys.argv[3].split(",")]:
    for o in bpy.data.objects:
        if "wear_amount" in o: o["wear_amount"] = a; o.update_tag()
    s.render.filepath = str(out / f"amount_{a:.2f}.png"); t0 = time.perf_counter()
    bpy.ops.render.render(write_still=True); print(f"amount {a}: {time.perf_counter()-t0:.1f}s")
