Reference prototype for procedural wear (not production, not for merge).
weather.py monkeypatches the build: per-face zone profiles, hull-frame
position/normal, depth below the edge above, a `wear_masks` node group,
EO dirt as a second BSDF, LWIR worn zone skins. Constants are eyeballed.
