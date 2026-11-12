---
name: sync-scenario
description: Write or change a product's `scenarios/sync-<product>.toml`, the fake recording the sync check sends through Oceanus: its ownship, where the rig sits on it, what turns the cameras and how fast, and the course's buoys and container that `seascape sync` places. Use when a product joins the sync check, when its ownship, mount or turn changes, or when its preview fails.
---

# Sync scenario

`scenarios/sync.toml` holds the turn, sea and outputs every product shares; `sync-<product>.toml` extends it with the ownship, rig and course. The existing product files are the working examples.

Decide by numbers, never by looking at frames. The preview's check is the verdict.

## Steps

1. **Ownship.** Find the hull the product ships on: `uv run seascape assets list`, then `uv run seascape assets show <asset>`. If the request or an existing product file doesn't name it, ask the user, with two to four hulls to pick from, your pick first, each with a one-line reason from those outputs and step 2's top height. If no hull fits, say so.
2. **Mount.** Print the hull's top along its length:

   ```bash
   uv run seascape assets top <asset>
   ```

   On the highest part of its station (a hardtop, a radar arch, a masthead), set `offset_y_m` and `offset_x_m` to that station's `y_m` and `x_m`, and `height_m` a little above its `top_z_m`. Anywhere else, ask the user where the product sits and show them the spot in `uv run seascape rig-view <scenario>`. Name the part and its numbers in a comment.
3. **Turn.** Under `[turn]`: `ptu = "<rig>"` when that rig's pan-tilt unit turns the cameras (its preset sets `ptu`); without it the ownship turns. Set `top_rate_deg_s` and, for an ownship, `accel_deg_s2`, from the product's spec or the user. Leave the moves to `sync.toml`.
4. **Objects.** Run the preview with its `objects` cleared: it fails and prints the course as `[[objects]]` tables, the container then one per ring of buoys, and a pan-tilt unit's inclination. Paste them at the end of the file, under a comment saying `seascape sync` placed them. Whenever a later preview fails, it prints the course again.

   ```bash
   uv run seascape sync scenarios/sync-<product>.toml --preview --set 'objects = []'
   ```

5. **Preview.** Run the preview again. It renders at 1 fps and one sample, and passes only when it prints `every video sees the course`.
6. **Record.** `uv run seascape sync scenarios/sync-<product>.toml`. Compare its file names with a real recording of the product: `<Model>_<video>_record_<ts>.mp4`, one `<Model>_<video>_metadata_<ts>.csv` per video, and `POWER_Const/<model>_sensors.json`.

## When it fails

- **Fails, printing the course:** if it differs from the stored objects, the rig, hull or turn changed. Paste it whole; never edit it by hand.
- **Fails, naming `pitch_deg`:** a fixed mount loses the near ring or the horizon. Set the rig's `pitch_deg` to the value it names.
- **A video misses a buoy or the container:** read the frames at that turn through `seascape.outputs.agc.Agc` for thermal, find what hides it (a crest, the hull, the frame's edge) and fix the cause in the scene or in `seascape/sync/layout.py`'s rule, for every product. Never place one by hand.
