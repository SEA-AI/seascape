---
name: sync-scenario
description: >-
  Writes, records and uploads a product's sync scenario, `scenarios/sweep-<product>.toml`: a rendered
  recording that Oceanus's sync check replays to confirm the product's frames and IMU rows line up.
  Use to add a product to the sync check, or to render and upload its recording to W&B. Also when
  its ownship, mount or turn changes, or `seascape sync --preview` fails after a change elsewhere.
---

# Sync scenario

`scenarios/sweep.toml` holds the turn, sea and outputs every product shares; `sweep-<product>.toml` extends it with the ownship, rig and course. The existing product files are the working examples.

Decide by the preview's printed check: a frame looks right at one turn and misses the course at another, and the preview tests every frame.

## Steps

1. **Ownship.** Find the hull the product ships on: `uv run seascape assets list`, then `uv run seascape assets show <asset>`. If the request or an existing product file doesn't name it, ask the user, with two to four hulls to pick from, your pick first, each with a one-line reason from those outputs and step 2's top height. If no hull fits, say so.
2. **Mount.** Print the hull's top along its length:

   ```bash
   uv run seascape assets top <asset>
   ```

   On the highest part of its station (a hardtop, a radar arch, a masthead), set `offset_y_m` and `offset_x_m` to that station's `y_m` and `x_m`, and `height_m` just above its `top_z_m`, as the shipped sync scenarios do; when in doubt, ask the user for the product's mount-to-camera height. Anywhere else, ask the user where the product sits and show them the spot in `uv run seascape rig-view <scenario>`. Name the part and its numbers in a comment.
3. **Turn.** Under `[turn]`: `ptu = "<rig>"` when that rig's pan-tilt unit turns the cameras (its preset sets `ptu`); without it the ownship turns. Set `top_rate_deg_s` and, for an ownship, `accel_deg_s2`, from the product's spec or the user. Leave the moves to `sweep.toml`.
4. **Objects.** Run the preview with its `objects` cleared: it fails and prints the course as `[[objects]]` tables, the container then one per ring of buoys, and a pan-tilt unit's elevation. Paste them at the end of the file, under a comment saying `seascape sync` placed them. Whenever a later preview fails, it prints the course again.

   ```bash
   uv run seascape sync scenarios/sweep-<product>.toml --preview --set 'objects = []'
   ```

5. **Sheet.** Check the course on `uv run seascape sync scenarios/sweep-<product>.toml --sheet`: every ring crosses every video it serves, and each camera's centre sits between the horizon and the nearest waterline.
6. **Preview.** Run the preview again. It renders at 1 fps and one sample, and passes only when it prints `every video sees the course`.
7. **Record.** Commit first, so the render records a commit that exists rather than one marked `-dirty`. To record a product again, move its earlier run out of `scenarios/sweep-<product>/` first: `seascape sync` refuses a folder that isn't empty, so an old run's frames never pass for the new one's. `uv run seascape sync --parallel scenarios/sweep-<product>.toml` renders the bands at once, encodes and writes `recordings/<rig>/<Model>_recordings_<timestamp>/`. Compare its file names with a real recording of the product: `<Model>_<video>_record_<ts>.mp4`, one `<Model>_<video>_metadata_<ts>.csv` per video, and `POWER_Const/<model>_sensors.json`.
8. **Publish** with the `publish-dataset` skill, from the commit that recorded it, purpose `sync`, the recording's folder as the data:

   ```bash
   uvx --from wandb python .claude/skills/publish-dataset/scripts/upload.py sync scenarios/sweep-<product>/recordings/<rig>/<Model>_recordings_<timestamp>
   ```

## When it fails

- **Fails, printing the course:** if it differs from the stored objects, the rig, hull or turn changed. Paste it whole, never edited by hand: the preview fails unless the stored course equals what `seascape sync` computes.
- **Fails, naming `pitch_deg`:** a fixed mount loses the near ring or the horizon. Set the rig's `pitch_deg` to the value it names.
- **A video misses a buoy or the container:** read the frames at that turn through `seascape.outputs.agc.Agc` for thermal, find what hides it (a crest, the hull, the frame's edge) and fix the cause in the scene or in `seascape/buoyage.py`'s rule, for every product. Never place one by hand: it drifts from that rule.
