---
name: sync-scenario
description: Write or change a product's `scenarios/sync-<product>.toml`, the fake recording the sync check sends through Oceanus: its ownship, where the rig sits on it, and the containers every recorded video must keep whole. Use when a product joins the sync check, when its ownship, mount or containers change, or when `seascape sync --preview` fails.
---

# Sync scenario

A sync scenario renders a product's recording with containers that every recorded video
sees whole in every frame, while the ownship rocks. `scenarios/sync.toml` holds the sea,
sky, seed and outputs; `sync-<product>.toml` extends it with the ownship, the rig and
the containers. The existing product files are the working examples.

Decide by numbers, never by looking at frames. The preview's check is the verdict.

## Steps

1. **Ownship.** Pick the hull the product ships on, by kind and length:
   `uv run seascape assets list`, then `uv run seascape assets show <asset>`.
   If no hull fits, say so instead of stretching one.
2. **Mount.** Print the hull's top along its length, in the frame the rig uses:

   ```bash
   uv run seascape assets profile <asset>
   ```

   Find where the product is installed on such a boat (a hardtop, a radar arch, a
   mast). Set `offset_y_m` to that station's `y_m`, `offset_x_m` to its `x_m`, and
   `height_m` a little above its `top_z_m`. Name the part and its numbers in a comment.
3. **Containers.** One `[[objects]]` per group of cameras that share a bearing, on that
   bearing (`bearing_deg` is from midships, clockwise), broadside: `heading_deg = 90`
   with `heading_from = "line_of_sight"`. Choose the range from the narrowest camera
   that must see it, with `L` the container's `length_m` in `assets.toml`:
   - whole across its field: `range_m > L / (2 tan(hfov / 2))`;
   - the frame centre looks `atan(height_m / range_m)` above the container. Keep that
     well under half the camera's vertical field, which the hull's pitch eats into.
4. **Preview.** It renders the clip's full span at 1 fps, a quarter the size and one
   sample, then checks the labels, in a few minutes:

   ```bash
   uv run seascape sync scenarios/sync-<product>.toml --preview
   ```

   It passes only when it prints `every video keeps its containers whole`.
5. **Record.** `uv run seascape sync scenarios/sync-<product>.toml` renders, encodes and
   writes `recordings/<rig>/<Model>_recordings_<timestamp>/`. Compare its file names
   with a real recording of the product: `<Model>_<video>_record_<ts>.mp4`, one
   `<Model>_<video>_metadata_<ts>.csv` per video, and
   `POWER_Const/<model>_sensors.json`. The video names come from the rig preset's
   `[videos]`.

## When the preview fails

The error names the recorded video, the time and what went wrong.

- **Touches the frame's edge, or fewer containers than at first:** move the container
  farther along its camera's bearing, raise the range so the frame centre looks down
  less, or calm the scene: `sea.wind_speed_mps` or `ownship.yaw`. Never loosen the check.
- **No container:** its bearing misses that video, or the range is past its reach.
- **A wide video sees a neighbour's container cut at its edge:** move that container
  out of the frame or fully into it. A cut container fails the check whatever camera it
  was meant for.

## Pitfalls

- Video names in errors are the recorded videos (`mast_T1`, `mast_RGBwFoV`), not the
  rig's raw cameras. A stitched video holds both of its cameras' containers.
- A rig lower than the waves' significant height looks into the crests.
- `bearing_deg` and `range_m` are from midships, not from the rig. Far off, the
  difference vanishes; at tens of metres, a rig metres aft shifts the bearing.
- The 1 fps preview steps over the peaks between its frames. Leave margin rather than
  settling for a pass by a pixel.
- A thermal png is raw 16-bit kelvin: run it through `seascape.outputs.agc.Agc` before
  showing it to a person.
