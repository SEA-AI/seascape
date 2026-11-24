---
name: new-scenario
description: >-
  Writes a new scenario under `scenarios/`, checks its mount and what its cameras see on
  `seascape rig-view` and `seascape plan`, and records it. Use to stage any new scene for a product: a test
  for a tracker, a fusion or a detector, or a sweep, `scenarios/sweep-<product>.toml`,
  the turn past moored buoys and a container that the sync test replays to confirm a
  product's frames and IMU rows line up. Also when a sweep's ownship, mount or turn
  changes, or its check fails after a change elsewhere.
---

# New scenario

Start from the shipped scenario nearest the scene and extend it: `extends` keeps the new file a diff, and `docs/scenarios.md` documents every table. Pick its name for what it tests.

## Steps

1. **Ownship.** Find the hull the product ships on: `uv run seascape assets list`, then `uv run seascape assets show <asset>`. If the request or an existing scenario doesn't name it, ask the user, with two to four hulls to pick from, your pick first, each with a one-line reason from those outputs and step 2's top height. If no hull fits, say so.
2. **Mount.** Print the hull's top along its length:

   ```bash
   uv run seascape assets top <asset>
   ```

   On the highest part of its station (a hardtop, a radar arch, a masthead), set `offset_y_m` and `offset_x_m` to that station's `y_m` and `x_m`, and `height_m` just above its `top_z_m`, as the sweep scenarios do; when in doubt, ask the user for the product's mount-to-camera height. Anywhere else, ask the user where the product sits. Name the part and its numbers in a comment.
3. **Objects and motion.** Place `[[objects]]`, a `[turn]` or the ownship's speed for what the scenario tests.
4. **Look.** `uv run seascape rig-view <scenario>` draws the hull with each camera's frustum. `uv run seascape plan <scenario> --video` replays the run from above and from behind the ownship. Check every object the test needs falls in the fields that should see it.
5. **Record.** Commit first, so the render records a commit that exists rather than one marked `-dirty`. `uv run seascape recording <scenario> --parallel` renders the bands at once, encodes and lays the videos out as the rig's `model` records them, in `<scenario>/` beside the TOML; it refuses a folder that isn't empty, so an old run's frames never pass for the new one's.
6. **Publish** with the `publish-dataset` skill.

## Sweeps

`scenarios/sweep.toml` holds the turn, sea and outputs every product shares; `sweep-<product>.toml` extends it with the ownship, rig and course. `seascape.buoyage` places the course: a ring of buoys for each pair of overlapping videos, and a container every video holds. It needs one rig, a turn, and every video looking one way. Trust the check over a frame: a frame can look right at one turn and miss the course at another; the check tests every frame.

1. **Turn.** Under `[turn]`: `ptu = "<rig>"` when that rig's pan-tilt unit turns the cameras (its preset sets `ptu`); without it the ownship turns. Set `top_rate_deg_s` and, for an ownship, `accel_deg_s2`, from the product's spec or the user. Leave the moves to `sweep.toml`.
2. **Objects.** Print the course: the container, then one `[[objects]]` table per ring of buoys, and a pan-tilt unit's elevation. Paste it at the end of the file, under a comment saying `seascape.buoyage` placed it.

   ```bash
   uv run python .claude/skills/new-scenario/scripts/buoyage.py scenarios/sweep-<product>.toml
   ```

3. **Look** on `plan --video`: every ring crosses every field it serves as the turn sweeps it; on `rig-view`, each camera's centre sits between the horizon and the nearest waterline.
4. **Check.** `check.py` renders at 1 fps and one sample into `scenarios/sweep-<product>-preview/`; it passes only when it prints `every video sees the course`.

   ```bash
   uv run python .claude/skills/new-scenario/scripts/check.py scenarios/sweep-<product>.toml
   ```

5. **Record** and **publish** as above, purpose `sync`, the recording's folder as the data:

   ```bash
   uvx --from wandb python .claude/skills/publish-dataset/scripts/upload.py sync scenarios/sweep-<product>/recordings/<rig>/<Model>_recordings_<timestamp>
   ```

   Compare its file names with a real recording of the product: `<Model>_<video>_record_<ts>.mp4`, one `<Model>_<video>_metadata_<ts>.csv` per video, and `POWER_Const/<model>_sensors.json`.

### When the check fails

- **Fails, saying the scenario stores another course:** the rig, hull or turn changed. Replace the stored course with the printed one whole, never edited by hand: the check renders nothing until they are equal.
- **Fails, naming `pitch_deg`:** a fixed mount loses the near ring or the horizon. Set the rig's `pitch_deg` to the value it names.
- **A video misses a buoy or the container:** read the frames at that turn through `seascape.outputs.agc.Agc` for thermal, find what hides it (a crest, the hull, the frame's edge) and fix the cause in the scene or in `seascape/buoyage.py`'s rule, for every product. Never place one by hand: it drifts from that rule.
