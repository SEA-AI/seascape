# Scenarios

A scenario is a TOML file describing the world, the platform, the sensors and the targets. `scenarios/baseline.toml` is the smallest one.

`--set` overrides any field for one run, as the TOML line it would be written as, presets included:

```bash
uv run seascape render scenarios/twin-pod.toml --set 'rig.pitch_deg = -5'
uv run seascape render scenarios/baseline.toml --set 'outputs.samples.eo = 8' --set 'sky.sun_elevation_deg = 5'
uv run seascape render scenarios/twin-pod.toml --set 'rig.pods = [{ preset = "port" }]'   # one pod
```

A variant worth keeping is a file, and `extends` makes it a diff:

```toml
extends = "twin-pod.toml"

[rig]
pitch_deg = -5.0
```

Scenarios carry a `#:schema` line, so editors with a TOML language server give you key completion, inline validation and hover docs. `seascape schema > schema/scenario.json` regenerates it from the models.

## Skies

By default the sky is Blender's Sky Texture: clear, at any sun elevation down to twilight. `sky.hdri` puts a photographed sky in its place, one of Poly Haven's pure skies listed in [`seascape/skies.toml`](../seascape/skies.toml), fetched into the asset cache on first use. The photo turns so its sun sits at `sun_bearing_deg`. Its elevation is the one it was photographed at, so a scenario with `hdri` leaves `sun_elevation_deg` and `aerosol_density` out. LWIR keeps its own clear sky and takes the photo's sun and clouds, each cloud a blackbody as warm as the air at `cloud_base_m`. A photo's radiance is in its own exposure, so an exr build compares only with builds of the same sky.

```bash
uv run seascape render scenarios/baseline.toml --set 'sky.hdri = "table_mountain_1"' --set 'sky.sun_bearing_deg = 20'
```

<p align="center">
  <img src="skies.jpg" alt="Every photographed sky, highest sun first, with its measured sun circled">
  <br>
  <sub>The library, highest sun first, each sun circled where <code>seascape.skies</code> measures it (<a href="skies.py"><code>docs/skies.py</code></a>).</sub>
</p>

## Sequences

`outputs.duration_s` turns a scenario into a clip. Targets make `speed_mps` along their heading, the ownship follows `[ownship.roll]`, `[ownship.pitch]` and `[ownship.heave]`, and the waves, and the gusts that roughen them, run downwind from `sea.wind_from_deg`. `scenarios/underway.toml` has all three:

```bash
uv run seascape render scenarios/underway.toml -o out/   # out/<camera>/0000.jpg, ...
uv run seascape video out/                               # out/<camera>.mp4
uv run seascape recording out/                           # out/recordings/<pod>/Pod_recordings_<ts>/
```

`outputs.loop = true` makes a seamless clip: every period, each wave's included, rounds to a whole fraction of `duration_s`. A looping target cannot be underway; give it a `drift`, as `scenarios/drifting.toml` does.

Every frame has its own entry in `calibration.json` and `labels.json`, stamped with `time_s`. `seascape video` paces the frames by it, so a clip re-encodes without re-rendering. The `.blend` from `seascape build` carries the motion as keyframes. `montage` and `panorama` take stills.
