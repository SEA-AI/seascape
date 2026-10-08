<h1 align="center">seascape</h1>

<p align="center">
  <em>Synthetic maritime scenes for sensor validation.</em><br>
  Multi-camera rigs · EO and LWIR · exact ground truth
</p>

<p align="center">
  <a href="https://github.com/SEA-AI/seascape/actions/workflows/ci.yml"><img src="https://github.com/SEA-AI/seascape/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/SEA-AI/seascape" alt="License"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/python-3.13-blue" alt="Python 3.13"></a>
  <a href="https://www.blender.org/"><img src="https://img.shields.io/badge/blender-5.2%20LTS-orange" alt="Blender 5.2 LTS"></a>
</p>

<p align="center">
  <img src="docs/hero.jpg" alt="A ship and a buoy under three skies, EO above LWIR, with their ground truth drawn on">
</p>

Real footage can't put a vessel at exactly 7 NM, hold the visibility constant, or show you the same ship from eight aspects. `seascape` renders maritime scenes where you choose all of that, and tells you exactly where everything was.

## Highlights

- **Ground truth by construction.** Camera extrinsics and intrinsics are known rather than estimated. Every frame comes with each target's box, range and bearing, and the horizon.
- **LWIR as well as EO.** Measured seawater optical constants, emissivity averaged over the wave slopes, and a band-integrated sky.
- **Multi-sensor rigs.** Several cameras, each with its own resolution, optics and band, in one scene.
- **Sequences.** Targets under way, ownship roll, pitch and heave, seamless loops, mp4.

> [!NOTE]
> LWIR is good enough to look at and to regression-test against, but not a radiometric reference: path extinction is not modelled, the atmospheric profile is fixed, and waves neither occlude nor shadow each other. A detection-range or contrast figure taken off a render needs review before anyone acts on it.

## Install

```bash
git clone https://github.com/SEA-AI/seascape.git
cd seascape
uv sync
```

Blender ships as a Python package, so there is nothing else to install.

## Quickstart

```bash
uv run seascape render scenarios/baseline.toml -o out/
```

```text
out/
├── bow_eo_0.jpg
├── bow_ir_0.jpg
├── calibration.json
├── labels.json
└── blender.log
```

`seascape build` writes the `.blend` instead, to open in Blender.

A scenario is a TOML file, and `extends` makes a variant a diff:

```toml
extends = "baseline.toml"

[sky]
hdri = "belfast_sunset"

[[objects]]
asset = "lateral_mark"
range_m = 100.0
bearing_deg = -6.0
```

<details>
<summary>Tab completion</summary>

With `seascape` on the `PATH`, add to `~/.zshrc`:

```bash
eval "$(_SEASCAPE_COMPLETE=zsh_source seascape)"
```

For bash, `bash_source` in `~/.bashrc`.

</details>

## Documentation

| | |
|---|---|
| [How seascape works](docs/primer/) | a tour for anyone new to Blender: rendering, HDR, EO and LWIR, the sea, limitations |
| [Scenarios](docs/scenarios.md) | `--set`, `extends`, skies, sequences |
| [Outputs](docs/outputs.md) | labels, LWIR radiometry, montage, panorama |
| [Assets](docs/assets.md) | the meshes, where they come from, adding one |
| [Render server](docs/render-server.md) | render from Claude, on a shared GPU |
| [Blender MCP](docs/blender-mcp.md) | an AI agent in the open Blender scene |
| [Contributing](CONTRIBUTING.md) | checks, render tests, conventions |

## Licence

MIT. Meshes and skies carry their own licences, in `seascape/assets.toml` and `seascape/skies.toml`; some require attribution, which travels with any released dataset. `seascape/data/water_nk.csv` is CC BY 4.0, from [Nalli et al. 2022](https://doi.org/10.6084/m9.figshare.19341533).
