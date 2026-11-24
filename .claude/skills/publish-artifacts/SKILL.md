---
name: publish-artifacts
description: >-
  Publishes a seascape recording to W&B as an artifact, with the scenario that made it
  as its input. Use whenever a recording is to reach W&B, wandb, an artifact or
  the registry, even when the request only says upload, push or share it, and to fetch
  or reproduce a published one.
---

# Publish artifacts

```bash
uv run seascape recording scenarios/<x>.toml --parallel --publish --entity sea-ai --project synthetic-recordings
```

One run renders, encodes and exports, then logs artifact `<x>-scenario` (type `scenario`, the resolved `scenario.toml` as `<x>.toml`) as the input of artifact `<x>` (type `recording`, each recording folder as `<rig>/<recording>`). `--name` renames the run and both artifacts. The run also shows the render as it goes: progress, each camera's frame about once a scenario second (not with `--parallel`), the rig-view, the previs views and their videos, then the recording's videos laid out in one. A render's folder publishes the same way: `uv run seascape recording scenarios/<x> --publish ...`.

Prerequisites:

- `uv sync --extra wandb`, then `uv run wandb login`.
- A clean commit: `--publish` refuses a `-dirty` checkout before rendering a scenario, a checkout left `-dirty` by an edit while it rendered, and a render's folder whose labels.json names no commit or a `-dirty` one.

## Collections

An artifact's name is its collection, and each publish under it adds a version: v0, v1, and so on. Content identical to an existing version reuses it. W&B never reuses a number, even after a deletion.

- **One collection per TOML.** Re-render the same TOML, after a fix or on a newer seascape, and publish it again: it becomes the next version of the same collection. A new TOML starts a new collection, so name it for what it tests and unique across the team: `tracker-crossing-ferries`, never `test2`.
- **The scenario versions alongside.** An unchanged TOML reuses its `<x>-scenario` version, so the lineage shows every recording rendered from it.

## Registry

Linking a version into `wandb-registry-dataset/<x>` is a person's call, made by hand in W&B. Check first that `wandb-registry-dataset` accepts type `recording`. The registry counts its own links: `wandb-registry-dataset/<x>:v<m>` can differ from the project's `<x>:v<n>`, and consumers use the registry's.

A consumer fetches an exact version, since `latest` moves with the next publish:

```bash
uvx wandb artifact get --root <dir> wandb-registry-dataset/<x>:v<m>
```

To reproduce one, check out its run's `config.commit`, fetch the `<x>-scenario` it used, and run the command above on that TOML.
