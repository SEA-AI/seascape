---
name: publish-artifacts
description: >-
  Publishes a seascape recording to W&B as an artifact, with the scenario that made it
  as its input. Use whenever a recording is to reach W&B, wandb or an artifact,
  even when the request only says upload, push or share it, and to fetch
  or reproduce a published one.
---

# Publish artifacts

```bash
uv run seascape recording scenarios/<x>.toml --parallel --publish --entity sea-ai --project synthetic-recordings
```

One run renders, encodes and exports, then logs artifact `<x>-scenario` (type `scenario`, the resolved `scenario.toml` as `<x>.toml`) as the input of artifact `<x>` (type `recording`, each recording folder as `<rig>/<recording>`). `--name` renames the run and both artifacts. The run also shows the render as it goes: progress, each camera's frame about once a scenario second (not with `--parallel`), the rig-view, the previs videos, then the recording's videos laid out in one. A render's folder publishes the same way: `uv run seascape recording scenarios/<x> --publish ...`.

Prerequisites:

- `uv sync --extra wandb`, then `uv run wandb login`.
- A clean commit: `--publish` refuses a `-dirty` checkout before rendering a scenario, a checkout left `-dirty` by an edit while it rendered, and a render's folder whose labels.json names no commit or a `-dirty` one.

## Collections

An artifact's name is its collection, and each publish under it adds a version: v0, v1, and so on. Content identical to an existing version reuses it. W&B never reuses a number, even after a deletion.

- **One collection per TOML.** Re-render the same TOML, after a fix or on a newer seascape, and publish it again: it becomes the next version of the same collection. The first render's folder is not empty, so give the next its own, with the collection's name: `-o scenarios/<x>-v2 --name <x>`. A new TOML starts a new collection, so name it for what it tests and unique across the team: `tracker-crossing-ferries`, never `test2`.
- **The scenario versions alongside.** An unchanged TOML reuses its `<x>-scenario` version, so the lineage shows every recording rendered from it.

## Fetch

`:latest` moves with the next publish; pin `:v<n>` to hold one:

```bash
uvx wandb artifact get --root <dir> sea-ai/synthetic-recordings/<x>:v<n>
```

To reproduce one, check out its run's `config.commit` and fetch the `<x>-scenario` it used. Render that TOML with `uv run seascape recording <toml>`; where `recording` at that commit takes only a folder, run `uv run seascape render <toml> -o <folder>`, `uv run seascape video <folder>`, then `uv run seascape recording <folder>`. Reproducing publishes nothing, so leave out `--publish`, which older commits lack.
