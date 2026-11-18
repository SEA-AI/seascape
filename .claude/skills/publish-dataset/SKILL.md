---
name: publish-dataset
description: >-
  Publishes a seascape render to W&B as a dataset, in the `synthetic-datasets` project and
  the Dataset registry, with the scenario that made it as one standalone TOML. Use whenever
  a render is to reach W&B, wandb, an artifact or the registry, even when the request only
  says upload, push or share a render: after rendering anything another team consumes (a
  sync recording, a tracker or fusion scenario), to publish a re-render of the same TOML,
  or to fetch or reproduce a published dataset.
---

# Publish a dataset

| What | Where |
|---|---|
| The upload | a run in the `synthetic-datasets` project, named `<stem>`, `job_type` `dataset-upload` as the registry's other uploads, tagged with the purpose (`sync`, `tracker`, …), `config` = `labels.json`'s `info` |
| The scenario | the run's input, artifact `<stem>-scenario` of type `scenario`: the `scenario.toml` the render wrote beside `labels.json`, with `extends`, presets and `--set` resolved, which renders on its own |
| The data | the run's output, artifact `<stem>` of type `dataset`, linked into `wandb-registry-dataset/<stem>` |

`<stem>` is the render's folder name, which defaults to the scenario file's name without `.toml`.

## Collections

An artifact's name is its collection, and each upload under it adds a version: v0, v1, and so on. Content identical to an existing version reuses it. W&B never reuses a number, even after a deletion.

- **One collection per TOML.** Re-render the same TOML, after a fix or on a newer seascape, and publish it again: it becomes the next version of the same collection. A new TOML starts a new collection, so name it for what it tests and unique across the team: `tracker-crossing-ferries`, never `test2`.
- **The scenario versions alongside.** An unchanged TOML reuses its `<stem>-scenario` version, so the lineage shows every dataset rendered from it:

  | Upload | Scenario | Dataset |
  |---|---|---|
  | first render | `<stem>-scenario:v0` | `<stem>:v0` |
  | re-render, same TOML | `<stem>-scenario:v0` | `<stem>:v1` |
  | TOML edited, rendered again | `<stem>-scenario:v1` | `<stem>:v2` |
- **The registry counts its own links.** `wandb-registry-dataset/<stem>:v<m>` can differ from the project's `<stem>:v<n>`. Consumers use the registry's.

A TOML a test reads stays in `scenarios/`, as the sync ones do. A one-off dataset's TOML needs no home in the repo: its scenario artifact is that home.

## Steps

1. **Render from a clean commit**, so `info.commit` names code that exists: a checkout, or the image at a main commit's tag, `docker run --gpus all -v <out>:/out ghcr.io/sea-ai/seascape:<sha> render <scenario> -o /out/<stem>`. `seascape render` and `sync` write `scenario.toml` beside `labels.json`: the scene as rendered, overrides included.
2. **Upload**, logged in (`uvx wandb login`) with `WANDB_ENTITY` set to the team that owns the registry:

   ```bash
   uvx --from wandb python .claude/skills/publish-dataset/scripts/upload.py <purpose> <data>
   ```

   `<data>` is the folder a consumer downloads: a sync recording's folder, or a render's `-o` folder. The script finds `labels.json` and `scenario.toml` in it or above it, so it needs only the output folder, not seascape: run it where the render is. `--name` sets the collection when the folder's name isn't the scenario's. A render without `.git` records no commit; pass `--commit <sha>` of the code it rendered. It prints the project's version and the registry's.

## Fetch and reproduce

A consumer fetches the registry's exact version: `latest` moves to the next upload, so a result pinned to it changes under the consumer.

```bash
uvx wandb artifact get --root <dir> wandb-registry-dataset/<stem>:v<m>
```

To reproduce one, check out its run's `config.commit` (or its seascape version), fetch the `<stem>-scenario` it used, and render that TOML with the purpose's command.
