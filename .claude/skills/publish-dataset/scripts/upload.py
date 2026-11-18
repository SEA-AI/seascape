"""Publish a render to W&B: a run, its scenario as the input, its data as a dataset."""

import argparse
import json
from pathlib import Path

import wandb

PROJECT, REGISTRY = "synthetic-datasets", "wandb-registry-dataset"


def main() -> None:
    """Upload `data` and the scenario its render wrote beside its labels."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("purpose", help="what the dataset is for: sync, tracker, ...")
    parser.add_argument("data", type=Path, help="the folder a consumer downloads")
    parser.add_argument("--name", help="the collection; default: the render's folder")
    parser.add_argument("--commit", help="the commit that rendered it, if unrecorded")
    args = parser.parse_args()

    found = [d / "labels.json" for d in (args.data, *args.data.parents)]
    labels = next((f for f in found if f.is_file()), None)
    if labels is None:
        raise SystemExit(f"no labels.json in {args.data} or above it")
    # `seascape render` and `sync` write it: what rendered, never a copy made later.
    scenario = labels.with_name("scenario.toml")
    if not scenario.is_file():
        raise SystemExit(f"no {scenario}: render with a seascape that writes it")
    info = json.loads(labels.read_text())["info"]
    info["commit"] = args.commit or info["commit"]
    if not info["commit"] or info["commit"].endswith("-dirty"):
        raise SystemExit(f"{labels} names no clean commit: pass the one that rendered")

    name = args.name or labels.parent.name
    with wandb.init(
        project=PROJECT,
        job_type="dataset-upload",
        name=name,
        tags=[args.purpose],
        config=info,
    ) as run:
        source = wandb.Artifact(f"{name}-scenario", type="scenario")
        source.add_file(str(scenario), name=f"{name}.toml")
        run.use_artifact(source)

        dataset = wandb.Artifact(
            name,
            type="dataset",
            description=f"seascape {info['version']} at {info['commit']}",
            metadata=info,
        )
        dataset.add_dir(str(args.data), name=args.data.name)
        run.log_artifact(dataset).wait()
        linked = dataset.link(f"{REGISTRY}/{name}")
        print(dataset.qualified_name, "->", linked.qualified_name)


if __name__ == "__main__":
    main()
