"""Publish a render to W&B: a run, its scenario as the input, its data as a dataset."""

import argparse
import json
import tempfile
from pathlib import Path

import tomli_w
import wandb

from seascape.config.loader import resolve

PROJECT, REGISTRY = "synthetic-datasets", "wandb-registry-dataset"


def main() -> None:
    """Upload `data` and the `scenario` it was rendered from."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("purpose", help="what the dataset is for: sync, tracker, ...")
    parser.add_argument("scenario", type=Path, help="the TOML that was rendered")
    parser.add_argument("data", type=Path, help="the folder a consumer downloads")
    parser.add_argument("--commit", help="the commit that rendered it, if unrecorded")
    args = parser.parse_args()

    found = [d / "labels.json" for d in (args.data, *args.data.parents)]
    labels = next((f for f in found if f.is_file()), None)
    if labels is None:
        raise SystemExit(f"no labels.json in {args.data} or above it")
    info = json.loads(labels.read_text())["info"]
    info["commit"] = args.commit or info["commit"]
    if not info["commit"] or info["commit"].endswith("-dirty"):
        raise SystemExit(f"{labels} names no clean commit: pass the one that rendered")

    name = args.scenario.stem
    with (
        tempfile.TemporaryDirectory() as tmp,
        wandb.init(
            project=PROJECT,
            job_type="dataset-upload",
            name=name,
            tags=[args.purpose],
            config=info,
        ) as run,
    ):
        # `extends` and presets resolved: one file renders the scenario on its own.
        standalone = Path(tmp) / f"{name}.toml"
        standalone.write_text(tomli_w.dumps(resolve(args.scenario)))
        scenario = wandb.Artifact(f"{name}-scenario", type="scenario")
        scenario.add_file(str(standalone))
        run.use_artifact(scenario)

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
