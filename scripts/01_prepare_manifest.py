from __future__ import annotations

import argparse
from pathlib import Path

from r1eval.data import load_benchmark, make_manifests
from r1eval.io_utils import load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_json(root / args.config)
    dataset = load_benchmark(
        config["dataset_id"],
        config["dataset_split"],
        root / "data" / "cache",
    )
    subset_sizes = {
        phase["subset_size"]
        for phase in config["phases"].values()
        if phase["subset_size"] is not None
    }
    if len(subset_sizes) != 1:
        raise ValueError(
            "This project expects one shared robustness subset size; "
            f"found {sorted(subset_sizes)}"
        )
    subset_size = subset_sizes.pop()
    all_path, subset_path = make_manifests(
        dataset,
        root / "data" / "manifests",
        subset_size=subset_size,
        seed=int(config["seed"]),
    )
    print(f"dataset rows: {len(dataset)}")
    print(f"all manifest: {all_path}")
    print(f"subset manifest: {subset_path}")


if __name__ == "__main__":
    main()

