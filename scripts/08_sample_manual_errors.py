from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_CONDITIONS = [
    "clean",
    "downsample_50",
    "downsample_25",
    "distractor_1",
    "distractor_3",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a fixed-seed, condition-stratified manual-review sheet."
    )
    parser.add_argument("--scored-dir", default="results/scored")
    parser.add_argument(
        "--output",
        default="results/analysis/manual_error_sample_100.csv",
    )
    parser.add_argument("--per-condition", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260724)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.per_condition < 1:
        raise ValueError("--per-condition must be at least 1")

    root = Path(__file__).resolve().parents[1]
    scored_dir = root / args.scored_dir
    files = sorted(scored_dir.glob("robustness__rl__*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No RL robustness shards found under {scored_dir}. "
            "Run scoring before preparing the annotation sample."
        )

    frame = pd.concat(
        [pd.read_csv(path, low_memory=False) for path in files],
        ignore_index=True,
    )
    frame = frame[
        frame["condition"].isin(DEFAULT_CONDITIONS)
        & frame["error"].fillna("").eq("")
        & ~frame["is_correct"].astype(bool)
    ].copy()

    sampled = []
    for offset, condition in enumerate(DEFAULT_CONDITIONS):
        group = frame[frame["condition"].eq(condition)]
        if len(group) < args.per_condition:
            raise ValueError(
                f"{condition} has only {len(group)} eligible errors; "
                f"{args.per_condition} were requested."
            )
        sampled.append(
            group.sample(
                n=args.per_condition,
                random_state=args.seed + offset,
                replace=False,
            )
        )

    output = pd.concat(sampled, ignore_index=True)
    output = output.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    output.insert(0, "annotation_id", [f"E{index + 1:03d}" for index in range(len(output))])
    output["annotator_1_label"] = ""
    output["annotator_1_notes"] = ""
    output["annotator_2_label"] = ""
    output["annotator_2_notes"] = ""
    output["adjudicated_label"] = ""
    output["adjudication_notes"] = ""

    keep = [
        "annotation_id",
        "row_id",
        "sample_id",
        "category",
        "condition",
        "question",
        "choices",
        "answer",
        "extracted_answer",
        "output_text",
        "image_paths",
        "condition_meta",
        "annotator_1_label",
        "annotator_1_notes",
        "annotator_2_label",
        "annotator_2_notes",
        "adjudicated_label",
        "adjudication_notes",
    ]
    output_path = root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output[keep].to_csv(output_path, index=False)
    print(output_path)
    print(f"rows={len(output)} conditions={output['condition'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
