from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score


VALID_LABELS = {"P", "F", "R", "A", "U"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and summarize a completed P/F/R/A/U annotation sheet."
    )
    parser.add_argument(
        "--input",
        default="results/analysis/manual_error_sample_100.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="results/analysis/manual_annotation_summary",
    )
    return parser.parse_args()


def normalized_labels(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.upper()


def require_complete(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        raise ValueError(f"Missing required column: {column}")
    labels = normalized_labels(frame[column])
    invalid = sorted(set(labels) - VALID_LABELS)
    if invalid:
        raise ValueError(
            f"{column} contains blank or invalid labels: {invalid}. "
            f"Allowed labels are {sorted(VALID_LABELS)}."
        )
    return labels


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    input_path = root / args.input
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(input_path, low_memory=False)
    first = require_complete(frame, "annotator_1_label")
    second = require_complete(frame, "annotator_2_label")
    adjudicated = require_complete(frame, "adjudicated_label")

    raw_agreement = float((first == second).mean())
    kappa = float(cohen_kappa_score(first, second, labels=sorted(VALID_LABELS)))
    disagreements = int((first != second).sum())

    summary = {
        "input": str(input_path),
        "rows": int(len(frame)),
        "raw_agreement": raw_agreement,
        "cohen_kappa": kappa,
        "disagreements": disagreements,
        "valid_labels": sorted(VALID_LABELS),
    }
    (output_dir / "reliability.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    counts = (
        pd.DataFrame(
            {
                "condition": frame["condition"].astype(str),
                "category": frame["category"].astype(str),
                "label": adjudicated,
            }
        )
        .groupby(["condition", "category", "label"], dropna=False)
        .size()
        .rename("n")
        .reset_index()
    )
    counts.to_csv(output_dir / "adjudicated_counts.csv", index=False)

    overall = (
        adjudicated.value_counts()
        .reindex(sorted(VALID_LABELS), fill_value=0)
        .rename_axis("label")
        .rename("n")
        .reset_index()
    )
    overall["percentage"] = overall["n"] / len(frame) * 100.0
    overall.to_csv(output_dir / "overall_counts.csv", index=False)

    print(output_dir / "reliability.json")
    print(output_dir / "overall_counts.csv")
    print(output_dir / "adjudicated_counts.csv")


if __name__ == "__main__":
    main()
