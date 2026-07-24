from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from r1eval.scoring import normalize_text


GROUP_COLUMNS = [
    "phase",
    "model_key",
    "model_id",
    "condition",
    "prompt_mode",
    "image_mode",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored-dir", default="results/scored")
    parser.add_argument("--output-dir", default="results/analysis")
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260723)
    return parser.parse_args()


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    repeats: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if len(values) == 0:
        return np.nan, np.nan
    draws = rng.choice(values, size=(repeats, len(values)), replace=True)
    means = draws.mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def build_summary(
    frame: pd.DataFrame,
    *,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for group_values, group in frame.groupby(GROUP_COLUMNS, dropna=False):
        correctness = group["is_correct"].astype(float).to_numpy()
        low, high = bootstrap_mean_ci(correctness, repeats=repeats, rng=rng)
        error_mask = group["error"].fillna("").astype(str).str.len() > 0
        valid_group = group.loc[~error_mask]
        row = dict(zip(GROUP_COLUMNS, group_values))
        row.update(
            {
                "n": len(group),
                "accuracy": float(correctness.mean()),
                "valid_n": int((~error_mask).sum()),
                "valid_accuracy": (
                    float(valid_group["is_correct"].astype(float).mean())
                    if len(valid_group)
                    else np.nan
                ),
                "error_rate": float(error_mask.mean()),
                "accuracy_ci_low": low,
                "accuracy_ci_high": high,
                "parse_rate": float(
                    (
                        ~group["parse_method"]
                        .fillna("")
                        .astype(str)
                        .str.startswith(("error", "tail"))
                    ).mean()
                ),
                "manual_review_rate": float(
                    group["needs_manual_review"].astype(bool).mean()
                ),
                "latency_seconds_mean": pd.to_numeric(
                    group.get("latency_seconds"), errors="coerce"
                ).mean(),
                "output_tokens_mean": pd.to_numeric(
                    group.get("output_tokens"), errors="coerce"
                ).mean(),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_robustness_drop(summary: pd.DataFrame) -> pd.DataFrame:
    key_columns = ["phase", "model_key", "prompt_mode", "image_mode"]
    clean = summary[summary["condition"] == "clean"][
        key_columns + ["accuracy"]
    ].rename(columns={"accuracy": "clean_accuracy"})
    merged = summary.merge(clean, on=key_columns, how="left")
    merged["accuracy_drop"] = merged["clean_accuracy"] - merged["accuracy"]
    return merged


def build_failure_detector(frame: pd.DataFrame) -> pd.DataFrame:
    improvement = frame[
        (frame["phase"] == "improvement") & (frame["model_key"] == "rl")
    ].copy()
    if improvement.empty:
        return pd.DataFrame()

    baseline = improvement[
        (improvement["prompt_mode"] == "paper")
        & (improvement["image_mode"] == "global")
    ][["row_id", "condition", "is_correct", "extracted_answer"]].rename(
        columns={
            "is_correct": "baseline_correct",
            "extracted_answer": "baseline_answer",
        }
    )
    improved = improvement[
        (improvement["prompt_mode"] == "evidence_first")
        & (improvement["image_mode"] == "global")
    ][["row_id", "condition", "is_correct", "extracted_answer"]].rename(
        columns={
            "is_correct": "improved_correct",
            "extracted_answer": "improved_answer",
        }
    )
    paired = baseline.merge(improved, on=["row_id", "condition"], how="inner")
    rows = []
    for condition, group in paired.groupby("condition"):
        baseline_answer = group["baseline_answer"].map(normalize_text)
        improved_answer = group["improved_answer"].map(normalize_text)
        disagreement = baseline_answer != improved_answer
        actual_failure = ~group["baseline_correct"].astype(bool)
        precision, recall, f1, _ = precision_recall_fscore_support(
            actual_failure,
            disagreement,
            average="binary",
            zero_division=0,
        )
        agreement = ~disagreement
        selective_accuracy = (
            float(group.loc[agreement, "baseline_correct"].astype(float).mean())
            if agreement.any()
            else np.nan
        )
        rows.append(
            {
                "condition": condition,
                "n": len(group),
                "baseline_accuracy": group["baseline_correct"].mean(),
                "improved_accuracy": group["improved_correct"].mean(),
                "accuracy_delta": (
                    group["improved_correct"].astype(float)
                    - group["baseline_correct"].astype(float)
                ).mean(),
                "agreement_coverage": agreement.mean(),
                "selective_accuracy": selective_accuracy,
                "failure_precision": precision,
                "failure_recall": recall,
                "failure_f1": f1,
            }
        )
    return pd.DataFrame(rows)


def make_manual_template(frame: pd.DataFrame) -> pd.DataFrame:
    failures = frame[
        (frame["phase"] == "robustness")
        & (frame["model_key"] == "rl")
        & (frame["prompt_mode"] == "paper")
        & (frame["image_mode"] == "global")
        & (~frame["is_correct"].astype(bool))
    ].copy()
    columns = [
        "row_id",
        "sample_id",
        "condition",
        "category",
        "level",
        "question",
        "answer",
        "extracted_answer",
        "image_paths",
        "output_text",
    ]
    failures = failures[columns].drop_duplicates(
        subset=["row_id", "condition"]
    )
    failures["error_class"] = ""
    failures["visual_fact_error"] = ""
    failures["notes"] = ""
    return failures


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    scored_dir = root / args.scored_dir
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(scored_dir.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No scored CSV files under {scored_dir}")
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    frame = frame.drop_duplicates(
        subset=[
            "row_id",
            "phase",
            "model_key",
            "condition",
            "prompt_mode",
            "image_mode",
        ],
        keep="last",
    )
    frame.to_csv(output_dir / "all_scored.csv", index=False)

    summary = build_summary(
        frame,
        repeats=args.bootstrap_repeats,
        seed=args.seed,
    )
    summary.to_csv(output_dir / "summary.csv", index=False)
    build_robustness_drop(summary).to_csv(
        output_dir / "robustness_drop.csv",
        index=False,
    )
    build_failure_detector(frame).to_csv(
        output_dir / "failure_detector.csv",
        index=False,
    )
    make_manual_template(frame).to_csv(
        output_dir / "manual_error_annotations.csv",
        index=False,
    )
    print(f"analysis written to {output_dir}")


if __name__ == "__main__":
    main()
