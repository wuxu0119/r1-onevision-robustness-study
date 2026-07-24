from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


CONDITION_ORDER = [
    "clean",
    "downsample_50",
    "downsample_25",
    "distractor_1",
    "distractor_3",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", default="results/analysis")
    parser.add_argument("--output-dir", default="results/figures")
    return parser.parse_args()


def save_robustness(summary: pd.DataFrame, output_dir: Path) -> None:
    frame = summary[
        (summary["phase"] == "robustness")
        & (summary["prompt_mode"] == "paper")
        & (summary["image_mode"] == "global")
    ].copy()
    if frame.empty:
        return
    frame["condition"] = pd.Categorical(
        frame["condition"],
        categories=CONDITION_ORDER,
        ordered=True,
    )
    frame = frame.sort_values("condition")
    plt.figure(figsize=(8.2, 4.6))
    sns.lineplot(
        data=frame,
        x="condition",
        y="accuracy",
        hue="model_key",
        marker="o",
    )
    plt.xlabel("Input condition")
    plt.ylabel("Accuracy")
    plt.ylim(0.25, 0.45)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "robustness_accuracy.png", dpi=240)
    plt.savefig(output_dir / "robustness_accuracy.pdf")
    plt.close()


def save_improvement(summary: pd.DataFrame, output_dir: Path) -> None:
    frame = summary[
        (summary["phase"] == "improvement")
        & (summary["model_key"] == "rl")
    ].copy()
    if "error_rate" in frame:
        frame = frame[frame["error_rate"] == 0].copy()
    if frame.empty:
        return
    frame["method"] = (
        frame["prompt_mode"].astype(str)
        + " / "
        + frame["image_mode"].astype(str)
    )
    plt.figure(figsize=(9.0, 4.8))
    sns.barplot(
        data=frame,
        x="condition",
        y="accuracy",
        hue="method",
        order=["clean", "downsample_25", "distractor_3"],
    )
    plt.xlabel("Input condition")
    plt.ylabel("Accuracy")
    plt.ylim(0.25, 0.45)
    plt.legend(title="", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "improvement_accuracy.png", dpi=240)
    plt.savefig(output_dir / "improvement_accuracy.pdf")
    plt.close()


def save_failure_detector(analysis_dir: Path, output_dir: Path) -> None:
    path = analysis_dir / "failure_detector.csv"
    if not path.exists():
        return
    frame = pd.read_csv(path)
    if frame.empty:
        return
    long = frame.melt(
        id_vars=["condition"],
        value_vars=[
            "baseline_accuracy",
            "improved_accuracy",
            "selective_accuracy",
        ],
        var_name="metric",
        value_name="value",
    )
    plt.figure(figsize=(8.2, 4.6))
    sns.barplot(data=long, x="condition", y="value", hue="metric")
    plt.xlabel("Input condition")
    plt.ylabel("Accuracy")
    plt.ylim(0.25, 0.55)
    plt.legend(title="")
    plt.tight_layout()
    plt.savefig(output_dir / "selective_accuracy.png", dpi=240)
    plt.savefig(output_dir / "selective_accuracy.pdf")
    plt.close()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    analysis_dir = root / args.analysis_dir
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")

    summary_path = analysis_dir / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = pd.read_csv(summary_path)
    save_robustness(summary, output_dir)
    save_improvement(summary, output_dir)
    save_failure_detector(analysis_dir, output_dir)
    print(f"figures written to {output_dir}")


if __name__ == "__main__":
    main()
