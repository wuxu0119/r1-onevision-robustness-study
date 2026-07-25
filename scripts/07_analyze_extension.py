from __future__ import annotations

import argparse
import ast
import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


BOOTSTRAP_REPEATS = 10_000
BASE_SEED = 20260724


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze matched-panel controls and confidence extensions."
    )
    parser.add_argument("--scored-dir", default="results/scored")
    parser.add_argument("--output-dir", default="results/analysis/extension")
    parser.add_argument("--bootstrap-repeats", type=int, default=BOOTSTRAP_REPEATS)
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    return parser.parse_args()


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"1", "true", "t", "yes"})
    )


def normalize_answer(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.upper()


def load_phase(
    scored_dir: Path,
    phase: str,
    model: str,
    *,
    required: bool = True,
) -> pd.DataFrame:
    paths = sorted(scored_dir.glob(f"{phase}__{model}__*.csv"))
    if not paths:
        if required:
            raise FileNotFoundError(
                f"Missing scored shards for phase={phase!r}, model={model!r} "
                f"under {scored_dir}."
            )
        return pd.DataFrame()
    frame = pd.concat(
        [pd.read_csv(path, low_memory=False) for path in paths],
        ignore_index=True,
    )
    if "replicate" not in frame:
        frame["replicate"] = 0
    frame["replicate"] = frame["replicate"].fillna(0).astype(int)
    frame["row_id"] = frame["row_id"].astype(int)
    frame["is_correct"] = as_bool(frame["is_correct"])
    if "error" not in frame:
        frame["error"] = ""
    frame["error"] = frame["error"].fillna("").astype(str)
    frame["normalized_answer"] = normalize_answer(frame["extracted_answer"])
    keys = ["row_id", "condition", "prompt_mode", "image_mode", "replicate"]
    frame = frame.drop_duplicates(keys, keep="last").reset_index(drop=True)
    return frame


def valid_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["error"].eq("")].copy()


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    repeats: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return math.nan, math.nan
    estimates = np.empty(repeats, dtype=float)
    for index in range(repeats):
        sample = rng.integers(0, len(values), size=len(values))
        estimates[index] = values[sample].mean()
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def exact_mcnemar_p(left_only: int, right_only: int) -> float:
    discordant = int(left_only + right_only)
    if discordant == 0:
        return 1.0
    tail = min(int(left_only), int(right_only))
    probability = sum(
        math.comb(discordant, index) for index in range(tail + 1)
    ) / (2**discordant)
    return float(min(1.0, 2.0 * probability))


def paired_contrast(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    model: str,
    left_name: str,
    right_name: str,
    repeats: int,
    seed: int,
) -> dict:
    columns = ["row_id", "is_correct", "normalized_answer"]
    merged = valid_rows(left)[columns].merge(
        valid_rows(right)[columns],
        on="row_id",
        suffixes=("_left", "_right"),
        how="inner",
        validate="one_to_one",
    )
    left_correct = merged["is_correct_left"].to_numpy(dtype=bool)
    right_correct = merged["is_correct_right"].to_numpy(dtype=bool)
    differences = right_correct.astype(float) - left_correct.astype(float)
    ci_low, ci_high = bootstrap_mean_ci(
        differences,
        repeats=repeats,
        rng=np.random.default_rng(seed),
    )
    left_only = int(np.sum(left_correct & ~right_correct))
    right_only = int(np.sum(~left_correct & right_correct))
    agreement = (
        merged["normalized_answer_left"].ne("")
        & merged["normalized_answer_right"].ne("")
        & merged["normalized_answer_left"].eq(merged["normalized_answer_right"])
    )
    return {
        "model": model,
        "left_condition": left_name,
        "right_condition": right_name,
        "n": int(len(merged)),
        "left_accuracy_pct": float(left_correct.mean() * 100),
        "right_accuracy_pct": float(right_correct.mean() * 100),
        "delta_pp": float(differences.mean() * 100),
        "ci_low_pp": float(ci_low * 100),
        "ci_high_pp": float(ci_high * 100),
        "left_only_correct": left_only,
        "right_only_correct": right_only,
        "mcnemar_exact_p": exact_mcnemar_p(left_only, right_only),
        "answer_agreement_pct": float(agreement.mean() * 100),
    }


def condition_slice(frame: pd.DataFrame, condition: str) -> pd.DataFrame:
    result = frame[
        frame["condition"].eq(condition)
        & frame["prompt_mode"].eq("paper")
        & frame["image_mode"].eq("global")
        & frame["replicate"].eq(0)
    ].copy()
    if result.empty:
        raise ValueError(f"Missing condition {condition!r}.")
    return result


def summarize_controls(
    scored_dir: Path,
    output_dir: Path,
    *,
    repeats: int,
    seed: int,
) -> None:
    fixed_conditions = [
        "control_target_rescaled",
        "control_blank_1_left",
        "control_blank_1_right",
        "control_noise_1_left",
        "control_noise_1_right",
        "distractor_1_seed0_left",
        "distractor_1_seed0_right",
        "distractor_1_seed1_left",
        "distractor_1_seed1_right",
        "distractor_1_seed2_left",
        "distractor_1_seed2_right",
    ]
    summaries: list[dict] = []
    contrasts: list[dict] = []
    seed_aggregates: list[dict] = []
    position_rows: list[dict] = []

    for model_index, model in enumerate(["base", "sft", "rl"]):
        robustness = load_phase(scored_dir, "robustness", model)
        controls = load_phase(scored_dir, "controls", model)
        clean = condition_slice(robustness, "clean")
        original = condition_slice(robustness, "distractor_1")
        by_condition = {
            condition: condition_slice(controls, condition)
            for condition in fixed_conditions
        }
        by_condition["clean"] = clean
        by_condition["distractor_1"] = original

        for condition, frame in by_condition.items():
            usable = valid_rows(frame)
            summaries.append(
                {
                    "model": model,
                    "condition": condition,
                    "n": int(len(frame)),
                    "valid_n": int(len(usable)),
                    "errors": int(len(frame) - len(usable)),
                    "accuracy_pct": float(usable["is_correct"].mean() * 100),
                    "parse_pct": float(
                        usable["normalized_answer"].ne("").mean() * 100
                    ),
                }
            )

        contrast_pairs: list[tuple[str, str]] = [
            ("clean", "control_target_rescaled"),
            ("clean", "distractor_1"),
        ]
        for position in ["left", "right"]:
            blank = f"control_blank_1_{position}"
            noise = f"control_noise_1_{position}"
            contrast_pairs.extend(
                [
                    ("control_target_rescaled", blank),
                    ("control_target_rescaled", noise),
                    (blank, noise),
                ]
            )
            for semantic_seed in range(3):
                semantic = f"distractor_1_seed{semantic_seed}_{position}"
                contrast_pairs.extend(
                    [
                        ("clean", semantic),
                        (blank, semantic),
                        (noise, semantic),
                    ]
                )
        contrast_pairs.extend(
            [
                ("control_blank_1_left", "control_blank_1_right"),
                ("control_noise_1_left", "control_noise_1_right"),
                ("distractor_1_seed0_left", "distractor_1_seed0_right"),
                ("distractor_1_seed1_left", "distractor_1_seed1_right"),
                ("distractor_1_seed2_left", "distractor_1_seed2_right"),
            ]
        )
        for pair_index, (left_name, right_name) in enumerate(contrast_pairs):
            contrasts.append(
                paired_contrast(
                    by_condition[left_name],
                    by_condition[right_name],
                    model=model,
                    left_name=left_name,
                    right_name=right_name,
                    repeats=repeats,
                    seed=seed + model_index * 100 + pair_index,
                )
            )

        semantic_conditions = [
            f"distractor_1_seed{semantic_seed}_{position}"
            for semantic_seed in range(3)
            for position in ["left", "right"]
        ]
        blank_conditions = [
            "control_blank_1_left",
            "control_blank_1_right",
        ]
        semantic = valid_rows(clean)[
            ["row_id", "is_correct", "normalized_answer"]
        ].rename(
            columns={
                "is_correct": "clean_correct",
                "normalized_answer": "clean_answer",
            }
        )
        for condition in [*blank_conditions, *semantic_conditions]:
            frame = valid_rows(by_condition[condition])[
                ["row_id", "is_correct", "normalized_answer"]
            ].rename(
                columns={
                    "is_correct": f"{condition}_correct",
                    "normalized_answer": f"{condition}_answer",
                }
            )
            semantic = semantic.merge(frame, on="row_id", how="inner")
        semantic_correctness_columns = [
            f"{condition}_correct" for condition in semantic_conditions
        ]
        blank_correctness_columns = [
            f"{condition}_correct" for condition in blank_conditions
        ]
        semantic_mean = (
            semantic[semantic_correctness_columns].astype(float).mean(axis=1)
        )
        blank_mean = (
            semantic[blank_correctness_columns].astype(float).mean(axis=1)
        )
        comparisons = {
            "semantic_mean_minus_clean": (
                semantic_mean - semantic["clean_correct"].astype(float)
            ),
            "semantic_mean_minus_matched_blank": semantic_mean - blank_mean,
        }
        answer_agreements = []
        for condition in semantic_conditions:
            position = condition.rsplit("_", 1)[1]
            blank_condition = f"control_blank_1_{position}"
            answer_agreements.append(
                semantic[f"{condition}_answer"].ne("")
                & semantic[f"{blank_condition}_answer"].ne("")
                & semantic[f"{condition}_answer"].eq(
                    semantic[f"{blank_condition}_answer"]
                )
            )
        mean_agreement = (
            np.column_stack([series.to_numpy() for series in answer_agreements])
            .mean(axis=1)
            .mean()
        )
        for comparison_index, (comparison, row_delta) in enumerate(
            comparisons.items()
        ):
            low, high = bootstrap_mean_ci(
                row_delta.to_numpy(),
                repeats=repeats,
                rng=np.random.default_rng(
                    seed + model_index * 10 + comparison_index + 1000
                ),
            )
            seed_aggregates.append(
                {
                    "model": model,
                    "comparison": comparison,
                    "n_rows": int(len(semantic)),
                    "semantic_trials": int(
                        len(semantic) * len(semantic_conditions)
                    ),
                    "clean_accuracy_pct": float(
                        semantic["clean_correct"].mean() * 100
                    ),
                    "matched_blank_accuracy_pct": float(blank_mean.mean() * 100),
                    "mean_semantic_accuracy_pct": float(
                        semantic_mean.mean() * 100
                    ),
                    "mean_delta_pp": float(row_delta.mean() * 100),
                    "ci_low_pp": float(low * 100),
                    "ci_high_pp": float(high * 100),
                    "semantic_blank_answer_agreement_pct": float(
                        mean_agreement * 100
                    ),
                }
            )

        clean_for_position = clean[["row_id", "is_correct"]].rename(
            columns={"is_correct": "clean_correct"}
        )
        for condition in [
            "control_blank_1_left",
            "control_blank_1_right",
            "control_noise_1_left",
            "control_noise_1_right",
            *semantic_conditions,
        ]:
            frame = valid_rows(by_condition[condition]).copy()
            frame["target_panel_index"] = frame["condition_meta"].map(
                parse_target_panel_index
            )
            frame = frame.merge(clean_for_position, on="row_id", how="left")
            for position, group in frame.dropna(
                subset=["target_panel_index"]
            ).groupby("target_panel_index"):
                position_rows.append(
                    {
                        "model": model,
                        "condition": condition,
                        "target_panel_index": int(position),
                        "n": int(len(group)),
                        "accuracy_pct": float(group["is_correct"].mean() * 100),
                        "clean_accuracy_same_rows_pct": float(
                            group["clean_correct"].mean() * 100
                        ),
                        "delta_pp": float(
                            (
                                group["is_correct"].astype(float)
                                - group["clean_correct"].astype(float)
                            ).mean()
                            * 100
                        ),
                    }
                )

    pd.DataFrame(summaries).to_csv(
        output_dir / "control_condition_summary.csv", index=False
    )
    pd.DataFrame(contrasts).to_csv(
        output_dir / "control_paired_contrasts.csv", index=False
    )
    pd.DataFrame(seed_aggregates).to_csv(
        output_dir / "semantic_seed_aggregate.csv", index=False
    )
    pd.DataFrame(position_rows).to_csv(
        output_dir / "target_position_descriptive.csv", index=False
    )


def parse_target_panel_index(value: object) -> float:
    try:
        parsed = ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        return math.nan
    if not isinstance(parsed, dict) or "target_panel_index" not in parsed:
        return math.nan
    return float(parsed["target_panel_index"])


def selection_metrics(
    accepted: np.ndarray,
    correct: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> dict:
    accepted = np.asarray(accepted, dtype=bool)
    correct = np.asarray(correct, dtype=bool)
    if len(accepted) != len(correct) or len(accepted) == 0:
        raise ValueError("Selection arrays must be non-empty and have equal length.")

    def point(a: np.ndarray, c: np.ndarray) -> tuple[float, float, float]:
        coverage = float(a.mean())
        selective = float(c[a].mean()) if a.any() else math.nan
        rejected_error = float((~c[~a]).mean()) if (~a).any() else math.nan
        return coverage, selective, rejected_error

    coverage, selective, rejected_error = point(accepted, correct)
    rng = np.random.default_rng(seed)
    boot = np.empty((repeats, 3), dtype=float)
    for index in range(repeats):
        sample = rng.integers(0, len(accepted), size=len(accepted))
        boot[index] = point(accepted[sample], correct[sample])

    intervals = []
    for column in range(3):
        values = boot[:, column]
        values = values[np.isfinite(values)]
        if len(values):
            intervals.append(tuple(np.quantile(values, [0.025, 0.975])))
        else:
            intervals.append((math.nan, math.nan))
    return {
        "n": int(len(correct)),
        "accepted_n": int(accepted.sum()),
        "rejected_n": int((~accepted).sum()),
        "coverage_pct": coverage * 100,
        "coverage_ci_low_pct": float(intervals[0][0] * 100),
        "coverage_ci_high_pct": float(intervals[0][1] * 100),
        "selective_accuracy_pct": selective * 100 if math.isfinite(selective) else math.nan,
        "selective_accuracy_ci_low_pct": float(intervals[1][0] * 100),
        "selective_accuracy_ci_high_pct": float(intervals[1][1] * 100),
        "rejected_error_pct": rejected_error * 100
        if math.isfinite(rejected_error)
        else math.nan,
        "rejected_error_ci_low_pct": float(intervals[2][0] * 100),
        "rejected_error_ci_high_pct": float(intervals[2][1] * 100),
        "baseline_accuracy_pct": float(correct.mean() * 100),
    }


def random_rejection_baseline(
    correct: np.ndarray,
    accepted_n: int,
    *,
    repeats: int,
    seed: int,
) -> dict:
    correct = np.asarray(correct, dtype=bool)
    if accepted_n <= 0:
        return {
            "random_selective_accuracy_pct": math.nan,
            "random_ci_low_pct": math.nan,
            "random_ci_high_pct": math.nan,
        }
    rng = np.random.default_rng(seed)
    estimates = np.empty(repeats, dtype=float)
    for index in range(repeats):
        selected = rng.choice(len(correct), size=accepted_n, replace=False)
        estimates[index] = correct[selected].mean()
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {
        "random_selective_accuracy_pct": float(estimates.mean() * 100),
        "random_ci_low_pct": float(low * 100),
        "random_ci_high_pct": float(high * 100),
    }


def gate_summary(
    paper: pd.DataFrame,
    alternative: pd.DataFrame,
    *,
    model: str,
    condition: str,
    split: str,
    method: str,
    repeats: int,
    seed: int,
) -> tuple[dict, pd.DataFrame]:
    columns = [
        "row_id",
        "sample_id",
        "category",
        "choices",
        "is_correct",
        "normalized_answer",
    ]
    left = valid_rows(paper)[columns].rename(
        columns={
            "is_correct": "paper_correct",
            "normalized_answer": "paper_answer",
        }
    )
    right = valid_rows(alternative)[
        ["row_id", "normalized_answer"]
    ].rename(columns={"normalized_answer": "alternative_answer"})
    merged = left.merge(right, on="row_id", how="inner", validate="one_to_one")
    accepted = (
        merged["paper_answer"].ne("")
        & merged["alternative_answer"].ne("")
        & merged["paper_answer"].eq(merged["alternative_answer"])
    ).to_numpy()
    correct = merged["paper_correct"].to_numpy(dtype=bool)
    metrics = selection_metrics(
        accepted,
        correct,
        repeats=repeats,
        seed=seed,
    )
    metrics.update(
        {
            "model": model,
            "condition": condition,
            "split": split,
            "method": method,
        }
    )
    metrics.update(
        random_rejection_baseline(
            correct,
            int(accepted.sum()),
            repeats=repeats,
            seed=seed + 5000,
        )
    )
    merged["accepted"] = accepted
    merged["item_type"] = merged["choices"].map(item_type)
    return metrics, merged


def item_type(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    return "free_response" if not text or text.lower() == "nan" else "multiple_choice"


def summarize_dual_prompt_gates(
    scored_dir: Path,
    output_dir: Path,
    *,
    repeats: int,
    seed: int,
) -> None:
    rows: list[dict] = []
    item_rows: list[dict] = []
    conditions = [
        "clean",
        "downsample_50",
        "downsample_25",
        "distractor_1",
        "distractor_3",
    ]

    robustness = {
        model: load_phase(scored_dir, "robustness", model)
        for model in ["base", "sft", "rl"]
    }
    alternatives = {
        "base": load_phase(scored_dir, "gate_generalization", "base"),
        "sft": load_phase(scored_dir, "gate_generalization", "sft"),
        "rl": load_phase(scored_dir, "gate_generalization", "rl"),
    }
    for model_index, model in enumerate(["base", "sft", "rl"]):
        for condition_index, condition in enumerate(conditions):
            paper = condition_slice(robustness[model], condition)
            alternative = alternatives[model][
                alternatives[model]["condition"].eq(condition)
                & alternatives[model]["prompt_mode"].eq("evidence_first")
                & alternatives[model]["image_mode"].eq("global")
                & alternatives[model]["replicate"].eq(0)
            ]
            summary, merged = gate_summary(
                paper,
                alternative,
                model=model,
                condition=condition,
                split="robustness_subset_300",
                method="dual_prompt_agreement",
                repeats=repeats,
                seed=seed + model_index * 100 + condition_index,
            )
            rows.append(summary)
            for item_index, (kind, group) in enumerate(
                merged.groupby("item_type")
            ):
                accepted = group["accepted"].to_numpy(dtype=bool)
                correct = group["paper_correct"].to_numpy(dtype=bool)
                metrics = selection_metrics(
                    accepted,
                    correct,
                    repeats=repeats,
                    seed=seed + 1000 + model_index * 100 + condition_index * 10 + item_index,
                )
                metrics.update(
                    {
                        "model": model,
                        "condition": condition,
                        "item_type": kind,
                        "split": "robustness_subset_300",
                    }
                )
                item_rows.append(metrics)

    pd.DataFrame(rows).to_csv(
        output_dir / "dual_prompt_gate_all_checkpoints.csv", index=False
    )
    pd.DataFrame(item_rows).to_csv(
        output_dir / "dual_prompt_gate_by_item_type.csv", index=False
    )


def summarize_heldout_gate(
    scored_dir: Path,
    output_dir: Path,
    *,
    repeats: int,
    seed: int,
) -> None:
    paper = load_phase(scored_dir, "reproduce", "rl")
    alternative = load_phase(scored_dir, "heldout_gate", "rl")
    subset_ids = set(
        condition_slice(load_phase(scored_dir, "robustness", "rl"), "clean")[
            "row_id"
        ].astype(int)
    )
    rows: list[dict] = []
    split_masks = {
        "design_subset_300": paper["row_id"].isin(subset_ids),
        "heldout_complement_642": ~paper["row_id"].isin(subset_ids),
        "all_942": pd.Series(True, index=paper.index),
    }
    for index, (split, mask) in enumerate(split_masks.items()):
        paper_split = paper[mask]
        alternative_split = alternative[
            alternative["row_id"].isin(set(paper_split["row_id"]))
        ]
        summary, _ = gate_summary(
            paper_split,
            alternative_split,
            model="rl",
            condition="clean",
            split=split,
            method="dual_prompt_agreement",
            repeats=repeats,
            seed=seed + 2000 + index,
        )
        rows.append(summary)
    pd.DataFrame(rows).to_csv(
        output_dir / "clean_heldout_gate.csv", index=False
    )


def majority_record(group: pd.DataFrame) -> dict:
    group = valid_rows(group).sort_values("replicate")
    answers = {
        int(row.replicate): str(row.normalized_answer)
        for row in group.itertuples()
        if str(row.normalized_answer)
    }
    counts = Counter(answers.values())
    if not counts:
        return {
            "valid_replicates": 0,
            "majority_answer": "",
            "majority_votes": 0,
            "vote_confidence": 0.0,
            "unique_majority": False,
            "majority_correct": False,
            "first_correct": False,
            "first_two_agree": False,
        }
    majority_votes = max(counts.values())
    leaders = sorted(answer for answer, count in counts.items() if count == majority_votes)
    majority_answer = leaders[0]
    matching = group[group["normalized_answer"].eq(majority_answer)]
    majority_correct = bool(matching.iloc[0]["is_correct"])
    first = group[group["replicate"].eq(0)]
    first_correct = bool(first.iloc[0]["is_correct"]) if len(first) else False
    first_two_agree = (
        0 in answers
        and 1 in answers
        and answers[0] == answers[1]
    )
    return {
        "valid_replicates": int(len(answers)),
        "majority_answer": majority_answer,
        "majority_votes": int(majority_votes),
        "vote_confidence": float(majority_votes / len(answers)),
        "unique_majority": len(leaders) == 1,
        "majority_correct": majority_correct,
        "first_correct": first_correct,
        "first_two_agree": first_two_agree,
    }


def summarize_self_consistency(
    scored_dir: Path,
    output_dir: Path,
    *,
    repeats: int,
    seed: int,
) -> None:
    confidence = load_phase(scored_dir, "confidence", "rl")
    item_records: list[dict] = []
    for (condition, row_id), group in confidence.groupby(["condition", "row_id"]):
        record = majority_record(group)
        record.update({"condition": condition, "row_id": int(row_id)})
        item_records.append(record)
    items = pd.DataFrame(item_records)
    items.to_csv(output_dir / "self_consistency_items.csv", index=False)

    rows: list[dict] = []
    thresholds = [0.50, 0.625, 0.75, 0.875, 1.00]
    for condition_index, (condition, group) in enumerate(
        items.groupby("condition")
    ):
        full = group[group["valid_replicates"].eq(8)].copy()
        if len(full) != 300:
            raise ValueError(
                f"Expected 300 complete eight-replicate rows for {condition}; "
                f"found {len(full)}."
            )
        first_correct = full["first_correct"].to_numpy(dtype=bool)
        baseline = selection_metrics(
            np.ones(len(full), dtype=bool),
            first_correct,
            repeats=repeats,
            seed=seed + 3000 + condition_index,
        )
        baseline.update(
            {
                "condition": condition,
                "method": "single_stochastic_run",
                "threshold": math.nan,
            }
        )
        rows.append(baseline)

        two_accept = full["first_two_agree"].to_numpy(dtype=bool)
        two_metrics = selection_metrics(
            two_accept,
            first_correct,
            repeats=repeats,
            seed=seed + 3100 + condition_index,
        )
        two_metrics.update(
            {
                "condition": condition,
                "method": "same_prompt_two_run_agreement",
                "threshold": 1.0,
            }
        )
        two_metrics.update(
            random_rejection_baseline(
                first_correct,
                int(two_accept.sum()),
                repeats=repeats,
                seed=seed + 3200 + condition_index,
            )
        )
        rows.append(two_metrics)

        majority_correct = full["majority_correct"].to_numpy(dtype=bool)
        for threshold_index, threshold in enumerate(thresholds):
            accepted = (
                full["unique_majority"].to_numpy(dtype=bool)
                & full["vote_confidence"].ge(threshold).to_numpy(dtype=bool)
            )
            metrics = selection_metrics(
                accepted,
                majority_correct,
                repeats=repeats,
                seed=seed + 3300 + condition_index * 10 + threshold_index,
            )
            metrics.update(
                {
                    "condition": condition,
                    "method": "eight_run_majority_confidence",
                    "threshold": threshold,
                }
            )
            metrics.update(
                random_rejection_baseline(
                    majority_correct,
                    int(accepted.sum()),
                    repeats=repeats,
                    seed=seed + 3400 + condition_index * 10 + threshold_index,
                )
            )
            rows.append(metrics)
    pd.DataFrame(rows).to_csv(
        output_dir / "self_consistency_risk_coverage.csv", index=False
    )


def summarize_budgeted_multiview(
    scored_dir: Path,
    output_dir: Path,
    *,
    repeats: int,
    seed: int,
) -> None:
    frame = load_phase(scored_dir, "multiview_budgeted", "rl")
    summary_rows: list[dict] = []
    paired_rows: list[dict] = []
    grouped = frame.groupby(["condition", "prompt_mode", "image_mode"])
    for (condition, prompt_mode, image_mode), group in grouped:
        usable = valid_rows(group)
        summary_rows.append(
            {
                "model": "rl",
                "condition": condition,
                "prompt_mode": prompt_mode,
                "image_mode": image_mode,
                "n": int(len(group)),
                "valid_n": int(len(usable)),
                "errors": int(len(group) - len(usable)),
                "accuracy_pct": float(usable["is_correct"].mean() * 100)
                if len(usable)
                else math.nan,
                "mean_output_tokens": float(usable["output_tokens"].mean())
                if len(usable)
                else math.nan,
                "mean_latency_seconds": float(
                    usable["latency_seconds"].mean()
                )
                if len(usable)
                else math.nan,
                "max_peak_memory_gb": float(usable["peak_memory_gb"].max())
                if len(usable)
                else math.nan,
            }
        )

    pair_index = 0
    for condition in ["clean", "downsample_25", "distractor_3"]:
        for prompt_mode in ["paper", "evidence_first"]:
            left = frame[
                frame["condition"].eq(condition)
                & frame["prompt_mode"].eq(prompt_mode)
                & frame["image_mode"].eq("global_budgeted")
            ]
            right = frame[
                frame["condition"].eq(condition)
                & frame["prompt_mode"].eq(prompt_mode)
                & frame["image_mode"].eq(
                    "global_plus_quadrants_budgeted"
                )
            ]
            row = paired_contrast(
                left,
                right,
                model="rl",
                left_name="global_budgeted",
                right_name="global_plus_quadrants_budgeted",
                repeats=repeats,
                seed=seed + 4000 + pair_index,
            )
            row.update(
                {
                    "condition": condition,
                    "prompt_mode": prompt_mode,
                }
            )
            paired_rows.append(row)
            pair_index += 1

    pd.DataFrame(summary_rows).to_csv(
        output_dir / "multiview_budgeted_summary.csv", index=False
    )
    pd.DataFrame(paired_rows).to_csv(
        output_dir / "multiview_budgeted_paired.csv", index=False
    )


def write_readme(output_dir: Path, metadata: dict) -> None:
    text = f"""# Extension analysis

Generated from scored row-level outputs with {metadata['bootstrap_repeats']:,}
bootstrap resamples and seed {metadata['seed']}.

## Interpretation boundaries

- Target-only resizing, blank/noise panels, and three semantic distractor
  seeds are evaluated in a fixed-canvas design. Blank, noise, and semantic
  panels share target scale and labels at explicit left/right target
  positions. Their paired contrasts reduce, but do not eliminate,
  image-tokenization confounding.
- Target-position and distractor-seed breakdowns are exploratory.
- The 642 clean rows outside the original 300-row subset form a genuine
  held-out set relative to development of the evidence-first prompt. The
  robustness-subset and cross-checkpoint analyses remain exploratory because
  they reuse the original 300 items.
- Two stochastic runs are the cost-matched baseline for the two-prompt gate.
  The eight-run curve uses more inference and must be compared at stated
  coverage, not as a free full-coverage accuracy gain.
- The budgeted multi-view comparison uses one 864x864 view versus five
  384x384 views (746,496 versus 737,280 nominal pixels). It is an engineering
  proxy for matched visual-token cost, not a guarantee of identical internal
  token counts.
- Random rejection is reported at the same accepted count. It tests whether a
  gate selects easier/more reliable examples beyond what coverage alone would
  produce.
- No output in this directory is a completed human error taxonomy.

## Files

- `control_condition_summary.csv`
- `control_paired_contrasts.csv`
- `semantic_seed_aggregate.csv`
- `target_position_descriptive.csv`
- `dual_prompt_gate_all_checkpoints.csv`
- `dual_prompt_gate_by_item_type.csv`
- `clean_heldout_gate.csv`
- `self_consistency_risk_coverage.csv`
- `self_consistency_items.csv`
- `multiview_budgeted_summary.csv`
- `multiview_budgeted_paired.csv`
- `metadata.json`
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.bootstrap_repeats < 1000:
        raise ValueError("--bootstrap-repeats must be at least 1000.")
    root = Path(__file__).resolve().parents[1]
    scored_dir = root / args.scored_dir
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summarize_controls(
        scored_dir,
        output_dir,
        repeats=args.bootstrap_repeats,
        seed=args.seed,
    )
    summarize_dual_prompt_gates(
        scored_dir,
        output_dir,
        repeats=args.bootstrap_repeats,
        seed=args.seed,
    )
    summarize_heldout_gate(
        scored_dir,
        output_dir,
        repeats=args.bootstrap_repeats,
        seed=args.seed,
    )
    summarize_self_consistency(
        scored_dir,
        output_dir,
        repeats=args.bootstrap_repeats,
        seed=args.seed,
    )
    summarize_budgeted_multiview(
        scored_dir,
        output_dir,
        repeats=args.bootstrap_repeats,
        seed=args.seed,
    )

    metadata = {
        "seed": args.seed,
        "bootstrap_repeats": args.bootstrap_repeats,
        "scored_dir": str(scored_dir),
        "output_dir": str(output_dir),
        "analysis_scope": "post-hoc extension with a clean held-out complement",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    write_readme(output_dir, metadata)
    print(f"extension analysis written to {output_dir}")


if __name__ == "__main__":
    main()
