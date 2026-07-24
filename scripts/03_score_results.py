from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from r1eval.io_utils import read_jsonl
from r1eval.scoring import score_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/raw")
    parser.add_argument("--output-dir", default="results/scored")
    return parser.parse_args()


def score_file(path: Path, output_dir: Path) -> Path:
    scored = []
    for record in read_jsonl(path):
        if record.get("error"):
            record.update(
                {
                    "extracted_answer": "",
                    "parse_method": "error",
                    "is_correct": False,
                    "needs_manual_review": True,
                }
            )
        else:
            result = score_output(
                record.get("output_text", ""),
                record.get("answer", ""),
                record.get("choices", ""),
            )
            record.update(
                {
                    "extracted_answer": result.extracted_answer,
                    "parse_method": result.parse_method,
                    "is_correct": result.is_correct,
                    "needs_manual_review": result.needs_manual_review,
                }
            )
        record["source_file"] = path.name
        scored.append(record)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{path.stem}.csv"
    pd.DataFrame(scored).to_csv(output_path, index=False)
    return output_path


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    results_dir = root / args.results_dir
    output_dir = root / args.output_dir
    paths = sorted(results_dir.glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No raw JSONL files under {results_dir}")
    for path in paths:
        print(score_file(path, output_dir))


if __name__ == "__main__":
    main()

