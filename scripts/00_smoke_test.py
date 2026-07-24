from __future__ import annotations

import argparse
from pathlib import Path

from r1eval.data import decode_image, load_benchmark
from r1eval.modeling import R1ModelRunner
from r1eval.prompts import build_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        default="Fancy-MLLM/R1-Onevision-7B-RL",
    )
    parser.add_argument(
        "--dataset-id",
        default="Fancy-MLLM/R1-Onevision-Bench",
    )
    parser.add_argument("--row-id", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    dataset = load_benchmark(args.dataset_id, "train", root / "data" / "cache")
    row = dataset[args.row_id]

    image_path = root / "data" / "smoke" / f"{args.row_id}.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    decode_image(row["image"]).save(image_path)

    prompt = build_prompt(
        question=row.get("question", ""),
        choices=row.get("choices", ""),
        prompt_mode="paper",
        image_mode="global",
    )
    runner = R1ModelRunner(args.model_id, device=args.device, seed=20260723)
    result = runner.generate(
        image_paths=[image_path],
        prompt=prompt,
        generation={
            "max_new_tokens": 1024,
            "temperature": 0.1,
            "top_p": 0.001,
            "top_k": 1,
            "repetition_penalty": 1.05,
        },
    )
    print(f"sample_id: {row.get('index', args.row_id)}")
    print(f"ground_truth: {row.get('answer')}")
    print(f"latency_seconds: {result['latency_seconds']:.3f}")
    print(f"peak_memory_gb: {result['peak_memory_gb']:.3f}")
    print(result["output_text"])


if __name__ == "__main__":
    main()

