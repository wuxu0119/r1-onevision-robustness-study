from __future__ import annotations

import argparse
import hashlib
import itertools
import traceback
from pathlib import Path

from tqdm import tqdm

from r1eval.data import load_benchmark
from r1eval.io_utils import append_jsonl, load_json, read_jsonl
from r1eval.modeling import R1ModelRunner
from r1eval.perturbations import prepare_condition_images
from r1eval.prompts import build_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.json")
    parser.add_argument(
        "--phase",
        required=True,
        choices=[
            "reproduce",
            "robustness",
            "improvement",
            "controls",
            "confidence",
            "heldout_gate",
            "gate_generalization",
            "multiview_budgeted",
        ],
    )
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attention", default="sdpa")
    parser.add_argument("--retry-errors", action="store_true")
    return parser.parse_args()


def completion_key(record: dict) -> tuple:
    return (
        int(record["row_id"]),
        record["condition"],
        record["prompt_mode"],
        record["image_mode"],
        int(record.get("replicate", 0)),
    )


def replicate_ids(phase: dict) -> list[int]:
    configured = phase.get("replicates", 1)
    if isinstance(configured, int):
        if configured < 1:
            raise ValueError("replicates must be at least 1")
        return list(range(configured))
    values = [int(value) for value in configured]
    if not values or len(set(values)) != len(values) or min(values) < 0:
        raise ValueError("replicates must be a non-empty set of non-negative integers")
    return values


def resolve_generation(config: dict, phase_name: str, phase: dict) -> dict:
    defaults = config["generation"]
    max_tokens_key = (
        "max_new_tokens_reproduce"
        if phase_name == "reproduce"
        else "max_new_tokens_other"
    )
    generation = {
        "max_new_tokens": int(defaults[max_tokens_key]),
        "temperature": float(defaults["temperature"]),
        "top_p": float(defaults["top_p"]),
        "top_k": int(defaults["top_k"]),
        "repetition_penalty": float(defaults["repetition_penalty"]),
    }
    overrides = phase.get("generation", {})
    for key in generation:
        if key in overrides:
            conversion = int if key in {"max_new_tokens", "top_k"} else float
            generation[key] = conversion(overrides[key])
    return generation


def deterministic_generation_seed(
    *,
    base_seed: int,
    row_id: int,
    phase: str,
    model_key: str,
    condition: str,
    prompt_mode: str,
    image_mode: str,
    replicate: int,
) -> int:
    identity = "\x1f".join(
        [
            str(base_seed),
            str(row_id),
            phase,
            model_key,
            condition,
            prompt_mode,
            image_mode,
            str(replicate),
        ]
    )
    digest = hashlib.sha256(identity.encode("utf-8")).digest()
    # torch.manual_seed accepts signed 64-bit values. Staying below 2**63 also
    # keeps the recorded seed portable across CUDA and CPU installations.
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_json(root / args.config)
    phase = config["phases"][args.phase]
    if args.model_key not in phase["models"]:
        raise ValueError(
            f"Model {args.model_key!r} is not configured for phase "
            f"{args.phase!r}: {phase['models']}"
        )
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("shard-id must be in [0, num-shards)")

    model_id = config["models"][args.model_key]
    subset_size = phase["subset_size"]
    manifest_name = "all.jsonl" if subset_size is None else f"subset_{subset_size}.jsonl"
    manifest_path = root / "data" / "manifests" / manifest_name
    manifest = read_jsonl(manifest_path)
    if not manifest:
        raise FileNotFoundError(
            f"Missing manifest {manifest_path}. Run scripts/01_prepare_manifest.py first."
        )
    manifest = [
        record
        for record in manifest
        if int(record["row_id"]) % args.num_shards == args.shard_id
    ]

    dataset = load_benchmark(
        config["dataset_id"],
        config["dataset_split"],
        root / "data" / "cache",
    )
    output_path = (
        root
        / "results"
        / "raw"
        / f"{args.phase}__{args.model_key}__{args.shard_id:02d}-of-{args.num_shards:02d}.jsonl"
    )
    existing = read_jsonl(output_path)
    completed = {
        completion_key(record)
        for record in existing
        if args.retry_errors is False or not record.get("error")
    }

    runner = R1ModelRunner(
        model_id,
        device=args.device,
        attention=args.attention,
        seed=int(config["seed"]) + args.shard_id,
    )
    generation = resolve_generation(config, args.phase, phase)

    combinations = list(
        itertools.product(
            phase["conditions"],
            phase["prompt_modes"],
            phase["image_modes"],
            replicate_ids(phase),
        )
    )
    total = len(manifest) * len(combinations)
    progress = tqdm(total=total, desc=f"{args.phase}/{args.model_key}/shard{args.shard_id}")
    for sample in manifest:
        row_id = int(sample["row_id"])
        for condition, prompt_mode, image_mode, replicate in combinations:
            key = (row_id, condition, prompt_mode, image_mode, replicate)
            if key in completed:
                progress.update(1)
                continue
            generation_seed = deterministic_generation_seed(
                base_seed=int(config["seed"]),
                row_id=row_id,
                phase=args.phase,
                model_key=args.model_key,
                condition=condition,
                prompt_mode=prompt_mode,
                image_mode=image_mode,
                replicate=replicate,
            )
            base_record = {
                **sample,
                "phase": args.phase,
                "model_key": args.model_key,
                "model_id": model_id,
                "condition": condition,
                "prompt_mode": prompt_mode,
                "image_mode": image_mode,
                "replicate": replicate,
                "shard_id": args.shard_id,
                "num_shards": args.num_shards,
                "generation": generation,
                "generation_seed": generation_seed,
            }
            try:
                image_paths, condition_meta = prepare_condition_images(
                    dataset=dataset,
                    row_id=row_id,
                    condition=condition,
                    image_mode=image_mode,
                    derived_dir=root / "data" / "derived",
                    seed=int(config["seed"]),
                )
                prompt = build_prompt(
                    question=sample["question"],
                    choices=sample["choices"],
                    prompt_mode=prompt_mode,
                    image_mode=image_mode,
                )
                generation_result = runner.generate(
                    image_paths=image_paths,
                    prompt=prompt,
                    generation=generation,
                    seed=generation_seed,
                )
                record = {
                    **base_record,
                    "image_paths": [str(path) for path in image_paths],
                    "condition_meta": condition_meta,
                    **generation_result,
                    "error": "",
                }
            except Exception as exc:  # keep long jobs resumable
                record = {
                    **base_record,
                    "image_paths": [],
                    "condition_meta": {},
                    "output_text": "",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            append_jsonl(output_path, record)
            progress.update(1)
    progress.close()
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
