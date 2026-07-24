from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from PIL import Image

from .io_utils import write_jsonl

if TYPE_CHECKING:
    from datasets import Dataset


BENCHMARK_COLUMNS = [
    "index",
    "question",
    "choices",
    "answer",
    "level",
    "category",
]


def load_benchmark(
    dataset_id: str,
    split: str,
    cache_dir: str | Path,
) -> "Dataset":
    from datasets import load_dataset

    return load_dataset(
        dataset_id,
        split=split,
        cache_dir=str(cache_dir),
    )


def decode_image(value: Any) -> Image.Image:
    """Decode common Hugging Face image representations to RGB PIL."""
    if isinstance(value, Image.Image):
        return value.convert("RGB")

    if isinstance(value, np.ndarray):
        return Image.fromarray(value).convert("RGB")

    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        if value.get("path"):
            return Image.open(value["path"]).convert("RGB")

    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(value)).convert("RGB")

    if isinstance(value, str):
        if len(value) < 1024:
            candidate = Path(value)
            try:
                if candidate.exists():
                    return Image.open(candidate).convert("RGB")
            except OSError:
                pass
        payload = value
        if "," in payload and payload.lstrip().startswith("data:image"):
            payload = payload.split(",", 1)[1]
        payload = "".join(payload.split())
        return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")

    raise TypeError(f"Unsupported image representation: {type(value)!r}")


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    if isinstance(value, (list, dict)):
        return str(value)
    return str(value)


def row_metadata(row_id: int, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "row_id": int(row_id),
        "sample_id": safe_text(row.get("index", row_id)),
        "question": safe_text(row.get("question")),
        "choices": safe_text(row.get("choices")),
        "answer": safe_text(row.get("answer")),
        "level": safe_text(row.get("level")),
        "category": safe_text(row.get("category")),
    }


def make_manifests(
    dataset: "Dataset",
    output_dir: str | Path,
    subset_size: int,
    seed: int,
) -> tuple[Path, Path]:
    from sklearn.model_selection import train_test_split

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = [row_metadata(i, dataset[i]) for i in range(len(dataset))]
    all_path = output_dir / "all.jsonl"
    write_jsonl(all_path, metadata)

    frame = pd.DataFrame(metadata)
    indices = np.arange(len(frame))
    stratify = frame["category"].fillna("unknown")
    selected, _ = train_test_split(
        indices,
        train_size=subset_size,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    selected_set = set(int(i) for i in selected)
    subset = [record for i, record in enumerate(metadata) if i in selected_set]
    subset.sort(key=lambda item: item["row_id"])

    subset_path = output_dir / f"subset_{subset_size}.jsonl"
    write_jsonl(subset_path, subset)
    return all_path, subset_path
