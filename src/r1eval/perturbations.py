from __future__ import annotations

import hashlib
import math
import random
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageDraw, ImageFont, ImageOps

from .data import decode_image


def _stable_seed(*parts: Any) -> int:
    digest = hashlib.sha256("::".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def downsample_and_restore(image: Image.Image, ratio: float) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    small_size = (max(1, round(width * ratio)), max(1, round(height * ratio)))
    small = image.resize(small_size, Image.Resampling.BILINEAR)
    return small.resize((width, height), Image.Resampling.BILINEAR)


def _fit_on_panel(image: Image.Image, panel_size: int) -> Image.Image:
    panel = Image.new("RGB", (panel_size, panel_size), "white")
    fitted = ImageOps.contain(
        image.convert("RGB"),
        (panel_size - 24, panel_size - 46),
        Image.Resampling.LANCZOS,
    )
    x = (panel_size - fitted.width) // 2
    y = 34 + (panel_size - 34 - fitted.height) // 2
    panel.paste(fitted, (x, y))
    return panel


def _draw_panel_label(panel: Image.Image, label: str) -> None:
    draw = ImageDraw.Draw(panel)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    draw.rectangle((0, 0, 96, 30), fill="white")
    draw.text((8, 4), label, fill="black", font=font)


def compose_distractor_panels(
    target: Image.Image,
    distractors: list[Image.Image],
    *,
    seed: int,
    panel_size: int = 512,
) -> tuple[Image.Image, int]:
    images = [target.convert("RGB"), *[image.convert("RGB") for image in distractors]]
    rng = random.Random(seed)
    positions = list(range(len(images)))
    rng.shuffle(positions)

    shuffled = [None] * len(images)
    target_position = positions[0]
    for source_index, destination_index in enumerate(positions):
        shuffled[destination_index] = images[source_index]

    columns = 2 if len(images) > 1 else 1
    rows = math.ceil(len(images) / columns)
    canvas = Image.new("RGB", (columns * panel_size, rows * panel_size), "#d9d9d9")
    for index, image in enumerate(shuffled):
        panel = _fit_on_panel(image, panel_size)
        _draw_panel_label(panel, f"Panel {chr(65 + index)}")
        x = (index % columns) * panel_size
        y = (index // columns) * panel_size
        canvas.paste(panel, (x, y))
    return canvas, target_position


def quadrant_crops(image: Image.Image, overlap: float = 0.12) -> list[Image.Image]:
    image = image.convert("RGB")
    width, height = image.size
    mid_x, mid_y = width // 2, height // 2
    pad_x, pad_y = round(width * overlap), round(height * overlap)
    boxes = [
        (0, 0, min(width, mid_x + pad_x), min(height, mid_y + pad_y)),
        (max(0, mid_x - pad_x), 0, width, min(height, mid_y + pad_y)),
        (0, max(0, mid_y - pad_y), min(width, mid_x + pad_x), height),
        (max(0, mid_x - pad_x), max(0, mid_y - pad_y), width, height),
    ]
    return [image.crop(box) for box in boxes]


def _save_if_missing(path: Path, factory: Callable[[], Image.Image]) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.png")
    factory().save(temporary, format="PNG")
    temporary.replace(path)


def prepare_condition_images(
    *,
    dataset: Any,
    row_id: int,
    condition: str,
    image_mode: str,
    derived_dir: str | Path,
    seed: int,
) -> tuple[list[Path], dict[str, Any]]:
    """Create deterministic image files for one sample and condition."""
    derived_dir = Path(derived_dir)
    sample_dir = derived_dir / f"{row_id:04d}" / condition
    global_path = sample_dir / "global.png"
    target = decode_image(dataset[row_id]["image"])
    condition_meta: dict[str, Any] = {}

    if condition == "clean":
        _save_if_missing(global_path, lambda: target)
    elif condition == "downsample_50":
        _save_if_missing(global_path, lambda: downsample_and_restore(target, 0.50))
        condition_meta["ratio"] = 0.50
    elif condition == "downsample_25":
        _save_if_missing(global_path, lambda: downsample_and_restore(target, 0.25))
        condition_meta["ratio"] = 0.25
    elif condition.startswith("distractor_"):
        count = int(condition.rsplit("_", 1)[1])
        rng = random.Random(_stable_seed(seed, row_id, condition))
        candidates = [index for index in range(len(dataset)) if index != row_id]
        distractor_ids = rng.sample(candidates, count)
        distractors = [decode_image(dataset[index]["image"]) for index in distractor_ids]
        composite, target_position = compose_distractor_panels(
            target,
            distractors,
            seed=_stable_seed(seed, row_id, condition, "position"),
        )
        _save_if_missing(global_path, lambda: composite)
        condition_meta.update(
            {
                "distractor_ids": distractor_ids,
                "target_panel_index": target_position,
            }
        )
    else:
        raise ValueError(f"Unknown condition: {condition}")

    if image_mode == "global":
        return [global_path], condition_meta
    if image_mode != "global_plus_quadrants":
        raise ValueError(f"Unknown image mode: {image_mode}")

    global_image = Image.open(global_path).convert("RGB")
    crop_paths: list[Path] = []
    for index, crop in enumerate(quadrant_crops(global_image)):
        crop_path = sample_dir / f"quadrant_{index}.png"
        _save_if_missing(crop_path, lambda crop=crop: crop)
        crop_paths.append(crop_path)
    return [global_path, *crop_paths], condition_meta

