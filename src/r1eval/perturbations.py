from __future__ import annotations

import hashlib
import math
import random
import re
from pathlib import Path
from typing import Any, Callable

import numpy as np
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
    target_position: int | None = None,
) -> tuple[Image.Image, int]:
    images = [target.convert("RGB"), *[image.convert("RGB") for image in distractors]]
    rng = random.Random(seed)
    if target_position is None:
        positions = list(range(len(images)))
        rng.shuffle(positions)
        shuffled = [None] * len(images)
        resolved_target_position = positions[0]
        for source_index, destination_index in enumerate(positions):
            shuffled[destination_index] = images[source_index]
    else:
        if not 0 <= target_position < len(images):
            raise ValueError(
                f"target_position must be in [0, {len(images)}), "
                f"got {target_position}"
            )
        resolved_target_position = target_position
        shuffled = [None] * len(images)
        shuffled[target_position] = images[0]
        remaining_positions = [
            index for index in range(len(images)) if index != target_position
        ]
        shuffled_distractors = images[1:]
        rng.shuffle(shuffled_distractors)
        for destination_index, image in zip(
            remaining_positions, shuffled_distractors
        ):
            shuffled[destination_index] = image

    columns = 2 if len(images) > 1 else 1
    rows = math.ceil(len(images) / columns)
    canvas = Image.new("RGB", (columns * panel_size, rows * panel_size), "#d9d9d9")
    for index, image in enumerate(shuffled):
        panel = _fit_on_panel(image, panel_size)
        _draw_panel_label(panel, f"Panel {chr(65 + index)}")
        x = (index % columns) * panel_size
        y = (index // columns) * panel_size
        canvas.paste(panel, (x, y))
    return canvas, resolved_target_position


def make_control_panel(
    mode: str,
    *,
    seed: int,
    size: tuple[int, int] = (512, 512),
) -> Image.Image:
    """Create a deterministic non-semantic panel for confound controls."""
    if mode == "blank":
        return Image.new("RGB", size, "white")
    if mode == "noise":
        rng = np.random.default_rng(seed % (2**32))
        values = rng.integers(
            0,
            256,
            size=(size[1], size[0], 3),
            dtype=np.uint8,
        )
        return Image.fromarray(values, mode="RGB")
    raise ValueError(f"Unknown control panel mode: {mode}")


def fit_on_budget_canvas(image: Image.Image, side: int) -> Image.Image:
    """Resize and pad one view to a fixed square visual-pixel budget."""
    canvas = Image.new("RGB", (side, side), "white")
    fitted = ImageOps.contain(
        image.convert("RGB"),
        (side, side),
        Image.Resampling.LANCZOS,
    )
    x = (side - fitted.width) // 2
    y = (side - fitted.height) // 2
    canvas.paste(fitted, (x, y))
    return canvas


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
    elif match := re.fullmatch(
        r"distractor_(\d+)(?:_seed(\d+))?(?:_(left|right))?",
        condition,
    ):
        count = int(match.group(1))
        seed_variant = int(match.group(2) or 0)
        fixed_position_name = match.group(3)
        fixed_target_position = (
            None
            if fixed_position_name is None
            else 0
            if fixed_position_name == "left"
            else 1
        )
        base_condition = f"distractor_{count}"
        available = [index for index in range(len(dataset)) if index != row_id]
        distractor_ids: list[int] = []
        # Select variants sequentially without replacement across seeds. The
        # variant-0 choice is unchanged from the original implementation.
        for variant in range(seed_variant + 1):
            content_seed_key = (
                base_condition
                if variant == 0
                else f"{base_condition}_seed{variant}"
            )
            rng = random.Random(_stable_seed(seed, row_id, content_seed_key))
            distractor_ids = rng.sample(available, count)
            chosen = set(distractor_ids)
            available = [index for index in available if index not in chosen]
        distractors = [decode_image(dataset[index]["image"]) for index in distractor_ids]
        composite, target_position = compose_distractor_panels(
            target,
            distractors,
            # Alternate semantic distractor seeds keep target placement fixed,
            # so the only intended change is distractor identity/content.
            seed=_stable_seed(seed, row_id, base_condition, "position"),
            target_position=fixed_target_position,
        )
        _save_if_missing(global_path, lambda: composite)
        condition_meta.update(
            {
                "distractor_ids": distractor_ids,
                "distractor_categories": [
                    str(dataset[index].get("category", ""))
                    for index in distractor_ids
                ],
                "distractor_seed_variant": seed_variant,
                "fixed_target_position": fixed_position_name or "",
                "target_panel_index": target_position,
            }
        )
    elif control_match := re.fullmatch(
        r"control_(blank|noise)_1(?:_(left|right))?",
        condition,
    ):
        control_mode = control_match.group(1)
        fixed_position_name = control_match.group(2)
        fixed_target_position = (
            None
            if fixed_position_name is None
            else 0
            if fixed_position_name == "left"
            else 1
        )
        control = make_control_panel(
            control_mode,
            seed=_stable_seed(
                seed,
                row_id,
                f"control_{control_mode}_1",
                "content",
            ),
        )
        # Reuse the distractor-1 position seed. Target scale, canvas geometry,
        # labels, and target-panel placement are therefore matched row by row.
        composite, target_position = compose_distractor_panels(
            target,
            [control],
            seed=_stable_seed(seed, row_id, "distractor_1", "position"),
            target_position=fixed_target_position,
        )
        _save_if_missing(global_path, lambda: composite)
        condition_meta.update(
            {
                "control_panel": control_mode,
                "fixed_target_position": fixed_position_name or "",
                "target_panel_index": target_position,
                "matched_condition": "distractor_1",
            }
        )
    elif condition == "control_target_rescaled":
        panel = _fit_on_panel(target, 512)
        _draw_panel_label(panel, "Panel A")
        _save_if_missing(global_path, lambda: panel)
        condition_meta.update(
            {
                "control_panel": "target_only_rescaled",
                "target_panel_index": 0,
                "matched_panel_size": 512,
            }
        )
    else:
        raise ValueError(f"Unknown condition: {condition}")

    if image_mode == "global":
        return [global_path], condition_meta
    global_image = Image.open(global_path).convert("RGB")
    if image_mode == "global_budgeted":
        budgeted_path = sample_dir / "global_budgeted_864.png"
        _save_if_missing(
            budgeted_path,
            lambda: fit_on_budget_canvas(global_image, 864),
        )
        condition_meta = {
            **condition_meta,
            "view_budget": "1x864x864",
            "nominal_pixel_budget": 864 * 864,
        }
        return [budgeted_path], condition_meta
    if image_mode == "global_plus_quadrants_budgeted":
        views = [global_image, *quadrant_crops(global_image)]
        paths: list[Path] = []
        for index, view in enumerate(views):
            path = sample_dir / f"budgeted_view_{index}_384.png"
            _save_if_missing(
                path,
                lambda view=view: fit_on_budget_canvas(view, 384),
            )
            paths.append(path)
        condition_meta = {
            **condition_meta,
            "view_budget": "5x384x384",
            "nominal_pixel_budget": 5 * 384 * 384,
        }
        return paths, condition_meta
    if image_mode != "global_plus_quadrants":
        raise ValueError(f"Unknown image mode: {image_mode}")

    crop_paths: list[Path] = []
    for index, crop in enumerate(quadrant_crops(global_image)):
        crop_path = sample_dir / f"quadrant_{index}.png"
        _save_if_missing(crop_path, lambda crop=crop: crop)
        crop_paths.append(crop_path)
    return [global_path, *crop_paths], condition_meta
