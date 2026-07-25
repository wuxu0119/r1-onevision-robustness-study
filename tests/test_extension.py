from __future__ import annotations

import base64
import io
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from r1eval.perturbations import prepare_condition_images


def encoded_image(color: str) -> str:
    image = Image.new("RGB", (320, 180), color)
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def image_size(path: str | Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


class MatchedPanelControlTests(unittest.TestCase):
    def dataset(self) -> list[dict]:
        return [
            {"image": encoded_image(color), "category": f"C{index}"}
            for index, color in enumerate(
                ["red", "blue", "green", "yellow", "purple", "orange"]
            )
        ]

    def test_fixed_controls_and_semantic_seeds_match_geometry(self) -> None:
        dataset = self.dataset()
        conditions = [
            "control_blank_1_left",
            "control_noise_1_left",
            "distractor_1_seed0_left",
            "distractor_1_seed1_left",
            "distractor_1_seed2_left",
        ]
        with tempfile.TemporaryDirectory() as temporary:
            observations = {}
            for condition in conditions:
                paths, metadata = prepare_condition_images(
                    dataset=dataset,
                    row_id=0,
                    condition=condition,
                    image_mode="global",
                    derived_dir=Path(temporary),
                    seed=20260723,
                )
                observations[condition] = {
                    "size": image_size(paths[0]),
                    "target_panel_index": metadata["target_panel_index"],
                    "distractor_ids": tuple(metadata.get("distractor_ids", [])),
                }

        self.assertEqual(
            {observation["size"] for observation in observations.values()},
            {(1024, 512)},
        )
        self.assertEqual(
            {
                observation["target_panel_index"]
                for observation in observations.values()
            },
            {0},
        )
        semantic_ids = {
            observations[condition]["distractor_ids"]
            for condition in conditions[2:]
        }
        self.assertEqual(len(semantic_ids), 3)

    def test_left_and_right_reuse_content(self) -> None:
        dataset = self.dataset()
        with tempfile.TemporaryDirectory() as temporary:
            left_paths, left_meta = prepare_condition_images(
                dataset=dataset,
                row_id=0,
                condition="distractor_1_seed1_left",
                image_mode="global",
                derived_dir=temporary,
                seed=20260723,
            )
            right_paths, right_meta = prepare_condition_images(
                dataset=dataset,
                row_id=0,
                condition="distractor_1_seed1_right",
                image_mode="global",
                derived_dir=temporary,
                seed=20260723,
            )
            self.assertEqual(left_meta["distractor_ids"], right_meta["distractor_ids"])
            self.assertEqual(left_meta["target_panel_index"], 0)
            self.assertEqual(right_meta["target_panel_index"], 1)
            self.assertEqual(image_size(left_paths[0]), image_size(right_paths[0]))

    def test_budgeted_view_geometry(self) -> None:
        dataset = self.dataset()
        with tempfile.TemporaryDirectory() as temporary:
            global_paths, global_meta = prepare_condition_images(
                dataset=dataset,
                row_id=0,
                condition="clean",
                image_mode="global_budgeted",
                derived_dir=temporary,
                seed=20260723,
            )
            multi_paths, multi_meta = prepare_condition_images(
                dataset=dataset,
                row_id=0,
                condition="clean",
                image_mode="global_plus_quadrants_budgeted",
                derived_dir=temporary,
                seed=20260723,
            )
            self.assertEqual(len(global_paths), 1)
            self.assertEqual(image_size(global_paths[0]), (864, 864))
            self.assertEqual(len(multi_paths), 5)
            self.assertTrue(
                all(image_size(path) == (384, 384) for path in multi_paths)
            )
            self.assertEqual(global_meta["nominal_pixel_budget"], 864 * 864)
            self.assertEqual(
                multi_meta["nominal_pixel_budget"], 5 * 384 * 384
            )

    def test_controls_and_semantic_seeds_match_random_geometry(self) -> None:
        dataset = [
            {"image": encoded_image(color), "category": f"C{index}"}
            for index, color in enumerate(
                ["red", "blue", "green", "yellow", "purple", "orange"]
            )
        ]
        conditions = [
            "distractor_1",
            "distractor_1_seed1",
            "distractor_1_seed2",
            "control_blank_1",
            "control_noise_1",
        ]
        with tempfile.TemporaryDirectory() as temporary:
            observations = {}
            for condition in conditions:
                paths, metadata = prepare_condition_images(
                    dataset=dataset,
                    row_id=0,
                    condition=condition,
                    image_mode="global",
                    derived_dir=Path(temporary),
                    seed=20260723,
                )
                observations[condition] = {
                    "size": image_size(paths[0]),
                    "target_panel_index": metadata["target_panel_index"],
                    "distractor_ids": tuple(metadata.get("distractor_ids", [])),
                }

        self.assertEqual(
            {observation["size"] for observation in observations.values()},
            {(1024, 512)},
        )
        self.assertEqual(
            {
                observation["target_panel_index"]
                for observation in observations.values()
            },
            {observations["distractor_1"]["target_panel_index"]},
        )
        semantic_ids = {
            observations[condition]["distractor_ids"]
            for condition in conditions[:3]
        }
        self.assertEqual(len(semantic_ids), 3)

    def test_noise_control_is_deterministic(self) -> None:
        dataset = self.dataset()
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_paths, _ = prepare_condition_images(
                dataset=dataset,
                row_id=0,
                condition="control_noise_1",
                image_mode="global",
                derived_dir=first,
                seed=11,
            )
            second_paths, _ = prepare_condition_images(
                dataset=dataset,
                row_id=0,
                condition="control_noise_1",
                image_mode="global",
                derived_dir=second,
                seed=11,
            )
            self.assertEqual(
                Path(first_paths[0]).read_bytes(),
                Path(second_paths[0]).read_bytes(),
            )


if __name__ == "__main__":
    unittest.main()
