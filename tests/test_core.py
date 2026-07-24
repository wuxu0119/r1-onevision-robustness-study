from __future__ import annotations

import base64
import io
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from r1eval.data import decode_image
from r1eval.modeling import processor_id_for_model
from r1eval.perturbations import prepare_condition_images
from r1eval.scoring import score_output


def encoded_image(color: str) -> str:
    image = Image.new("RGB", (320, 180), color)
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return base64.b64encode(buffer.getvalue()).decode()


class ScoringTests(unittest.TestCase):
    def test_choice_answer(self) -> None:
        result = score_output("<answer>B</answer>", "B")
        self.assertTrue(result.is_correct)
        self.assertEqual(result.extracted_answer, "B")

    def test_numeric_answer(self) -> None:
        result = score_output("The final answer is 42.", "42")
        self.assertTrue(result.is_correct)

    def test_wrong_answer(self) -> None:
        result = score_output("<answer>A</answer>", "C")
        self.assertFalse(result.is_correct)


class ModelingTests(unittest.TestCase):
    def test_sft_uses_compatible_base_processor(self) -> None:
        self.assertEqual(
            processor_id_for_model("Fancy-MLLM/R1-Onevision-7B"),
            "Qwen/Qwen2.5-VL-7B-Instruct",
        )
        self.assertEqual(
            processor_id_for_model("Fancy-MLLM/R1-Onevision-7B-RL"),
            "Fancy-MLLM/R1-Onevision-7B-RL",
        )


class ImageTests(unittest.TestCase):
    def test_decode_base64(self) -> None:
        decoded = decode_image(encoded_image("red"))
        self.assertEqual(decoded.size, (320, 180))

    def test_all_conditions_and_multiview(self) -> None:
        dataset = [
            {"image": encoded_image(color)}
            for color in ["red", "blue", "green", "yellow"]
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for condition in [
                "clean",
                "downsample_50",
                "downsample_25",
                "distractor_1",
                "distractor_3",
            ]:
                paths, _ = prepare_condition_images(
                    dataset=dataset,
                    row_id=0,
                    condition=condition,
                    image_mode="global_plus_quadrants",
                    derived_dir=root,
                    seed=1,
                )
                self.assertEqual(len(paths), 5)
                self.assertTrue(all(path.exists() for path in paths))


if __name__ == "__main__":
    unittest.main()
