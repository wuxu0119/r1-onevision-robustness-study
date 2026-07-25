from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_inference_script",
    ROOT / "scripts" / "02_run_inference.py",
)
assert SPEC is not None and SPEC.loader is not None
RUN_INFERENCE = importlib.util.module_from_spec(SPEC)
RUNNER_IMPORT_ERROR = None
try:
    SPEC.loader.exec_module(RUN_INFERENCE)
except ModuleNotFoundError as exc:  # local document-build runtime lacks ML deps
    RUNNER_IMPORT_ERROR = exc


@unittest.skipIf(
    RUNNER_IMPORT_ERROR is not None,
    f"optional inference dependencies unavailable: {RUNNER_IMPORT_ERROR}",
)
class ExtensionRunnerTests(unittest.TestCase):
    def test_completion_key_distinguishes_replicates(self) -> None:
        common = {
            "row_id": 3,
            "condition": "clean",
            "prompt_mode": "paper",
            "image_mode": "global",
        }
        self.assertNotEqual(
            RUN_INFERENCE.completion_key({**common, "replicate": 0}),
            RUN_INFERENCE.completion_key({**common, "replicate": 1}),
        )
        self.assertEqual(
            RUN_INFERENCE.completion_key(common),
            RUN_INFERENCE.completion_key({**common, "replicate": 0}),
        )

    def test_generation_seed_is_repeatable_and_replicate_specific(self) -> None:
        arguments = {
            "base_seed": 20260723,
            "row_id": 17,
            "phase": "confidence",
            "model_key": "rl",
            "condition": "clean",
            "prompt_mode": "paper",
            "image_mode": "global",
        }
        seed_0 = RUN_INFERENCE.deterministic_generation_seed(
            **arguments,
            replicate=0,
        )
        self.assertEqual(
            seed_0,
            RUN_INFERENCE.deterministic_generation_seed(
                **arguments,
                replicate=0,
            ),
        )
        self.assertNotEqual(
            seed_0,
            RUN_INFERENCE.deterministic_generation_seed(
                **arguments,
                replicate=1,
            ),
        )
        self.assertGreaterEqual(seed_0, 0)
        self.assertLess(seed_0, 2**63)

    def test_confidence_generation_overrides(self) -> None:
        config = json.loads(
            (ROOT / "configs" / "extension.json").read_text(encoding="utf-8")
        )
        phase = config["phases"]["confidence"]
        generation = RUN_INFERENCE.resolve_generation(
            config,
            "confidence",
            phase,
        )
        self.assertEqual(generation["max_new_tokens"], 2048)
        self.assertEqual(generation["temperature"], 0.7)
        self.assertEqual(generation["top_p"], 0.9)
        self.assertEqual(generation["top_k"], 50)
        self.assertEqual(RUN_INFERENCE.replicate_ids(phase), list(range(8)))

    def test_extension_phase_matrix(self) -> None:
        config = json.loads(
            (ROOT / "configs" / "extension.json").read_text(encoding="utf-8")
        )
        controls = config["phases"]["controls"]
        self.assertEqual(controls["models"], ["base", "sft", "rl"])
        self.assertEqual(
            controls["conditions"],
            [
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
            ],
        )
        self.assertEqual(controls["subset_size"], 300)

        confidence = config["phases"]["confidence"]
        self.assertEqual(confidence["models"], ["rl"])
        self.assertEqual(
            confidence["conditions"],
            ["clean", "distractor_3", "downsample_25"],
        )
        self.assertEqual(confidence["subset_size"], 300)

        gate = config["phases"]["gate_generalization"]
        self.assertEqual(gate["models"], ["base", "sft", "rl"])
        self.assertEqual(len(gate["conditions"]), 5)

        multiview = config["phases"]["multiview_budgeted"]
        self.assertEqual(multiview["models"], ["rl"])
        self.assertEqual(
            multiview["image_modes"],
            ["global_budgeted", "global_plus_quadrants_budgeted"],
        )


if __name__ == "__main__":
    unittest.main()
