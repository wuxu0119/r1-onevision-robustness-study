from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "07_analyze_extension.py"
SPEC = importlib.util.spec_from_file_location("extension_analysis", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


class ExtensionAnalysisTests(unittest.TestCase):
    def test_exact_mcnemar_is_symmetric(self) -> None:
        self.assertAlmostEqual(
            analysis.exact_mcnemar_p(12, 28),
            analysis.exact_mcnemar_p(28, 12),
        )
        self.assertEqual(analysis.exact_mcnemar_p(0, 0), 1.0)

    def test_selection_metrics(self) -> None:
        accepted = np.array([True, True, False, False])
        correct = np.array([True, False, False, False])
        result = analysis.selection_metrics(
            accepted,
            correct,
            repeats=1000,
            seed=7,
        )
        self.assertEqual(result["n"], 4)
        self.assertEqual(result["accepted_n"], 2)
        self.assertAlmostEqual(result["coverage_pct"], 50.0)
        self.assertAlmostEqual(result["selective_accuracy_pct"], 50.0)
        self.assertAlmostEqual(result["rejected_error_pct"], 100.0)


if __name__ == "__main__":
    unittest.main()
