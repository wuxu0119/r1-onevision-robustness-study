from __future__ import annotations

import unittest

from r1eval.scoring import score_output


class ConservativeScoringTests(unittest.TestCase):
    def test_answer_tag_choice(self) -> None:
        result = score_output("<answer>B</answer>", "B")
        self.assertTrue(result.is_correct)
        self.assertEqual(result.extracted_answer, "B")

    def test_boxed_choice_after_final_phrase_newline(self) -> None:
        result = score_output(
            "Thus, the correct answer is:\n\\[ \\boxed{B} \\]",
            "B",
        )
        self.assertTrue(result.is_correct)
        self.assertEqual(result.extracted_answer, "B")
        self.assertEqual(result.parse_method, "boxed_letter")

    def test_final_phrase_can_override_earlier_box(self) -> None:
        result = score_output(
            "An early calculation suggested \\boxed{A}.\n"
            "After checking the diagram, the final answer is C.",
            "C",
        )
        self.assertTrue(result.is_correct)
        self.assertEqual(result.extracted_answer, "C")
        self.assertEqual(result.parse_method, "final_option_phrase_letter")

    def test_later_box_can_override_earlier_final_phrase(self) -> None:
        result = score_output(
            "The answer is A. That used the wrong sign.\n"
            "Recomputing gives \\boxed{D}.",
            "D",
        )
        self.assertTrue(result.is_correct)
        self.assertEqual(result.extracted_answer, "D")

    def test_final_answer_tag_has_absolute_precedence(self) -> None:
        result = score_output(
            "<answer>B</answer>\nA later comment mentions \\boxed{C}.",
            "B",
        )
        self.assertTrue(result.is_correct)
        self.assertEqual(result.extracted_answer, "B")
        self.assertEqual(result.parse_method, "answer_tag_letter")

    def test_markdown_answer_overrides_option_discussion(self) -> None:
        result = score_output(
            "Option D: an attractive distractor.\n"
            "After checking the graph, **Answer:** B. Bt ORG2 isolate",
            "B",
        )
        self.assertTrue(result.is_correct)
        self.assertEqual(result.extracted_answer, "B")
        self.assertEqual(result.parse_method, "final_option_phrase_letter")

    def test_unqualified_option_in_discussion_is_not_a_final_answer(self) -> None:
        result = score_output(
            "Option A: first hypothesis.\nOption D: another possibility.",
            "D",
        )
        self.assertFalse(result.is_correct)
        self.assertTrue(result.needs_manual_review)

    def test_multiple_letters_after_answer_label_are_not_single_choice(self) -> None:
        result = score_output(
            "The correct options are A, B, and D.\nAnswer: A, B, D",
            "A",
        )
        self.assertFalse(result.is_correct)
        self.assertTrue(result.needs_manual_review)

    def test_last_answer_tag_wins(self) -> None:
        result = score_output(
            "<answer>A</answer>\nCorrection: <answer>E</answer>",
            "E",
        )
        self.assertTrue(result.is_correct)
        self.assertEqual(result.extracted_answer, "E")

    def test_answer_choice_list_is_not_treated_as_prediction(self) -> None:
        result = score_output(
            "Answer choices:\nA. red\nB. blue\nC. green\nD. yellow",
            "D",
        )
        self.assertFalse(result.is_correct)
        self.assertTrue(result.needs_manual_review)
        self.assertNotEqual(result.extracted_answer, "D")

    def test_plain_letter_only_tail_is_accepted(self) -> None:
        result = score_output("B.", "B")
        self.assertTrue(result.is_correct)
        self.assertEqual(result.extracted_answer, "B")

    def test_numeric_behavior_is_unchanged(self) -> None:
        result = score_output(
            "An earlier estimate was 41.\nThe final answer is 42.",
            "42",
        )
        self.assertTrue(result.is_correct)

    def test_free_response_answer_tag_behavior_is_unchanged(self) -> None:
        result = score_output("<answer>10 cm</answer>", "10 cm")
        self.assertTrue(result.is_correct)
        self.assertEqual(result.parse_method, "answer_tag_numeric")


if __name__ == "__main__":
    unittest.main()
