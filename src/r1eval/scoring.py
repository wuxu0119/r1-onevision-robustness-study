from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Any


ANSWER_TAG = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
FINAL_PATTERNS = [
    re.compile(
        r"(?:final\s+answer|answer|therefore|thus)\s*(?:is|:|=)?\s*(.*)",
        re.IGNORECASE,
    ),
]
LETTER_PATTERN = re.compile(r"(?<![A-Z0-9])([A-E])(?![A-Z0-9])", re.IGNORECASE)
NUMBER_PATTERN = re.compile(
    r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?"
)


@dataclass
class ScoreResult:
    extracted_answer: str
    parse_method: str
    is_correct: bool
    needs_manual_review: bool


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", "" if value is None else str(value))
    text = text.lower().strip()
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[`*_{}$\\]", "", text)
    text = re.sub(r"[。．.,;:!?！？]", "", text)
    return text


def answer_span(output_text: str) -> tuple[str, str]:
    tagged = ANSWER_TAG.findall(output_text)
    if tagged:
        return tagged[-1].strip(), "answer_tag"

    tail = output_text[-800:]
    for pattern in FINAL_PATTERNS:
        matches = pattern.findall(tail)
        if matches:
            return matches[-1].strip(), "final_phrase"
    return tail.strip(), "tail"


def extract_letter(text: str) -> str:
    matches = LETTER_PATTERN.findall(text.upper())
    return matches[-1].upper() if matches else ""


def extract_last_number(text: str) -> float | None:
    matches = NUMBER_PATTERN.findall(text)
    if not matches:
        return None
    try:
        return float(matches[-1].replace(",", ""))
    except ValueError:
        return None


def _numeric_equal(prediction: str, truth: str) -> bool:
    predicted_number = extract_last_number(prediction)
    truth_number = extract_last_number(truth)
    if predicted_number is None or truth_number is None:
        return False
    return math.isclose(predicted_number, truth_number, rel_tol=1e-5, abs_tol=1e-5)


def score_output(output_text: str, ground_truth: Any, choices: Any = "") -> ScoreResult:
    truth = "" if ground_truth is None else str(ground_truth).strip()
    candidate, method = answer_span(output_text)
    normalized_truth = normalize_text(truth)
    normalized_candidate = normalize_text(candidate)

    if len(truth) == 1 and truth.upper() in "ABCDE":
        letter = extract_letter(candidate)
        return ScoreResult(
            extracted_answer=letter or candidate,
            parse_method=f"{method}_letter" if letter else method,
            is_correct=letter == truth.upper(),
            needs_manual_review=not bool(letter),
        )

    exact = bool(normalized_truth) and (
        normalized_candidate == normalized_truth
        or normalized_truth in normalized_candidate
    )
    numeric = _numeric_equal(candidate, truth)
    correct = exact or numeric
    needs_review = not correct and (
        method == "tail"
        or not normalized_candidate
        or (not normalized_truth and not str(choices).strip())
    )
    return ScoreResult(
        extracted_answer=candidate,
        parse_method=f"{method}_numeric" if numeric else method,
        is_correct=correct,
        needs_manual_review=needs_review,
    )

