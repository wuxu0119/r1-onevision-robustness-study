from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Any


ANSWER_TAG = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
BOXED_CHOICE = re.compile(
    r"\\boxed\s*\{\s*([A-E])\s*\}",
    re.IGNORECASE,
)
_CHOICE_VALUE = (
    r"(?:(?:option|choice)\s*)?"
    r"(?:\\\[|\[|\(|\\\()?"
    r"[\s\\*_]*"
    r"(?:\\boxed\s*\{\s*)?"
    r"([A-E])\b"
)
FINAL_CHOICE_PATTERNS = [
    # Qualified labels are explicit even without a colon:
    # "the correct answer is B", "final choice: C".
    re.compile(
        r"[*_]{0,3}(?:the\s+)?(?:final|correct|best|closest)\s+"
        r"(?:answer|choice|option)\b"
        r"(?:\s+(?:provided|selected))?"
        r"[*_]{0,3}\s*(?:is\s*)?[*_]{0,3}\s*(?::|=)?\s*"
        + _CHOICE_VALUE,
        re.IGNORECASE,
    ),
    # An unqualified label needs an explicit delimiter. This matches
    # "Answer: B", "**Answer:** B", and "Answer is B", but not an
    # option-by-option discussion such as "Option D: ...".
    re.compile(
        r"[*_]{0,3}(?:answer|choice)\b[*_]{0,3}\s*"
        r"(?:(?:is)\b\s*|[*_]{0,3}\s*(?::|=)\s*[*_]{0,3}\s*)"
        + _CHOICE_VALUE,
        re.IGNORECASE,
    ),
    # Common concluding formulations without an answer label.
    re.compile(
        r"(?:corresponds?|matching)\s+to\s+(?:option|choice)\s*"
        + _CHOICE_VALUE,
        re.IGNORECASE,
    ),
]
PLAIN_CHOICE = re.compile(
    r"^\s*[\[(]?\s*([A-E])\s*[\]).]?\s*$",
    re.IGNORECASE,
)
LEADING_CHOICE = re.compile(
    r"^\s*[\[(]?\s*([A-E])\b",
    re.IGNORECASE,
)
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


def _single_choice_context(text: str, letter_end: int) -> bool:
    """Reject an apparent answer that immediately continues as a choice list."""
    continuation = text[letter_end : letter_end + 50]
    continuation = re.sub(r"^[\s*_.)\]}]+", "", continuation)
    return not bool(
        re.match(
            r"(?:[,/&]|\band\b|\bor\b)\s*(?:option|choice)?\s*[A-E]\b",
            continuation,
            re.IGNORECASE,
        )
    )


def extract_explicit_choice(output_text: str) -> tuple[str, str]:
    """Return the last high-confidence MC answer outside ``<answer>`` tags.

    Boxed choice letters and final answer/choice/option phrases are compared by
    their actual location in the output. This prevents an earlier boxed
    hypothesis from overriding a later plain final answer, and vice versa.
    """
    tail = output_text[-1600:]
    candidates: list[tuple[int, int, str, str]] = []
    for pattern in FINAL_CHOICE_PATTERNS:
        for match in pattern.finditer(tail):
            if _single_choice_context(tail, match.end(1)):
                candidates.append(
                    (
                        match.start(1),
                        1,
                        match.group(1).upper(),
                        "final_option_phrase",
                    )
                )
    for match in BOXED_CHOICE.finditer(tail):
        candidates.append(
            (
                match.start(1),
                2,
                match.group(1).upper(),
                "boxed",
            )
        )
    if not candidates:
        return "", ""
    _, _, letter, method = max(candidates)
    return letter, method


def extract_plain_choice(text: str) -> str:
    """Accept a bare option letter, but not an arbitrary letter in a tail."""
    match = PLAIN_CHOICE.fullmatch(text)
    return match.group(1).upper() if match else ""


def extract_leading_choice(text: str) -> str:
    """Accept a leading choice only when it is not the start of a list."""
    match = LEADING_CHOICE.match(text)
    if not match or not _single_choice_context(text, match.end(1)):
        return ""
    return match.group(1).upper()


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

    if len(truth) == 1 and truth.upper() in "ABCDE":
        tagged = ANSWER_TAG.findall(output_text)
        if tagged:
            # The final explicit answer block has absolute precedence over
            # text outside the block, including later commentary.
            candidate = tagged[-1].strip()
            method = "answer_tag"
            letter = extract_letter(candidate)
        else:
            letter, method = extract_explicit_choice(output_text)
            if letter:
                candidate = letter
            else:
                candidate, method = answer_span(output_text)
                letter = (
                    extract_plain_choice(candidate)
                    if method == "tail"
                    else extract_leading_choice(candidate)
                )
        return ScoreResult(
            extracted_answer=letter or candidate,
            parse_method=f"{method}_letter" if letter else method,
            is_correct=letter == truth.upper(),
            needs_manual_review=not bool(letter),
        )

    candidate, method = answer_span(output_text)
    normalized_truth = normalize_text(truth)
    normalized_candidate = normalize_text(candidate)
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
