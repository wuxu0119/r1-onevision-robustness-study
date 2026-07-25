from __future__ import annotations

from typing import Any


PAPER_INSTRUCTION = (
    "First output the thinking process in <think> </think> tags and then "
    "output the final answer in <answer> </answer> tags."
)

EVIDENCE_INSTRUCTION = """Inspect the visual input carefully before solving.
First output an <evidence> block containing:
1. the visual values, labels, and entities relevant to the question;
2. their spatial, mathematical, or logical relations;
3. any observation that is uncertain;
4. which image or panel each fact comes from.
Then verify the listed evidence against all provided views. Reason only after
this verification. Put the reasoning in <think> </think> tags and the shortest
possible final answer in <answer> </answer> tags."""


def compose_task(question: Any, choices: Any) -> str:
    question_text = "" if question is None else str(question).strip()
    choices_text = "" if choices is None else str(choices).strip()
    if choices_text and "choices:" not in question_text.lower():
        question_text += f"\nChoices:\n{choices_text}"
    return question_text


def build_prompt(
    *,
    question: Any,
    choices: Any,
    prompt_mode: str,
    image_mode: str,
) -> str:
    task = compose_task(question, choices)
    if prompt_mode == "paper":
        instruction = PAPER_INSTRUCTION
    elif prompt_mode == "evidence_first":
        instruction = EVIDENCE_INSTRUCTION
    else:
        raise ValueError(f"Unknown prompt mode: {prompt_mode}")

    view_note = ""
    if image_mode in {
        "global_plus_quadrants",
        "global_plus_quadrants_budgeted",
    }:
        view_note = (
            "\nThe first image is the global view. The following images are "
            "overlapping local views of the same visual input. Use them to "
            "check small text, symbols, and the relevant panel."
        )
    return f"{instruction}{view_note}\n\n{task}"
