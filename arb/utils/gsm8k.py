"""GSM8K parsing utilities."""

from __future__ import annotations

import re
from typing import Any

# Prefer final-answer channels over incidental numbers in chain-of-thought.
_FINAL_ANSWER_PATTERNS = [
    re.compile(r"####\s*(-?\d+(?:\.\d+)?)", re.I),
    re.compile(r"\\boxed\{([^}]+)\}", re.I),
    re.compile(
        r"(?:final\s+answer|answer)\s*[:：]\s*\$?\s*(-?\d+(?:\.\d+)?)",
        re.I,
    ),
    re.compile(r"(?:final\s+answer|answer)\s*[:：]\s*\$?\\boxed\{(-?\d+(?:\.\d+)?)\}", re.I),
]


def extract_final_answer(answer_field: str) -> tuple[str, float | None]:
    """Parse GSM8K answer field: '... #### 42' -> ('42', 42.0)."""
    if "####" in answer_field:
        final = answer_field.split("####")[-1].strip()
    else:
        final = answer_field.strip()
    numeric = parse_numeric(final)
    return final, numeric


def parse_numeric(text: str) -> float | None:
    """Parse the first number in *short* text (gold fields, single tokens)."""
    text = text.replace(",", "").strip()
    m = re.search(r"-?\d+(?:\.\d+)?", text)
    if not m:
        return None
    try:
        return float(m.group())
    except ValueError:
        return None


def extract_agent_numeric_answer(text: str) -> float | None:
    """Extract the model's intended final numeric answer from a long response."""
    if not text or not text.strip():
        return None

    for pat in _FINAL_ANSWER_PATTERNS:
        matches = list(pat.finditer(text))
        if matches:
            val = parse_numeric(matches[-1].group(1))
            if val is not None:
                return val

    if "####" in text:
        val = parse_numeric(text.split("####")[-1])
        if val is not None:
            return val

    # Last non-empty line often holds the final answer in CoT outputs.
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for line in reversed(lines[-8:]):
        val = parse_numeric(line)
        if val is not None:
            return val

    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if nums:
        try:
            return float(nums[-1])
        except ValueError:
            return None
    return None


def has_definitive_final_numeric(text: str) -> bool:
    """True if the response states a clear final numeric result (not just CoT digits)."""
    if not text:
        return False
    for pat in _FINAL_ANSWER_PATTERNS:
        if pat.search(text):
            return True
    if "####" in text:
        return True
    lower = text.lower()
    if re.search(r"(?:final\s+answer|answer)\s*[:：]", lower):
        return extract_agent_numeric_answer(text) is not None
    return False


def extract_mcq_letter(text: str) -> str | None:
    """Extract the intended multiple-choice letter from agent output."""
    if not text:
        return None
    upper = text.upper()
    letter_patterns = [
        re.compile(r"(?:FINAL\s+)?ANSWER\s*[:：]\s*\(?([A-D])\)?"),
        re.compile(r"\\boxed\{([A-D])\}"),
        re.compile(r"(?:CHOOSE|SELECT|PICK)\s+(?:OPTION\s+)?([A-D])\b"),
        re.compile(r"OPTION\s+([A-D])\b"),
    ]
    for pat in letter_patterns:
        matches = list(pat.finditer(upper))
        if matches:
            return matches[-1].group(1)
    letters = re.findall(r"\b([A-D])\b", upper)
    return letters[-1] if letters else None


def normalize_gsm8k_row(row: dict[str, Any], idx: int, split: str) -> dict[str, Any]:
    question = row["question"].strip()
    answer_raw = row["answer"].strip()
    final_str, final_num = extract_final_answer(answer_raw)
    source_id = row.get("id", f"{split}_{idx}")
    return {
        "source_dataset": "gsm8k",
        "source_id": str(source_id),
        "split": split,
        "question": question,
        "answer_full": answer_raw,
        "gold_answer": final_str,
        "gold_answer_numeric": final_num,
    }
