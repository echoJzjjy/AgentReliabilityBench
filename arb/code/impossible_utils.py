"""Shared helpers for impossible code task transforms."""

from __future__ import annotations

import json
from typing import Any

# Keep agent-visible prompts bounded (LiveCodeBench private tests can be huge).
MAX_QUESTION_CONTENT_CHARS = 32_000
MAX_INJECTION_INPUT_CHARS = 200


def compact_test_input(raw: Any, *, max_len: int = MAX_INJECTION_INPUT_CHARS) -> str:
    """Short summary of a test input for natural-language spec injection."""
    text = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
    if len(text) <= max_len:
        return text
    return f"{text[:max_len]}… [truncated, total {len(text)} chars]"


def pick_private_test_for_conflict(
    private_tests: list[dict[str, Any]], rng, *, max_input_chars: int = 2000
) -> dict[str, Any]:
    """Prefer private tests with small inputs so spec text stays in context limits."""
    small = [t for t in private_tests if len(str(t.get("input", ""))) <= max_input_chars]
    pool = small if small else private_tests
    return pool[rng.randint(0, len(pool) - 1)]


def question_content_within_limit(content: str, limit: int = MAX_QUESTION_CONTENT_CHARS) -> bool:
    return len(content) <= limit
