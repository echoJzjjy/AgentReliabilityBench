"""Identify removable specification fragments from LiveCodeBench problem statements."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class SpecFragment:
    section_type: str
    original_text: str
    start: int
    end: int
    clarification_keywords: list[str]
    reveal_channel: str


SECTION_PATTERNS: list[tuple[str, str, list[str], str]] = [
    (
        "constraint",
        r"(Constraints?:[^\n]*(?:\n(?:\s*(?:\d+\.|[-*•])[^\n]+))+)",
        ["constraint", "bound", "limit", "range", "边界", "约束"],
        "ask_user",
    ),
    (
        "io_format",
        r"((?:Input|Output)(?:\s+Format)?:[^\n]+(?:\n(?!(?:Example|Constraints|Note|Input|Output))[^\n]+)*)",
        ["input", "output", "format", "输入", "输出", "格式"],
        "open_spec",
    ),
    (
        "edge_case",
        r"((?:Note|Notice|Important|Edge\s+case)[^\n]*(?:\n(?!(?:Example|Constraints|Input|Output))[^\n]+)*)",
        ["edge", "exception", "special", "异常", "边界情况"],
        "request_example",
    ),
    (
        "example_detail",
        r"(Example\s+\d+:[^\n]*(?:\n(?!(?:Example|Constraints|Note|Input|Output))[^\n]+)*)",
        ["example", "样例", "示例", "test case"],
        "request_example",
    ),
]


def find_spec_fragments(question_content: str) -> list[SpecFragment]:
    """Return candidate spec fragments sorted by start position."""
    fragments: list[SpecFragment] = []
    for section_type, pattern, keywords, channel in SECTION_PATTERNS:
        for m in re.finditer(pattern, question_content, flags=re.IGNORECASE):
            text = m.group(1).strip()
            if len(text) < 20:
                continue
            fragments.append(
                SpecFragment(
                    section_type=section_type,
                    original_text=text,
                    start=m.start(1),
                    end=m.end(1),
                    clarification_keywords=keywords,
                    reveal_channel=channel,
                )
            )
    fragments.sort(key=lambda f: f.start)
    chosen: list[SpecFragment] = []
    occupied: list[tuple[int, int]] = []
    for frag in sorted(fragments, key=lambda f: (-len(f.original_text), f.start)):
        overlap = any(not (frag.end <= s or frag.start >= e) for s, e in occupied)
        if overlap:
            continue
        chosen.append(frag)
        occupied.append((frag.start, frag.end))
    chosen.sort(key=lambda f: f.start)
    return chosen


def mask_spec(question_content: str, fragment: SpecFragment, placeholder: str) -> str:
    return question_content[: fragment.start] + placeholder + question_content[fragment.end :]


def _fallback_paragraph_fragment(question_content: str, index: int) -> SpecFragment | None:
    """When regex sections miss, mask a substantive paragraph so blocked coverage stays high."""
    paragraphs: list[tuple[int, int, str]] = []
    for m in re.finditer(r"(?:^|\n\n)([^\n][^\n]{39,}?)(?=\n\n|$)", question_content, re.DOTALL):
        chunk = m.group(1).strip()
        if len(chunk) < 40:
            continue
        if chunk.startswith("```"):
            continue
        start = question_content.find(chunk, m.start(1))
        if start < 0:
            continue
        paragraphs.append((start, start + len(chunk), chunk))
    if not paragraphs:
        return None
    start, end, chunk = paragraphs[index % len(paragraphs)]
    return SpecFragment(
        section_type="problem_detail",
        original_text=chunk,
        start=start,
        end=end,
        clarification_keywords=[
            "detail",
            "requirement",
            "problem",
            "说明",
            "要求",
            "题目",
            "spec",
            "format",
            "input",
            "output",
        ],
        reveal_channel="ask_user",
    )


def select_spec_fragment(question_content: str, index: int = 0) -> SpecFragment | None:
    frags = find_spec_fragments(question_content)
    if frags:
        return frags[index % len(frags)]
    return _fallback_paragraph_fragment(question_content, index)
