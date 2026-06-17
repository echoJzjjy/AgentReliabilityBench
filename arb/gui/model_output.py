"""Extract executable GUI actions from verbose / reasoning model outputs (sim_layer v4)."""

from __future__ import annotations

import re
from dataclasses import dataclass

_ACTION_LINE = re.compile(
    r"^(?:"
    r"REPORT_FAILURE:\s*.+|"
    r"REQUEST_ACCESS:\s*.+|"
    r"ASK_USER:\s*.+|"
    r"OPEN_ATTACHMENT:\s*.+|"
    r"OPEN\s+\S+|"
    r"SET\s+\S+\.\S+\s+.+|"
    r"READ\s+\S+|"
    r"SAVE\s+\S+|"
    r"SUBMIT|"
    r"DONE"
    r")\s*$",
    re.I,
)

# Qwen3-Thinking / Phi-4 reasoning wrappers
_THINKING_BLOCKS = (
    re.compile(r"<think>.*?</think>", re.I | re.S),
    re.compile(r"<\|im_start\|>think.*?<\|im_end\|>", re.I | re.S),
    re.compile(r"<think>.*?</think>", re.I | re.S),
)


@dataclass(frozen=True)
class PreparedTurn:
    raw: str
    prepared: str
    extraction: str  # none | strip_thinking | last_action_line | verbatim


def strip_reasoning_blocks(text: str) -> str:
    out = text
    for pat in _THINKING_BLOCKS:
        out = pat.sub("", out)
    return out.strip()


def _clean_candidate(line: str) -> str:
    line = line.strip()
    line = re.sub(r"^#+\s*", "", line)
    line = re.sub(r"^(?:step\s*)?\d+[\.\):]\s*", "", line, flags=re.I)
    line = re.sub(r"^[-*]\s*", "", line)
    line = line.strip().strip("`")
    return line


def extract_action_line(text: str) -> str | None:
    """Return the last line that looks like a single GUI action."""
    for line in reversed(text.splitlines()):
        clean = _clean_candidate(line)
        if not clean:
            continue
        upper = clean.upper()
        if upper in ("SUBMIT", "DONE"):
            return "SUBMIT"
        if _ACTION_LINE.match(clean):
            return clean
        m = re.search(r"(REPORT_FAILURE:\s*.+)$", clean, re.I)
        if m:
            return m.group(1).strip()
    return None


def prepare_agent_turn(raw: str, *, reasoning_model: bool = False) -> PreparedTurn:
    """Normalize model text before action_parser.

    Instruct models: preserve multi-line SAVE bodies verbatim (only strip thinking tags).
    Reasoning models: strip thinking blocks and take the last executable action line.
    """
    text = (raw or "").strip()
    if not text:
        return PreparedTurn(raw=raw, prepared=text, extraction="none")

    stripped = strip_reasoning_blocks(text)
    if stripped != text:
        text = stripped
        base = "strip_thinking"
    else:
        base = "verbatim"

    upper = text.upper()
    if upper.startswith("SAVE ") or upper.startswith("REPORT_FAILURE:"):
        return PreparedTurn(raw=raw, prepared=text, extraction=base)

    if not reasoning_model:
        return PreparedTurn(raw=raw, prepared=text, extraction=base)

    action = extract_action_line(text)
    if action and action != text:
        return PreparedTurn(raw=raw, prepared=action, extraction=f"{base}+last_action_line")

    return PreparedTurn(raw=raw, prepared=text, extraction=base)


def is_reasoning_model_slug(model_slug: str | None) -> bool:
    if not model_slug:
        return False
    low = model_slug.lower()
    return any(k in low for k in ("thinking", "reasoning", "r1", "o1"))
