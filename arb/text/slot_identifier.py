"""Identify maskable numeric slots in GSM8K questions (rules + optional local LLM)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


# Context patterns -> semantic type + clarification keywords
SLOT_PATTERNS: list[tuple[str, str, list[str]]] = [
    (r"\$[\d,]+(?:\.\d+)?", "price", ["price", "cost", "dollar", "each", "per"]),
    (r"[\d,]+(?:\.\d+)?\s*(?:dollars?|cents?)", "price", ["price", "cost", "dollar"]),
    (r"[\d,]+(?:\.\d+)?\s*(?:hours?|minutes?|days?|weeks?|months?|years?)", "duration", ["time", "hour", "minute", "day", "long", "duration"]),
    (r"[\d,]+(?:\.\d+)?\s*(?:miles?|km|kilometers?|meters?)", "distance", ["distance", "mile", "km", "far"]),
    (r"[\d,]+(?:\.\d+)?\s*(?:people|students?|kids?|children|adults?|items?|apples?|books?|tickets?)", "count", ["how many", "number", "count", "people", "items"]),
    (r"[\d,]+(?:\.\d+)?\s*(?:per|each|every)", "rate", ["rate", "per", "each", "every", "speed"]),
    (r"\b[\d,]+(?:\.\d+)?\b", "quantity", ["value", "number", "amount", "how much", "many"]),
]


@dataclass
class CandidateSlot:
    original_text: str
    original_value: str
    start: int
    end: int
    semantic_type: str
    clarification_keywords: list[str]
    score: float


def _score_slot(text: str, semantic_type: str, question: str) -> float:
    """Prefer slots that appear early and have rich context."""
    pos = question.find(text)
    early_bonus = max(0, 1.0 - pos / max(len(question), 1))
    type_bonus = {"price": 0.3, "count": 0.25, "duration": 0.2, "rate": 0.2}.get(semantic_type, 0.1)
    # Avoid masking "1" or trivial digits
    digits = re.sub(r"[^\d.]", "", text)
    trivial_penalty = -0.5 if digits in ("1", "2", "0") else 0.0
    return early_bonus + type_bonus + trivial_penalty


def find_slots_rule_based(question: str, max_slots: int = 1) -> list[CandidateSlot]:
    seen_spans: set[tuple[int, int]] = set()
    candidates: list[CandidateSlot] = []

    for pattern, sem_type, keywords in SLOT_PATTERNS:
        for m in re.finditer(pattern, question, flags=re.IGNORECASE):
            span = (m.start(), m.end())
            if span in seen_spans:
                continue
            seen_spans.add(span)
            orig = m.group()
            val = re.sub(r"[^\d.\-]", "", orig.replace(",", "")) or orig
            candidates.append(
                CandidateSlot(
                    original_text=orig,
                    original_value=val,
                    start=m.start(),
                    end=m.end(),
                    semantic_type=sem_type,
                    clarification_keywords=keywords,
                    score=_score_slot(orig, sem_type, question),
                )
            )

    candidates.sort(key=lambda c: -c.score)
    return candidates[:max_slots]


def find_slots_with_llm(
    question: str,
    model_path: str,
    max_slots: int = 1,
    max_new_tokens: int = 256,
) -> list[CandidateSlot]:
    """Use local HuggingFace model to pick one essential numeric slot."""
    rule_fallback = find_slots_rule_based(question, max_slots)
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        return rule_fallback

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    prompt = f"""You are building a benchmark. Given this math word problem, identify ONE numeric quantity that is essential to solve it.
Return JSON only: {{"original_text": "<exact substring from problem>", "semantic_type": "price|count|duration|rate|quantity", "clarification_keywords": ["word1", "word2"]}}

Problem:
{question}
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

    try:
        # Extract JSON object
        m = re.search(r"\{[^{}]+\}", text, re.DOTALL)
        if not m:
            return rule_fallback
        obj = json.loads(m.group())
        orig = obj["original_text"]
        idx = question.find(orig)
        if idx < 0:
            return rule_fallback
        return [
            CandidateSlot(
                original_text=orig,
                original_value=re.sub(r"[^\d.\-]", "", orig.replace(",", "")) or orig,
                start=idx,
                end=idx + len(orig),
                semantic_type=obj.get("semantic_type", "quantity"),
                clarification_keywords=obj.get("clarification_keywords", ["value", "number"]),
                score=1.0,
            )
        ][:max_slots]
    except (json.JSONDecodeError, KeyError):
        return rule_fallback


def select_slot(
    question: str,
    *,
    use_llm: bool = False,
    model_path: str | None = None,
    max_slots: int = 1,
) -> CandidateSlot | None:
    if use_llm and model_path:
        slots = find_slots_with_llm(question, model_path, max_slots)
    else:
        slots = find_slots_rule_based(question, max_slots)
    return slots[0] if slots else None
