"""Text + Blocked-but-Recoverable: mask essential slot + clarification channel."""

from __future__ import annotations

from typing import Any

from arb.schemas import MaskedSlot, TaskState, TaskSubstrate, TextTaskInstance
from arb.text.slot_identifier import select_slot


def mask_question(question: str, slot, mask_token: str) -> str:
    return question[: slot.start] + mask_token + question[slot.end :]


def build_blocked_instance(
    base: dict[str, Any],
    index: int,
    *,
    mask_token: str = "[UNKNOWN]",
    use_llm: bool = False,
    model_path: str | None = None,
) -> TextTaskInstance | None:
    question = base["question"]
    slot = select_slot(question, use_llm=use_llm, model_path=model_path)
    if slot is None:
        return None

    masked_q = mask_question(question, slot, mask_token)
    slot_id = f"slot_{slot.semantic_type}_{index}"

    ms = MaskedSlot(
        slot_id=slot_id,
        original_text=slot.original_text,
        original_value=slot.original_value,
        masked_token=mask_token,
        clarification_keywords=slot.clarification_keywords,
        semantic_type=slot.semantic_type,
    )

    split = base["split"]
    return TextTaskInstance(
        id=f"text_blocked_{split}_{index:05d}",
        substrate=TaskSubstrate.TEXT.value,
        state=TaskState.BLOCKED_BUT_RECOVERABLE.value,
        source_dataset=base["source_dataset"],
        source_id=base["source_id"],
        split=split,
        question=masked_q,
        gold_answer=base["gold_answer"],
        gold_answer_numeric=base.get("gold_answer_numeric"),
        masked_slots=[ms.to_dict()],
        clarification_enabled=True,
        answer_format="free_form",
        expected_behavior="ask_clarification_then_solve",
        transform_ops=["mask_essential_slot", "enable_ask_for_clarification"],
        metadata={"original_question": question},
    )
