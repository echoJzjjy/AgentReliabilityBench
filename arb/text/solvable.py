"""Text + Solvable: GSM8K with unified interface."""

from __future__ import annotations

from typing import Any

from arb.schemas import TaskState, TaskSubstrate, TextTaskInstance


def build_solvable_instance(base: dict[str, Any], index: int) -> TextTaskInstance:
    split = base["split"]
    source_id = base["source_id"]
    tid = f"text_solvable_{split}_{index:05d}"

    return TextTaskInstance(
        id=tid,
        substrate=TaskSubstrate.TEXT.value,
        state=TaskState.SOLVABLE.value,
        source_dataset=base["source_dataset"],
        source_id=source_id,
        split=split,
        question=base["question"],
        gold_answer=base["gold_answer"],
        gold_answer_numeric=base.get("gold_answer_numeric"),
        clarification_enabled=False,
        answer_format="free_form",
        expected_behavior="compute_and_answer",
        transform_ops=["gsm8k_direct"],
        metadata={"answer_full": base.get("answer_full")},
    )
