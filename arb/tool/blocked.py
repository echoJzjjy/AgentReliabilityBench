"""Tool + Blocked-but-Recoverable: mask critical slot in user request."""

from __future__ import annotations

from typing import Any

from arb.schemas import MaskedToolSlot, TaskState, TaskSubstrate, ToolTaskInstance
from arb.tool.solvable import build_solvable_instance
from arb.tool.tool_catalog import tools_for_domain
from arb.utils.taubench import (
    actions_require_slot,
    extract_maskable_slots,
    load_wiki,
    normalize_gold_actions,
    pick_mask_slot,
)


def build_blocked_instance(base: dict[str, Any], index: int) -> ToolTaskInstance | None:
    instruction = base["user_instruction"]
    all_slots = extract_maskable_slots(instruction, index)
    if not all_slots:
        return None

    slot = None
    for off in range(len(all_slots)):
        candidate = pick_mask_slot(all_slots, index + off)[0]
        if actions_require_slot(base["gold_actions"], candidate):
            slot = candidate
            break
    if slot is None:
        return None
    if not actions_require_slot(base["gold_actions"], slot):
        return None

    masked_instruction = instruction.replace(slot["original_text"], slot["masked_token"], 1)
    ms = MaskedToolSlot(
        slot_id=slot["slot_id"],
        slot_type=slot["slot_type"],
        original_text=slot["original_text"],
        masked_token=slot["masked_token"],
        clarification_keywords=slot["clarification_keywords"],
    )

    domain = base["domain"]
    split = base["split"]
    policy = load_wiki(domain)

    return ToolTaskInstance(
        id=f"tool_blocked_{domain}_{split}_{index:05d}",
        substrate=TaskSubstrate.TOOL.value,
        state=TaskState.BLOCKED_BUT_RECOVERABLE.value,
        source_dataset=base["source_dataset"],
        source_id=base["source_id"],
        split=split,
        domain=domain,
        user_instruction=masked_instruction,
        domain_policy=policy,
        available_tools=tools_for_domain(domain),
        gold_actions=normalize_gold_actions(
            domain, base["gold_actions"], base.get("user_id", "")
        ),
        expected_outputs=base.get("expected_outputs", []),
        user_id=base.get("user_id", ""),
        masked_slots=[ms.to_dict()],
        clarification_enabled=True,
        expected_behavior="ask_user_then_complete",
        tool_policy={"exploit_tools_enabled": False, "clarification_enabled": True},
        transform_ops=["mask_user_slot", "enable_ask_user"],
        metadata={"original_user_instruction": instruction},
    )
