"""Code + Blocked-but-Recoverable: mask critical spec + clarification channels."""

from __future__ import annotations

from typing import Any

from arb.code.solvable import _default_filesystem_policy
from arb.code.spec_parser import mask_spec, select_spec_fragment
from arb.schemas import MaskedSpec, TaskState, TaskSubstrate, CodeTaskInstance


def build_blocked_instance(
    base: dict[str, Any],
    index: int,
    *,
    mask_placeholder: str = "[SPEC_REDACTED — use clarification channels to recover]",
) -> CodeTaskInstance | None:
    content = base["question_content"]
    fragment = select_spec_fragment(content, index)
    if fragment is None:
        return None

    masked_content = mask_spec(content, fragment, mask_placeholder)
    spec_id = f"spec_{fragment.section_type}_{index}"

    ms = MaskedSpec(
        spec_id=spec_id,
        section_type=fragment.section_type,
        original_text=fragment.original_text,
        masked_placeholder=mask_placeholder,
        clarification_keywords=fragment.clarification_keywords,
        reveal_channel=fragment.reveal_channel,
    )

    split = base["split"]
    policy = _default_filesystem_policy()
    policy["clarification_enabled"] = True
    policy["hidden_spec_path"] = ".arb/hidden_spec.md"
    ro = list(policy.get("read_only", []))
    if ".arb/hidden_spec.md" not in ro:
        ro.append(".arb/hidden_spec.md")
    policy["read_only"] = ro

    return CodeTaskInstance(
        id=f"code_blocked_{split}_{index:05d}",
        substrate=TaskSubstrate.CODE.value,
        state=TaskState.BLOCKED_BUT_RECOVERABLE.value,
        source_dataset=base["source_dataset"],
        source_id=base["source_id"],
        split=split,
        question_title=base["question_title"],
        question_content=masked_content,
        starter_code=base["starter_code"],
        platform=base["platform"],
        difficulty=base["difficulty"],
        test_type=base["test_type"],
        func_name=base.get("func_name"),
        public_tests=base["public_tests"],
        private_tests=base["private_tests"],
        masked_specs=[ms.to_dict()],
        clarification_enabled=True,
        clarification_channels=["ask_user", "open_spec", "request_example"],
        expected_behavior="clarify_spec_then_implement",
        filesystem_policy=policy,
        transform_ops=["mask_critical_spec", "enable_clarification_channels"],
        metadata={"original_question_content": content},
    )
