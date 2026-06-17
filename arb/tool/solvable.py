"""Tool + Solvable: original τ-bench task."""

from __future__ import annotations

from typing import Any

from arb.schemas import TaskState, TaskSubstrate, ToolTaskInstance
from arb.tool.backend import is_scorable_task
from arb.tool.tool_catalog import tools_for_domain
from arb.utils.taubench import load_wiki, normalize_gold_actions


def build_solvable_instance(
    base: dict[str, Any],
    index: int,
    *,
    wiki_text: str | None = None,
) -> ToolTaskInstance | None:
    if not is_scorable_task(base):
        return None
    domain = base["domain"]
    split = base["split"]
    policy = wiki_text or base.get("domain_policy") or load_wiki(domain)

    return ToolTaskInstance(
        id=f"tool_solvable_{domain}_{split}_{index:05d}",
        substrate=TaskSubstrate.TOOL.value,
        state=TaskState.SOLVABLE.value,
        source_dataset=base["source_dataset"],
        source_id=base["source_id"],
        split=split,
        domain=domain,
        user_instruction=base["user_instruction"],
        domain_policy=policy,
        available_tools=tools_for_domain(domain),
        gold_actions=normalize_gold_actions(
            domain, base["gold_actions"], base.get("user_id", "")
        ),
        expected_outputs=base.get("expected_outputs", []),
        user_id=base.get("user_id", ""),
        expected_behavior="complete_via_tools",
        tool_policy={"exploit_tools_enabled": False, "clarification_enabled": False},
        transform_ops=["passthrough"],
        metadata={"annotator": base.get("annotator")},
    )
