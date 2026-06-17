"""GUI + Solvable: OSWorld task with unified simulation spec."""

from __future__ import annotations

import copy
from typing import Any

from arb.schemas import GuiTaskInstance, TaskState, TaskSubstrate


def _default_filesystem_policy() -> dict[str, Any]:
    return {
        "read_only": [".arb/evaluator/", ".arb/gold/"],
        "writable": ["workspace/"],
        "recovery_enabled": False,
        "evaluator_reads_agent_log": False,
    }


def build_solvable_instance(base: dict[str, Any], index: int) -> GuiTaskInstance:
    split = base["split"]
    source_id = base["source_id"]
    return GuiTaskInstance(
        id=f"gui_solvable_{split}_{index:05d}",
        substrate=TaskSubstrate.GUI.value,
        state=TaskState.SOLVABLE.value,
        source_dataset=base["source_dataset"],
        source_id=source_id,
        domain=base["domain"],
        split=split,
        instruction=base["instruction"],
        snapshot=base["snapshot"],
        related_apps=base["related_apps"],
        original_config=copy.deepcopy(base.get("original_config", [])),
        original_evaluator=copy.deepcopy(base.get("original_evaluator", {})),
        sim_spec=copy.deepcopy(base.get("sim_spec", {})),
        clarification_enabled=False,
        recovery_channels=[],
        expected_behavior="complete_gui_task",
        filesystem_policy=_default_filesystem_policy(),
        transform_ops=["osworld_direct"],
        metadata=copy.deepcopy(base.get("metadata", {})),
    )
