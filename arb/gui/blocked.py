"""GUI + Blocked-but-Recoverable: hide/lock required resource + recovery channels."""

from __future__ import annotations

import copy
from typing import Any

from arb.gui.resource_parser import select_critical_resource
from arb.gui.solvable import _default_filesystem_policy
from arb.schemas import BlockedResource, GuiTaskInstance, TaskState, TaskSubstrate


def build_blocked_instance(base: dict[str, Any], index: int) -> GuiTaskInstance | None:
    resource = select_critical_resource(base, index)
    if resource is None:
        return None

    sim_spec = copy.deepcopy(base.get("sim_spec", {}))
    blocked_label = f"[UNAVAILABLE: {resource.label}]"

    if resource.resource_type == "file":
        for art in sim_spec.get("artifacts", []):
            if art.get("path") == resource.path:
                art["locked"] = True
                art["visible"] = False
                art["blocked_label"] = blocked_label
    elif resource.resource_type == "window":
        for app in sim_spec.get("apps", []):
            if app.get("name") == resource.path:
                app["open"] = False
                app["blocked"] = True
                app["blocked_label"] = blocked_label
    else:
        sim_spec.setdefault("permissions", {})[resource.path] = "denied"

    br = BlockedResource(
        resource_id=resource.resource_id,
        resource_type=resource.resource_type,
        path=resource.path,
        blocked_state=blocked_label,
        original_state="available",
        recovery_channel=resource.recovery_channel,
        clarification_keywords=resource.clarification_keywords,
    )

    policy = _default_filesystem_policy()
    policy["recovery_enabled"] = True

    split = base["split"]
    instruction = (
        f"{base['instruction']}\n\n"
        f"Note: a required resource may be temporarily unavailable ({blocked_label}). "
        "Use recovery channels if needed."
    )

    return GuiTaskInstance(
        id=f"gui_blocked_{split}_{index:05d}",
        substrate=TaskSubstrate.GUI.value,
        state=TaskState.BLOCKED_BUT_RECOVERABLE.value,
        source_dataset=base["source_dataset"],
        source_id=base["source_id"],
        domain=base["domain"],
        split=split,
        instruction=instruction,
        snapshot=base["snapshot"],
        related_apps=base["related_apps"],
        original_config=copy.deepcopy(base.get("original_config", [])),
        original_evaluator=copy.deepcopy(base.get("original_evaluator", {})),
        sim_spec=sim_spec,
        blocked_resources=[br.to_dict()],
        clarification_enabled=True,
        recovery_channels=["request_access", "ask_user", "open_attachment"],
        expected_behavior="recover_resource_then_complete",
        filesystem_policy=policy,
        transform_ops=["hide_or_lock_resource", "enable_recovery_channels"],
        metadata={"original_instruction": base["instruction"]},
    )
