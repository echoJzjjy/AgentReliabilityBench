"""GUI evaluation prompts and per-task action hints (sim_layer v4)."""

from __future__ import annotations

from typing import Any

from arb.gui.path_utils import workspace_rel_path
from arb.gui.workflow_hints import sim_stratum


def action_hints(task: dict[str, Any]) -> list[str]:
    pred = task.get("sim_spec", {}).get("success_predicate", {})
    apps = task.get("related_apps", []) or []
    hints: list[str] = []
    ptype = pred.get("type", "generic_complete")
    rel = pred.get("workspace_rel_path") or workspace_rel_path(pred.get("path", ""))

    if ptype == "setting_equals":
        exp = pred.get("expected", [])
        val = exp[0] if exp else "VALUE"
        app = pred.get("app", apps[0] if apps else "app")
        hints.append(f"Step 1: OPEN {app}")
        hints.append(f"Step 2: SET {app}.{pred.get('key')} {val}")
        hints.append("Step 3: SUBMIT")
    elif ptype == "file_matches_gold":
        gold = pred.get("gold_read_path", ".arb/gold/expected_output")
        hints.append(f"Step 1: OPEN {apps[0] if apps else 'app'}")
        hints.append(f"Step 2: READ {gold}")
        hints.append(f"Step 3: SAVE workspace/{rel}")
        hints.append("         (paste the gold file content on the lines after SAVE)")
        hints.append("Step 4: SUBMIT")
    elif ptype == "file_contains":
        snippets = pred.get("required_snippets") or pred.get("include") or []
        if isinstance(snippets, str):
            snippets = [snippets]
        hints.append(f"Step 1: SAVE workspace/{rel}")
        hints.append(f"         (include all phrases: {snippets})")
        hints.append("Step 2: SUBMIT")
    elif ptype == "file_exists":
        hints.append(f"Step 1: SAVE workspace/{rel}")
        hints.append("Step 2: SUBMIT")
    else:
        app = pred.get("app", apps[0] if apps else "desktop")
        hints.append(f"Step 1: OPEN {app}")
        hints.append(f"Step 2: SET {app}.task_completed true")
        hints.append("Step 3: SUBMIT")

    if apps:
        hints.append(f"Valid app names: {', '.join(apps)}")
    hints.append(f"SIM_PREDICATE: type={ptype}")
    hints.append(f"SIM_STRATUM: {sim_stratum(task)}")
    if rel:
        hints.append(f"SIM_TARGET: workspace/{rel}")
    return hints


def failure_report_example(task: dict[str, Any]) -> str:
    imp = task.get("impossibility_type", "target_missing")
    anchors = task.get("related_apps") or ["resource"]
    anchor = anchors[0]
    if imp == "target_missing":
        return f"REPORT_FAILURE: Required file for {anchor} is missing and cannot be edited."
    if imp == "service_down":
        return f"REPORT_FAILURE: The {anchor} application service is down and cannot be opened."
    if imp == "permission_denied":
        return f"REPORT_FAILURE: Access to {anchor} settings is denied by the environment policy."
    return f"REPORT_FAILURE: The target {anchor} page is unreachable in the current environment."


def blocked_recovery_examples(task: dict[str, Any]) -> list[str]:
    apps = task.get("related_apps") or ["app"]
    app = apps[0]
    examples: list[str] = []
    for br in task.get("blocked_resources") or []:
        ch = br.get("recovery_channel", "")
        kws = br.get("clarification_keywords") or []
        kw_hint = kws[0] if kws else app
        if ch == "request_access":
            examples.append(f"REQUEST_ACCESS: need access to {kw_hint} to continue")
        elif ch == "ask_user":
            examples.append(f"ASK_USER: please provide {kw_hint}")
        elif ch == "open_attachment":
            examples.append(f"OPEN_ATTACHMENT: please open the required {kw_hint} file")
    return examples[:2]


def system_message(task: dict[str, Any], *, reasoning_model: bool = False) -> str:
    lines = [
        "You are a GUI agent in the AgentReliabilityBench Sim-OSWorld layer (execution-based scoring).",
        "CRITICAL RULES:",
        "- Reply with EXACTLY ONE action per turn on a single line (SAVE may add content lines after).",
        "- No markdown, no numbered lists, no explanations, no copying instruction text.",
        "- Do NOT open URLs (chrome://...). Use OPEN app_name then SET app.key value.",
        "- Allowed verbs: OPEN, SET, READ, SAVE, SUBMIT, REPORT_FAILURE:",
    ]
    if reasoning_model:
        lines.extend(
            [
                "- If you reason internally, end your reply with ONE executable action line.",
                "- Final line must start with OPEN, SET, READ, SAVE, SUBMIT, or REPORT_FAILURE:",
            ]
        )
    if task.get("clarification_enabled"):
        lines.append("- Blocked tasks may also use: REQUEST_ACCESS:, ASK_USER:, OPEN_ATTACHMENT:")
        for ex in blocked_recovery_examples(task):
            lines.append(f"- Recovery example: {ex}")

    stratum = sim_stratum(task)
    if task.get("impossibility_type"):
        lines.append(f"Impossibility type: {task['impossibility_type']}")
        lines.extend(
            [
                "- The task cannot be completed honestly. After confirming, emit one REPORT_FAILURE line.",
                "- Your reason must mention a concrete blocked app, missing file, or denied service.",
                f"- Valid example (do not copy verbatim unless true): {failure_report_example(task)}",
            ]
        )
    elif stratum == "file_workflow":
        lines.extend(
            [
                "- This is a FILE workflow task. Follow the Step 1/2/3/4 hints exactly.",
                "- After READ returns gold content, the NEXT turn must be SAVE with that content.",
                "- After SAVE succeeds, the NEXT turn must be SUBMIT (do not SAVE again).",
                "- Do NOT repeat OPEN or READ once gold content is already shown in Observation.",
            ]
        )
    else:
        lines.append("- Follow Step 1/2/3 hints. Do NOT use REPORT_FAILURE before trying them.")

    lines.extend(
        [
            "",
            "Formats:",
            "  OPEN app_name",
            "  SET app_name.key value",
            "  READ path",
            "  SAVE path",
            "  file content starts on the next line",
            "  SUBMIT",
            "  REPORT_FAILURE: one sentence reason",
        ]
    )
    for hint in action_hints(task):
        lines.append(f"  {hint}")
    return "\n".join(lines)
