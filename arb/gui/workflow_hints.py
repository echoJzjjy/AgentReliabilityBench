"""Deterministic next-step hints for multi-turn GUI file workflows."""

from __future__ import annotations

from typing import Any


def sim_stratum(task: dict[str, Any]) -> str:
    ptype = task.get("sim_spec", {}).get("success_predicate", {}).get("type", "")
    if ptype in ("file_matches_gold", "file_contains", "file_exists"):
        return "file_workflow"
    return "app_control"


def file_target_rel(task: dict[str, Any]) -> str | None:
    pred = task.get("sim_spec", {}).get("success_predicate", {})
    return pred.get("workspace_rel_path") or None


def after_open_hint(task: dict[str, Any]) -> str | None:
    pred = task.get("sim_spec", {}).get("success_predicate", {})
    ptype = pred.get("type", "")
    if ptype == "file_matches_gold":
        return "Next action (one line only): READ .arb/gold/expected_output"
    if ptype == "file_contains":
        rel = file_target_rel(task)
        snippets = pred.get("required_snippets") or pred.get("include") or []
        if isinstance(snippets, str):
            snippets = [snippets]
        if rel and snippets:
            return (
                f"Next action (one line only): SAVE workspace/{rel}\n"
                + "\n".join(str(s) for s in snippets)
                + "\n"
            )
    if ptype == "file_exists":
        rel = file_target_rel(task)
        if rel:
            return f"Next action (one line only): SAVE workspace/{rel}\ncreated\n"
    return None


def after_read_gold_hint(task: dict[str, Any], gold_content: str) -> str:
    rel = file_target_rel(task) or "output.txt"
    body = gold_content if gold_content.endswith("\n") else gold_content + "\n"
    return (
        f"Gold file loaded. Next action MUST be a SAVE (two lines):\n"
        f"SAVE workspace/{rel}\n"
        f"{body}"
        f"(copy the gold content exactly, then SUBMIT on the following turn)"
    )


def extract_gold_from_observation(observation: str) -> str | None:
    marker = "Contents of .arb/gold/expected_output:"
    if marker not in observation:
        return None
    return observation.split(marker, 1)[1].lstrip("\n")
