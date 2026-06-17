"""Path normalization for simulated GUI workspace."""

from __future__ import annotations


def workspace_rel_path(path: str) -> str:
    """Map OSWorld vm paths to workspace-relative paths."""
    rel = (path or "").replace("home/user/", "").lstrip("/")
    return rel


def save_target_path(task: dict) -> str | None:
    """Relative path (under workspace/) for SAVE on file predicates."""
    pred = task.get("sim_spec", {}).get("success_predicate", {})
    rel = pred.get("workspace_rel_path") or workspace_rel_path(pred.get("path", ""))
    return rel or None
