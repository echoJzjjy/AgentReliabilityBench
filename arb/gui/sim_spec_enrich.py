"""Enrich OSWorld-derived sim_spec for solvable simulated-layer completion."""

from __future__ import annotations

import copy
import re
from typing import Any

from arb.gui.path_utils import workspace_rel_path

_APP_ALIASES: dict[str, str] = {
    "google-chrome": "chrome",
    "google_chrome": "chrome",
    "libreoffice calc": "libreoffice_calc",
    "libreoffice_calc": "libreoffice_calc",
    "libreoffice writer": "libreoffice_writer",
    "libreoffice_writer": "libreoffice_writer",
    "libreoffice impress": "libreoffice_impress",
    "libreoffice impress": "libreoffice_impress",
    "vs code": "vs_code",
    "vscode": "vs_code",
    "code": "vs_code",
    "calc": "libreoffice_calc",
    "image": "gimp",
    "pdf": "evince",
    "terminal": "terminal",
    "os": "desktop",
}


def normalize_app_name(name: str) -> str:
    key = (name or "desktop").strip()
    low = key.lower()
    if low in _APP_ALIASES:
        return _APP_ALIASES[low]
    key_sp = low.replace("_", " ")
    if key_sp in _APP_ALIASES:
        return _APP_ALIASES[key_sp]
    return re.sub(r"\s+", "_", key.strip())


def normalize_related_apps(apps: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for app in apps or []:
        norm = normalize_app_name(app)
        if norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out or ["desktop"]


def _gold_content(source_id: str, basename: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9_]+", "_", source_id or basename)
    return f"SIM_GOLD_OUTPUT:{token}\n# target={basename}\n"


def enrich_sim_spec(sim_spec: dict[str, Any], *, source_id: str, related_apps: list[str]) -> dict[str, Any]:
    """Attach sim-layer completion metadata used by workspace, prompts, and baselines."""
    spec = copy.deepcopy(sim_spec)
    apps = normalize_related_apps(related_apps)
    pred = spec.setdefault("success_predicate", {})
    ptype = pred.get("type", "generic_complete")

    if ptype == "generic_complete":
        pred["app"] = normalize_app_name(pred.get("app") or apps[0])

    if ptype == "setting_equals":
        pred["app"] = normalize_app_name(pred.get("app") or apps[0])

    if ptype in ("file_matches_gold", "file_contains", "file_exists"):
        rel = workspace_rel_path(pred.get("path", ""))
        if not rel and ptype == "file_contains":
            rel = f"sim_output/{source_id}.txt"
        pred["workspace_rel_path"] = rel
        pred["basename"] = rel.rsplit("/", 1)[-1] if rel else "output.txt"

    if ptype == "file_matches_gold":
        pred["gold_content"] = _gold_content(source_id, pred.get("basename", "output.txt"))
        pred["gold_read_path"] = ".arb/gold/expected_output"

    if ptype == "file_contains":
        includes = pred.get("include") or []
        if isinstance(includes, str):
            includes = [includes]
        if not includes:
            includes = [f"REQUIRED_{source_id}"]
        pred["required_snippets"] = list(includes)
        pred["starter_content"] = "Initial workspace document.\n"

    # Normalize app entries in sim_spec.apps
    normalized_apps: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for app in spec.get("apps", []):
        name = normalize_app_name(app.get("name", ""))
        if name in seen_names:
            continue
        seen_names.add(name)
        entry = copy.deepcopy(app)
        entry["name"] = name
        normalized_apps.append(entry)
    for name in apps:
        if name not in seen_names:
            normalized_apps.append({"name": name, "open": False, "state": {}})
            seen_names.add(name)
    spec["apps"] = normalized_apps

    spec["sim_layer"] = {
        "version": 4,
        "completion_mode": ptype,
        "primary_app": apps[0],
        "stratum": "file_workflow" if ptype in ("file_matches_gold", "file_contains", "file_exists") else "app_control",
    }
    return spec


def enrich_task_dict(task: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(task)
    out["related_apps"] = normalize_related_apps(out.get("related_apps", []))
    out["sim_spec"] = enrich_sim_spec(
        out.get("sim_spec", {}),
        source_id=out.get("source_id", out.get("id", "task")),
        related_apps=out["related_apps"],
    )
    return out
