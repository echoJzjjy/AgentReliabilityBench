"""Normalize OSWorld task JSON into ARB GUI substrate records."""

from __future__ import annotations

from typing import Any

from arb.gui.sim_spec_enrich import normalize_related_apps


def _as_dict(value: Any) -> dict[str, Any]:
    """Normalize evaluator result/expected: dict, list[dict], or empty."""
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                return item
    return {}


def _as_dict_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        return [value]
    return []


def _func_name(func: Any) -> str:
    if isinstance(func, str):
        return func
    if isinstance(func, list) and func:
        first = func[0]
        return first if isinstance(first, str) else "generic"
    return "generic"


def _result_path(result: dict[str, Any]) -> str:
    return result.get("path") or result.get("dest") or ""


def _expected_rules(expected: dict[str, Any]) -> dict[str, Any]:
    if "rules" in expected and isinstance(expected["rules"], dict):
        return expected["rules"]
    return expected


def _extract_file_paths(config: list[dict[str, Any]]) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    seen: set[str] = set()
    for step in config or []:
        stype = step.get("type", "")
        params = step.get("parameters", {}) or {}
        if stype == "download":
            for f in params.get("files", []) or []:
                path = f.get("path", "")
                if path and path not in seen:
                    seen.add(path)
                    artifacts.append(
                        {
                            "path": path,
                            "kind": "file",
                            "url": f.get("url", ""),
                            "basename": path.rsplit("/", 1)[-1],
                        }
                    )
        elif stype == "open":
            path = params.get("path", "")
            if path and path not in seen:
                seen.add(path)
                artifacts.append({"path": path, "kind": "document", "basename": path.rsplit("/", 1)[-1]})
    return artifacts


def _extract_apps(config: list[dict[str, Any]], related_apps: list[str]) -> list[dict[str, Any]]:
    launched: set[str] = set()
    apps: list[dict[str, Any]] = []
    for step in config or []:
        if step.get("type") != "launch":
            continue
        cmd = step.get("parameters", {}).get("command", []) or []
        cmd_str = " ".join(cmd).lower()
        for app in related_apps or []:
            if app.lower() in cmd_str and app not in launched:
                launched.add(app)
                apps.append({"name": app, "open": True, "state": {}})
    for app in related_apps or []:
        if app not in launched:
            apps.append({"name": app, "open": False, "state": {}})
    return apps


def _infer_success_predicate(evaluator: dict[str, Any], related_apps: list[str]) -> dict[str, Any]:
    evaluator = evaluator or {}
    func = _func_name(evaluator.get("func", "generic"))
    result = _as_dict(evaluator.get("result"))
    expected = _as_dict(evaluator.get("expected"))
    rtype = result.get("type", "")
    app = related_apps[0] if related_apps else "desktop"

    if func == "match_in_list" and rtype == "default_search_engine":
        rules = _expected_rules(expected)
        return {
            "type": "setting_equals",
            "app": app,
            "key": "default_search_engine",
            "expected": rules.get("expected", []),
        }

    if func in ("compare_table", "compare_csv", "compare_text_file"):
        dest = _result_path(result)
        if not dest:
            for step in evaluator.get("postconfig", []) or []:
                if step.get("type") == "download":
                    for f in step.get("parameters", {}).get("files", []) or []:
                        dest = f.get("dest", f.get("path", ""))
                        break
        return {
            "type": "file_matches_gold",
            "path": dest,
            "gold_path": ".arb/gold/expected_output",
        }

    if func in ("check_include_exclude", "compare_docx_files", "compare_pptx_files"):
        rules = _expected_rules(expected)
        return {
            "type": "file_contains" if func == "check_include_exclude" else "file_matches_gold",
            "path": _result_path(result),
            "gold_path": ".arb/gold/expected_output",
            "include": rules.get("include", expected.get("include", [])),
            "exclude": rules.get("exclude", expected.get("exclude", [])),
        }

    if func == "is_file_exists":
        return {"type": "file_exists", "path": _result_path(result)}

    if rtype in ("vm_file", "cloud_file"):
        return {
            "type": "file_matches_gold",
            "path": _result_path(result),
            "gold_path": ".arb/gold/expected_output",
        }

    return {"type": "generic_complete", "app": app, "marker": "task_completed"}


def derive_sim_spec(task: dict[str, Any]) -> dict[str, Any]:
    config = task.get("config", []) or []
    related = normalize_related_apps(task.get("related_apps", []) or [])
    evaluator = task.get("evaluator", {}) or {}
    artifacts = _extract_file_paths(config)
    apps = _extract_apps(config, related)
    predicate = _infer_success_predicate(evaluator, related)
    return {
        "goal_type": predicate.get("type", "generic_complete"),
        "target_app": related[0] if related else "desktop",
        "artifacts": artifacts,
        "apps": apps,
        "success_predicate": predicate,
    }


def normalize_osworld_task(
    task: dict[str, Any],
    *,
    domain: str,
    index: int,
    split: str = "test",
) -> dict[str, Any]:
    task_id = task.get("id", f"unknown_{index}")
    related = normalize_related_apps(task.get("related_apps", []) or [])
    sim_spec = derive_sim_spec({**task, "related_apps": related})
    return {
        "source_dataset": "osworld",
        "source_id": task_id,
        "domain": domain,
        "split": split,
        "instruction": task.get("instruction", ""),
        "snapshot": task.get("snapshot", domain),
        "related_apps": related,
        "original_config": task.get("config", []),
        "original_evaluator": task.get("evaluator", {}),
        "sim_spec": sim_spec,
        "metadata": {
            "source_url": task.get("source", ""),
            "proxy": task.get("proxy", False),
            "evaluator_func": (task.get("evaluator") or {}).get("func", ""),
        },
    }


def task_has_transformable_resource(base: dict[str, Any]) -> bool:
    spec = base.get("sim_spec", {})
    return bool(spec.get("artifacts")) or bool(spec.get("apps"))
