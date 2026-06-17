"""Normalize free-form LLM output into one GUI environment action per turn (sim_layer v4)."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any

from arb.gui.workflow_hints import extract_gold_from_observation, file_target_rel

_ACTION_PREFIXES = (
    "REQUEST_ACCESS:",
    "ASK_USER:",
    "OPEN_ATTACHMENT:",
    "REPORT_FAILURE:",
    "READ ",
    "OPEN ",
    "SET ",
    "SAVE ",
    "CLICK ",
    "TYPE ",
)

_APP_ALIASES: dict[str, str] = {
    "google-chrome": "chrome",
    "google_chrome": "chrome",
    "chrome_browser": "chrome",
    "libreoffice_calc": "libreoffice_calc",
    "libreoffice calc": "libreoffice_calc",
    "libreoffice_writer": "libreoffice_writer",
    "libreoffice writer": "libreoffice_writer",
    "libreoffice_impress": "libreoffice_impress",
    "libreoffice impress": "libreoffice_impress",
    "vs_code": "vs_code",
    "vscode": "vs_code",
    "code": "vs_code",
    "gimp": "gimp",
    "vlc": "vlc",
    "thunderbird": "thunderbird",
}


@dataclass(frozen=True)
class ParseResult:
    action: str
    assist_kind: str  # none | structured | heuristic | workflow_recovery
    assist_reason: str | None = None


def _clean_line(line: str) -> str:
    line = line.strip()
    line = re.sub(r"^#+\s*", "", line)
    line = re.sub(r"^(?:step\s*)?\d+[\.\):]\s*", "", line, flags=re.I)
    line = re.sub(r"^[-*]\s*", "", line)
    return line.strip()


def _resolve_app(name: str, related_apps: list[str]) -> str:
    raw = name.strip().strip("\"'")
    key = raw.lower().replace("-", "_")
    key_sp = raw.lower().replace("_", " ")
    if key in _APP_ALIASES:
        return _APP_ALIASES[key]
    if key_sp in _APP_ALIASES:
        return _APP_ALIASES[key_sp]
    for app in related_apps:
        if key == app.lower() or key.replace("_", "") == app.lower().replace("_", ""):
            return app
    return raw


def _is_url_like_app(name: str) -> bool:
    low = name.lower()
    return "://" in low or low.startswith("chrome://") or low.startswith("about:")


def _observation_requires_save(observation: str) -> bool:
    low = observation.lower()
    if "next action must be a save" in low:
        return True
    return bool(re.search(r"next action \(one line only\):\s*save\b", low))


def _observation_requires_submit(observation: str) -> bool:
    return bool(re.search(r"next action:\s*submit\b", observation.lower()))


def _find_structured_line(lines: list[str]) -> str | None:
    for line in lines:
        clean = _clean_line(line)
        if not clean:
            continue
        upper = clean.upper()
        if upper in ("SUBMIT", "DONE"):
            return "SUBMIT"
        for prefix in _ACTION_PREFIXES:
            if upper.startswith(prefix.upper()):
                if prefix == "SET " and not re.match(r"SET\s+\w+\.\w+\s+", clean, re.I):
                    continue
                return clean
        m = re.search(r"(REPORT_FAILURE:\s*.+)$", clean, re.I)
        if m:
            return m.group(1).strip()
        m = re.search(r"\b(OPEN\s+\S+)$", clean, re.I)
        if m and not _is_url_like_app(m.group(1)[5:].strip()):
            return m.group(1).strip()
        m = re.search(r"\b(SET\s+\w+\.\w+\s+.+)$", clean, re.I)
        if m:
            return m.group(1).strip()
    return None


def _workflow_recovery(
    text: str,
    task: dict[str, Any],
    last_observation: str | None,
) -> str | None:
    """Recover file workflow only when env explicitly asks for SAVE and model repeats OPEN/READ."""
    if not last_observation:
        return None

    lower = text.lower().strip()
    if lower in ("submit", "done") or _observation_requires_submit(last_observation):
        return None

    pred = task.get("sim_spec", {}).get("success_predicate", {})
    ptype = pred.get("type", "")
    gold = extract_gold_from_observation(last_observation)
    rel = file_target_rel(task)

    if ptype == "file_matches_gold" and gold and rel:
        if not _observation_requires_save(last_observation):
            return None
        if any(k in lower for k in ("open", "read", "launch", "start")):
            body = gold if gold.endswith("\n") else gold + "\n"
            return f"SAVE workspace/{rel}\n{body}"

    if ptype == "file_contains" and rel and _observation_requires_save(last_observation):
        snippets = pred.get("required_snippets") or pred.get("include") or []
        if isinstance(snippets, str):
            snippets = [snippets]
        m = re.search(r"SAVE workspace/([^\n]+)\n(.*)", last_observation, re.S | re.I)
        if m:
            path, body = m.group(1).strip(), m.group(2).strip()
            if body and not body.startswith("("):
                return f"SAVE workspace/{path}\n{body.split('(copy')[0].strip()}\n"
        if snippets and any(k in lower for k in ("open", "read", "launch", "start")):
            body = "\n".join(str(s) for s in snippets) + "\n"
            return f"SAVE workspace/{rel}\n{body}"

    return None


def _heuristic_action(text: str, task: dict[str, Any]) -> str | None:
    lower = text.lower()
    pred = task.get("sim_spec", {}).get("success_predicate", {})
    apps = task.get("related_apps", []) or []
    primary = apps[0] if apps else "desktop"
    ptype = pred.get("type", "generic_complete")

    if re.search(r"\b(submit|done|finish)\b", lower) and "report_failure" not in lower:
        return "SUBMIT"

    if ptype == "setting_equals":
        app = pred.get("app", primary)
        key = pred.get("key", "setting")
        expected = pred.get("expected", [])
        if any(k in lower for k in ("search engine", "bing", "default search", key.replace("_", " "))):
            val = expected[0] if expected else "Microsoft Bing"
            return f"SET {app}.{key} {val}"
        if re.search(r"\bopen\b", lower) and not _is_url_like_app(text):
            return f"OPEN {app}"

    for app in apps:
        app_l = app.lower()
        app_sp = app_l.replace("_", " ")
        if re.search(rf"\b(open|launch|start)\s+({re.escape(app_l)}|{re.escape(app_sp)})\b", lower):
            return f"OPEN {app}"

    if ptype == "generic_complete":
        app = pred.get("app", primary)
        if "task_completed" in lower or ("set" in lower and "true" in lower):
            return f"SET {app}.task_completed true"
        if "open" in lower and apps and not _is_url_like_app(text):
            return f"OPEN {app}"

    if ptype == "file_matches_gold":
        rel = file_target_rel(task) or "output.txt"
        if "read" in lower and "gold" in lower:
            return "READ .arb/gold/expected_output"
        if "save" in lower or "write" in lower:
            return f"SAVE workspace/{rel}\nSIM_GOLD_OUTPUT\n"

    if ptype == "file_contains":
        rel = file_target_rel(task) or "output.txt"
        snippets = pred.get("required_snippets") or pred.get("include") or []
        if isinstance(snippets, str):
            snippets = [snippets]
        body = "\n".join(str(s) for s in snippets) + "\n"
        if "save" in lower or "write" in lower:
            return f"SAVE workspace/{rel}\n{body}"

    if ptype == "file_exists":
        rel = file_target_rel(task) or "output.txt"
        if "save" in lower:
            return f"SAVE workspace/{rel}\ncreated\n"

    return None


def normalize_agent_output_with_meta(
    raw: str,
    task: dict[str, Any],
    *,
    last_observation: str | None = None,
) -> ParseResult:
    """Return a single executable action string plus parser assist metadata."""
    text = raw.strip()
    if not text:
        return ParseResult(action=text, assist_kind="none")

    state = task.get("state", "")
    lines = [ln for ln in text.splitlines() if ln.strip()]

    if state.startswith("impossible"):
        m = re.search(r"REPORT_FAILURE:\s*(.+)", text, re.I | re.S)
        if m:
            reason = m.group(1).strip().split("\n")[0].strip()
            reason = re.sub(r'["\']+$', "", reason)
            return ParseResult(
                action=f"REPORT_FAILURE: {reason}",
                assist_kind="structured",
                assist_reason="report_failure_extract",
            )

    recovered = _workflow_recovery(text, task, last_observation)
    if recovered:
        return ParseResult(
            action=recovered,
            assist_kind="workflow_recovery",
            assist_reason="env_save_hint",
        )

    structured = _find_structured_line(lines)
    if structured:
        upper = structured.upper()
        if upper.startswith("OPEN "):
            app_token = structured[5:].strip()
            if _is_url_like_app(app_token):
                heuristic = _heuristic_action(text, task)
                if heuristic:
                    return ParseResult(
                        action=heuristic,
                        assist_kind="heuristic",
                        assist_reason="reject_url_open",
                    )
            app = _resolve_app(app_token, task.get("related_apps", []))
            return ParseResult(action=f"OPEN {app}", assist_kind="structured")
        if upper.startswith("SAVE "):
            return ParseResult(action=structured, assist_kind="structured")
        return ParseResult(action=structured, assist_kind="structured")

    heuristic = _heuristic_action(text, task)
    if heuristic:
        return ParseResult(action=heuristic, assist_kind="heuristic")

    first = _clean_line(lines[0]) if lines else text
    return ParseResult(action=first, assist_kind="none")


def normalize_agent_output(
    raw: str,
    task: dict[str, Any],
    *,
    last_observation: str | None = None,
) -> str:
    return normalize_agent_output_with_meta(
        raw, task, last_observation=last_observation
    ).action
