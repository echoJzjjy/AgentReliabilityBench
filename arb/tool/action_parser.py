"""Parse agent turns into ASK_USER / TOOL / DONE / REPORT_FAILURE."""

from __future__ import annotations

import json
import re
from typing import Any

KNOWN_PREFIXES = ("ASK_USER:", "TOOL:", "DONE:", "REPORT_FAILURE:")

_THINKING_RE = re.compile(
    r"<(?:redacted_)?think(?:ing)?>.*?</(?:redacted_)?think(?:ing)?>",
    re.IGNORECASE | re.DOTALL,
)


def normalize_agent_output(text: str) -> str:
    """Strip chain-of-thought wrappers; keep the last protocol action line."""
    raw = text.strip()
    if not raw:
        return raw
    cleaned = _THINKING_RE.sub("", raw).strip()
    if not cleaned:
        cleaned = raw
    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
    for idx in range(len(lines) - 1, -1, -1):
        line = lines[idx]
        upper = line.upper()
        if not any(upper.startswith(p) for p in KNOWN_PREFIXES):
            continue
        if upper.startswith("TOOL:"):
            extra: list[str] = []
            for follow in lines[idx + 1 :]:
                if any(follow.upper().startswith(p) for p in KNOWN_PREFIXES):
                    break
                extra.append(follow)
            if extra:
                return line + "\n" + "\n".join(extra)
        return line
    for line in reversed(lines):
        if re.match(r"(?i)^tool\s*[:.]?\s*\w+", line):
            return line
    return cleaned


def parse_agent_turn(text: str) -> dict[str, Any]:
    """Return {type, question?, tool_name?, arguments?, message?}."""
    raw = normalize_agent_output(text)
    if not raw:
        return {"type": "empty"}

    upper = raw.upper()
    for prefix in KNOWN_PREFIXES:
        if upper.startswith(prefix):
            body = raw[len(prefix) :].strip()
            p = prefix.rstrip(":").lower()
            if p == "ask_user":
                return {"type": "ask_user", "question": body}
            if p == "tool":
                name, args = _parse_tool_body(body)
                return {"type": "tool", "tool_name": name, "arguments": args}
            if p == "done":
                return {"type": "done", "message": body}
            if p == "report_failure":
                return {"type": "report_failure", "message": body}

    j = _try_json_tool(raw)
    if j:
        return {"type": "tool", "tool_name": j["name"], "arguments": j.get("arguments", {})}

    m = re.search(r"<tool_call>\s*(\w+)\s*</tool_call>", raw, re.I)
    if m:
        args = _extract_json_after(raw, m.end()) or {}
        return {"type": "tool", "tool_name": m.group(1), "arguments": args}

    m = re.search(r'"name"\s*:\s*"([a-z_0-9]+)"\s*,\s*"arguments"\s*:\s*(\{.*?\})', raw, re.I | re.S)
    if m:
        try:
            args = json.loads(m.group(2))
        except json.JSONDecodeError:
            args = {}
        return {"type": "tool", "tool_name": m.group(1), "arguments": args}

    m = re.match(r"(?i)^(?:tool|function|action)\s*[:.]?\s*([a-z_][a-z0-9_]*)\s*$", raw.split("\n")[0])
    if m:
        rest = "\n".join(raw.split("\n")[1:]).strip()
        args = _parse_json_blob(rest) if rest else {}
        return {"type": "tool", "tool_name": m.group(1).lower(), "arguments": args}

    m = re.match(r"(?i)^tool\s*[:.]?\s*([a-z_][a-z0-9_]*)", raw)
    if m:
        name = re.sub(r"[{\"].*$", "", m.group(1)).strip().lower()
        blob = raw[m.end() :]
        args = _parse_json_blob(blob) or {}
        return {"type": "tool", "tool_name": name, "arguments": args}

    if _looks_like_ask(raw):
        return {"type": "ask_user", "question": raw}

    if _looks_like_failure_report(raw):
        return {"type": "report_failure", "message": raw}

    return {"type": "unknown", "raw": raw}


def _parse_tool_body(body: str) -> tuple[str, dict[str, Any]]:
    lines = body.split("\n")
    first = lines[0].strip()
    name = first.split()[0] if first else ""
    name = re.sub(r"[{\"].*$", "", name).strip().lower()
    rest = "\n".join(lines[1:]).strip() or first
    if rest == first and "{" in first:
        rest = first[first.find("{") :]
    args = _parse_json_blob(rest) if rest else {}
    if not name and isinstance(args, dict) and "name" in args:
        name = str(args.pop("name", "")).lower()
    return name, args if isinstance(args, dict) else {}


def _parse_json_blob(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return {}
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else {}
        except json.JSONDecodeError:
            pass
    return {}


def _try_json_tool(raw: str) -> dict[str, Any] | None:
    blob = _parse_json_blob(raw)
    if not blob:
        return None
    if "name" in blob:
        return {"name": str(blob["name"]), "arguments": blob.get("arguments") or blob.get("parameters") or {}}
    if "function" in blob and isinstance(blob["function"], dict):
        fn = blob["function"]
        return {"name": fn.get("name", ""), "arguments": fn.get("arguments") or {}}
    return None


def _extract_json_after(text: str, start: int) -> dict[str, Any] | None:
    sub = text[start:].strip()
    return _parse_json_blob(sub)


def _looks_like_ask(raw: str) -> bool:
    lower = raw.lower()
    if "?" not in raw:
        return False
    cues = ("could you", "please provide", "what is", "can you tell", "clarify", "which order", "zip code", "email")
    return any(c in lower for c in cues) and not lower.strip().startswith("tool")


def _looks_like_failure_report(raw: str) -> bool:
    lower = raw.lower()
    markers = (
        "cannot complete",
        "can't complete",
        "unable to complete",
        "not possible",
        "impossible to",
        "policy forbids",
        "permission denied",
        "not found in",
        "cannot be completed",
        "report that",
    )
    return any(m in lower for m in markers)
