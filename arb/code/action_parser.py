"""Normalize free-form LLM replies into structured code-environment actions."""

from __future__ import annotations

import re

_STRUCTURED_PREFIXES = (
    "WRITE ",
    "READ ",
    "RUN_TESTS",
    "SUBMIT",
    "REPORT_FAILURE:",
    "CLARIFY:",
    "OPEN_SPEC:",
    "REQUEST_EXAMPLE:",
)

_MAX_READ_PATH_LEN = 512


def _extract_code_block(text: str) -> str | None:
    for m in re.finditer(r"```(?:python|py|Python)?\s*\n([\s\S]+?)```", text):
        code = m.group(1).strip()
        if code and ("def " in code or "class " in code or "import " in code):
            return code
    return None


def _extract_inline_write(text: str) -> str | None:
    m = re.search(r"WRITE\s+solution\.py\s*\n([\s\S]+)", text, re.IGNORECASE)
    if m:
        code = m.group(1).strip()
        if code.startswith("```"):
            code = re.sub(r"^```(?:python|py)?\n?", "", code)
            code = re.sub(r"\n?```$", "", code)
        return code.strip()
    return None


def _sanitize_read_path(raw: str) -> str:
    """Keep a single relative path token; models often append WRITE/code on later lines."""
    path = raw.replace("\\", "/").strip()
    path = path.splitlines()[0].strip()
    if path.startswith('"') and path.endswith('"'):
        path = path[1:-1]
    if path.startswith("'") and path.endswith("'"):
        path = path[1:-1]
    path = path.split()[0] if path.split() else ""
    if len(path) > _MAX_READ_PATH_LEN:
        path = path[:_MAX_READ_PATH_LEN]
    return path


def _parse_read_action(text: str) -> str:
    m = re.search(r"(?im)^READ\s+(\S+)", text)
    if not m:
        return text
    path = _sanitize_read_path(m.group(1))
    return f"READ {path}" if path else text


def _parse_write_action(text: str) -> str | None:
    m = re.search(r"(?im)^WRITE\s+(\S+)\s*\n([\s\S]+)", text)
    if not m:
        return None
    rel = m.group(1).strip()
    code = m.group(2).strip()
    if code.startswith("```"):
        code = re.sub(r"^```(?:python|py)?\n?", "", code)
        code = re.sub(r"\n?```$", "", code)
    return f"WRITE {rel}\n{code}"


def _normalize_structured_action(text: str) -> str:
    """Collapse multi-action replies; never pass READ + trailing code as the path."""
    upper = text.upper()

    if upper.startswith("REPORT_FAILURE:"):
        body = text.split(":", 1)[1].strip().splitlines()[0]
        return f"REPORT_FAILURE: {body}"

    for prefix in ("CLARIFY:", "OPEN_SPEC:", "REQUEST_EXAMPLE:"):
        if upper.startswith(prefix):
            body = text.split(":", 1)[1].strip().splitlines()[0]
            return f"{prefix} {body}"

    if re.fullmatch(r"(?i)SUBMIT\.?", text.strip()) or upper.strip() == "SUBMIT":
        return "SUBMIT"

    if re.fullmatch(r"(?i)RUN_TESTS\.?", text.strip()) or upper.strip() == "RUN_TESTS":
        return "RUN_TESTS"

    # Prefer WRITE when model glued READ + WRITE in one turn (common failure mode).
    if re.search(r"(?im)^WRITE\s+", text):
        parsed = _parse_write_action(text)
        if parsed:
            return parsed
        inline = _extract_inline_write(text)
        if inline:
            return inline

    if upper.startswith("READ "):
        return _parse_read_action(text)

    if upper.startswith("WRITE "):
        parsed = _parse_write_action(text)
        if parsed:
            return parsed

    return text


def normalize_agent_action(raw: str) -> str:
    """Best-effort mapping from natural language / markdown to one environment action."""
    text = raw.strip()
    if not text:
        return text

    if any(text.upper().startswith(p) for p in _STRUCTURED_PREFIXES):
        return _normalize_structured_action(text)

    for prefix in ("REPORT_FAILURE:", "CLARIFY:", "OPEN_SPEC:", "REQUEST_EXAMPLE:"):
        m = re.search(rf"(?i){re.escape(prefix)}\s*([^\n]+)", text)
        if m:
            return f"{prefix} {m.group(1).strip()}"

    if re.fullmatch(r"(?i)SUBMIT\.?", text) or re.search(
        r"(?im)^(?:ACTION:\s*)?SUBMIT\s*$", text
    ):
        return "SUBMIT"

    if re.fullmatch(r"(?i)RUN_TESTS\.?", text) or re.search(
        r"(?im)^(?:ACTION:\s*)?RUN_TESTS\s*$", text
    ):
        return "RUN_TESTS"

    m = re.search(r"(?im)^READ\s+(\S+)", text)
    if m:
        path = _sanitize_read_path(m.group(1))
        if path:
            return f"READ {path}"

    inline = _extract_inline_write(text)
    if inline:
        return inline

    code = _extract_code_block(text)
    if code:
        return f"WRITE solution.py\n{code}"

    if re.match(r"^(def |class |import |from )", text):
        return f"WRITE solution.py\n{text}"

    return text
