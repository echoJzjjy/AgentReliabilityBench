"""Parse and normalize τ-bench (tau-bench) task definitions."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

ORDER_ID_RE = re.compile(r"#W\d{7}")
ZIP_RE = re.compile(r"\b(\d{5})\b")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.\w+\b", re.I)
USER_ID_RE = re.compile(r"\b([a-z]+_[a-z]+_\d{4})\b")


def _tau_repo_for_tasks(path: Path) -> Path | None:
    for parent in path.parents:
        if (parent / "tau_bench").is_dir() and (parent / "tau_bench" / "types.py").is_file():
            return parent
    root = Path(__file__).resolve().parents[2] / "data" / "raw" / "taubench" / "repo"
    if (root / "tau_bench").is_dir():
        return root
    return None


def _import_tasks_module(path: Path) -> list[Any]:
    import importlib.util
    import sys

    repo = _tau_repo_for_tasks(path)
    if repo is None:
        raise ValueError(f"Cannot import τ-bench tasks module without repo: {path}")
    repo_str = str(repo.resolve())
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    spec = importlib.util.spec_from_file_location(f"tau_tasks_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for var_name in ("TASKS_TEST", "TASKS", "TASKS_TRAIN", "TASKS_DEV", "tasks"):
        tasks = getattr(mod, var_name, None)
        if isinstance(tasks, list):
            return tasks
    raise ValueError(f"No task list in {path}")


def parse_tasks_py(path: Path) -> list[dict[str, Any]]:
    """Load tasks*.py (dict list or pydantic Task list) into ARB raw rows."""
    text = path.read_text(encoding="utf-8")
    if "from tau_bench" in text or "TASKS_TEST" in text:
        tasks = _import_tasks_module(path)
        return [normalize_tau_task(t, idx, domain_from_path(path)) for idx, t in enumerate(tasks)]

    var_name = None
    for candidate in ("TASKS_TEST", "TASKS_TRAIN", "TASKS_DEV", "tasks"):
        if f"{candidate} = " in text:
            var_name = candidate
            break
    if var_name is None:
        raise ValueError(f"No tasks assignment in {path}")
    chunk = text.split(f"{var_name} = ", 1)[1]
    chunk = f"{var_name} = " + chunk
    end = chunk.rfind("]")
    if end < 0:
        raise ValueError(f"Malformed tasks list in {path}")
    chunk = chunk[: end + 1]
    ns: dict[str, Any] = {}
    exec(compile(chunk, str(path), "exec"), ns)  # noqa: S102 — trusted benchmark data file
    tasks = ns.get(var_name) or ns.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError(f"Expected list in {path}")
    return [normalize_tau_task(t, idx, domain_from_path(path)) for idx, t in enumerate(tasks)]


def domain_from_path(path: Path) -> str:
    for part in path.parts:
        if part in ("retail", "airline", "telecom"):
            return part
    return "retail"


def normalize_tau_task(raw: Any, index: int, domain: str) -> dict[str, Any]:
    """Map τ-bench task dict or pydantic Task to ARB raw row."""
    if hasattr(raw, "model_dump"):
        raw = raw.model_dump()
    elif not isinstance(raw, dict):
        raw = dict(raw)

    actions = []
    for a in raw.get("actions", []):
        if hasattr(a, "model_dump"):
            a = a.model_dump()
        args = a.get("arguments") if "arguments" in a else a.get("kwargs", {})
        actions.append({"name": a["name"], "arguments": args or {}})

    outputs = raw.get("outputs", raw.get("expected_outputs", []))

    return {
        "source_dataset": "sierra-research/tau-bench",
        "source_id": f"{domain}_{index}",
        "split": "test",
        "domain": domain,
        "user_id": raw.get("user_id", ""),
        "user_instruction": raw.get("instruction", ""),
        "gold_actions": actions,
        "expected_outputs": outputs,
        "annotator": raw.get("annotator"),
        "metadata": {"annotator": raw.get("annotator")},
    }


def load_tasks_json(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "tasks" in data:
        data = data["tasks"]
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}")
    domain = path.stem.replace("_tasks", "").replace("_full", "").replace("taubench_", "")
    if domain.endswith(".json"):
        domain = "retail"
    if "retail" in path.name:
        domain = "retail"
    elif "airline" in path.name:
        domain = "airline"
    else:
        domain = domain or "retail"
    return [normalize_tau_task(t, i, domain) for i, t in enumerate(data)]


def load_wiki(domain: str, repo_dir: Path | None = None) -> str:
    if repo_dir:
        wiki_path = repo_dir / "tau_bench" / "envs" / domain / "wiki.md"
        if wiki_path.is_file():
            return wiki_path.read_text(encoding="utf-8")
    fixture = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / f"taubench_{domain}_wiki.md"
    if fixture.is_file():
        return fixture.read_text(encoding="utf-8")
    return f"# {domain} domain policy\n\n(Fallback stub — install τ-bench for full wiki.)\n"


def merge_domain_rows(raw_dir: Path, domains: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for domain in domains:
        jsonl = raw_dir / f"{domain}.jsonl"
        if jsonl.is_file():
            from arb.utils.io import load_jsonl_list

            rows.extend(load_jsonl_list(jsonl))
            continue
        tasks_py = raw_dir / "repo" / "tau_bench" / "envs" / domain / "tasks.py"
        if tasks_py.is_file():
            rows.extend(parse_tasks_py(tasks_py))
            continue
        for name in (f"taubench_{domain}_full.json", f"taubench_{domain}_tasks.json"):
            fix = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / name
            if fix.is_file():
                rows.extend(load_tasks_json(fix))
                break
    return rows


def extract_maskable_slots(instruction: str, index: int) -> list[dict[str, Any]]:
    """Rule-based slot detection for blocked-but-recoverable."""
    slots: list[dict[str, Any]] = []

    for i, m in enumerate(ORDER_ID_RE.finditer(instruction)):
        oid = m.group(0)
        slots.append(
            {
                "slot_id": f"order_id_{i}",
                "slot_type": "order_id",
                "original_text": oid,
                "masked_token": "[ORDER_ID_UNKNOWN]",
                "clarification_keywords": ["order", "order id", "order number", oid.lower()],
            }
        )

    zips = list(ZIP_RE.finditer(instruction))
    if zips:
        m = zips[0]
        z = m.group(1)
        slots.append(
            {
                "slot_id": "zip_code",
                "slot_type": "zip_code",
                "original_text": z,
                "masked_token": "[ZIP_UNKNOWN]",
                "clarification_keywords": ["zip", "zip code", "postal", z],
            }
        )

    emails = list(EMAIL_RE.finditer(instruction))
    if emails:
        e = emails[0].group(0)
        slots.append(
            {
                "slot_id": "email",
                "slot_type": "email",
                "original_text": e,
                "masked_token": "[EMAIL_UNKNOWN]",
                "clarification_keywords": ["email", e.lower()],
            }
        )

    if not slots:
        return []

    return slots


def pick_mask_slot(slots: list[dict[str, Any]], index: int) -> list[dict[str, Any]]:
    """Pick one slot deterministically per index."""
    if not slots:
        return []
    return [slots[index % len(slots)]]


AUTH_TOOLS = {"find_user_id_by_email", "find_user_id_by_name_zip", "get_user_details"}


def normalize_gold_actions(
    domain: str,
    actions: list[dict[str, Any]],
    user_id: str = "",
) -> list[dict[str, Any]]:
    """Map retail-only auth tools to airline-native tools when user_id is known."""
    if domain != "airline":
        return actions
    out: list[dict[str, Any]] = []
    for act in actions:
        if act.get("name") == "find_user_id_by_name_zip" and user_id:
            out.append({"name": "get_user_details", "arguments": {"user_id": user_id}})
        elif act.get("name") == "find_user_id_by_email" and user_id:
            out.append({"name": "get_user_details", "arguments": {"user_id": user_id}})
        else:
            out.append(act)
    return out


def actions_require_slot(actions: list[dict[str, Any]], slot: dict[str, Any]) -> bool:
    """Whether gold actions need the masked value."""
    orig = slot["original_text"]
    stype = slot["slot_type"]
    for act in actions:
        args = act.get("arguments", {})
        blob = json.dumps(args).lower()
        if stype == "order_id" and orig.lower() in blob:
            return True
        if stype == "zip_code" and orig in json.dumps(args):
            return True
        if stype == "email" and orig.lower() in blob:
            return True
    return False
