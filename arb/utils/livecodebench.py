"""Parse and normalize LiveCodeBench code_generation_lite rows."""

from __future__ import annotations

import base64
import json
import pickle
import zlib
from typing import Any


def decode_test_cases(raw: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Decode public/private test cases from JSON string or list."""
    if isinstance(raw, list):
        return raw
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = json.loads(
            pickle.loads(zlib.decompress(base64.b64decode(raw.encode("utf-8"))))
        )
    if not isinstance(parsed, list):
        return []
    return [dict(t) for t in parsed]


def decode_metadata(raw: str | dict[str, Any] | None) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def infer_test_type(tests: list[dict[str, Any]]) -> str:
    if not tests:
        return "functional"
    t0 = tests[0].get("testtype", "functional")
    return str(t0).lower()


def normalize_lcb_row(row: dict[str, Any], index: int, split: str = "test") -> dict[str, Any]:
    """Normalize one LiveCodeBench row into ARB base record."""
    public = decode_test_cases(row.get("public_test_cases", "[]"))
    private = decode_test_cases(row.get("private_test_cases", "[]"))
    meta = decode_metadata(row.get("metadata"))
    qid = str(row.get("question_id", index))
    test_type = infer_test_type(public or private)

    return {
        "source_dataset": "livecodebench/code_generation_lite",
        "source_id": qid,
        "split": split,
        "question_title": row.get("question_title", ""),
        "question_content": row.get("question_content", ""),
        "platform": row.get("platform", "unknown"),
        "contest_id": row.get("contest_id", ""),
        "contest_date": row.get("contest_date", ""),
        "starter_code": row.get("starter_code", ""),
        "difficulty": row.get("difficulty", "unknown"),
        "test_type": test_type,
        "func_name": meta.get("func_name"),
        "public_tests": public,
        "private_tests": private,
        "metadata": meta,
        "index": index,
    }
