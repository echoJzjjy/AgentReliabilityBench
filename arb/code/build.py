"""Orchestrate building all four code benchmark cells from LiveCodeBench."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from arb.code.blocked import build_blocked_instance
from arb.code.impossible_no_exploit import build_impossible_no_exploit
from arb.code.impossible_with_exploit import build_impossible_with_exploit
from arb.code.solvable import build_solvable_instance
from arb.utils.io import load_jsonl_list, write_jsonl


def _sample_rows(rows: list[dict[str, Any]], n: int | None, seed: int) -> list[dict[str, Any]]:
    if n is None or n >= len(rows):
        return rows
    rng = random.Random(seed)
    indices = rng.sample(range(len(rows)), n)
    return [rows[i] for i in sorted(indices)]


def build_code_benchmark(
    raw_split_path: Path,
    output_dir: Path,
    *,
    sample_size: int | None = None,
    seed: int = 42,
    mask_placeholder: str | None = None,
    impossible_no_exploit_ratio: dict[str, float] | None = None,
) -> dict[str, Path]:
    rows = load_jsonl_list(raw_split_path)
    split = raw_split_path.stem
    rows = _sample_rows(rows, sample_size, seed)

    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    placeholder = mask_placeholder or "[SPEC_REDACTED — use clarification channels to recover]"

    solvable = [build_solvable_instance(r, i).to_dict() for i, r in enumerate(rows)]

    blocked: list[dict[str, Any]] = []
    for i, r in enumerate(rows):
        inst = build_blocked_instance(r, i, mask_placeholder=placeholder)
        if inst:
            blocked.append(inst.to_dict())

    ratio = impossible_no_exploit_ratio or {"spec_test_conflict": 0.5, "missing_external_resource": 0.5}
    impossible_plans: list[tuple[int, dict[str, Any], str]] = []
    for i, r in enumerate(rows):
        t = (
            "spec_test_conflict"
            if rng.random() < ratio.get("spec_test_conflict", 0.5)
            else "missing_external_resource"
        )
        impossible_plans.append((i, r, t))

    impossible_no: list[dict[str, Any]] = []
    impossible_exploit: list[dict[str, Any]] = []
    for i, r, t in impossible_plans:
        inst_no = build_impossible_no_exploit(r, i, rng, force_type=t)
        inst_ex = build_impossible_with_exploit(r, i, rng, force_type=t)
        if inst_no:
            impossible_no.append(inst_no.to_dict())
        if inst_ex:
            impossible_exploit.append(inst_ex.to_dict())

    paths = {
        "solvable": output_dir / f"{split}_solvable.jsonl",
        "blocked_but_recoverable": output_dir / f"{split}_blocked_but_recoverable.jsonl",
        "impossible_no_exploit": output_dir / f"{split}_impossible_no_exploit.jsonl",
        "impossible_with_exploit": output_dir / f"{split}_impossible_with_exploit.jsonl",
    }
    write_jsonl(paths["solvable"], solvable)
    write_jsonl(paths["blocked_but_recoverable"], blocked)
    write_jsonl(paths["impossible_no_exploit"], impossible_no)
    write_jsonl(paths["impossible_with_exploit"], impossible_exploit)

    manifest = {
        "split": split,
        "source": str(raw_split_path),
        "substrate": "code",
        "counts": {
            k: len(load_jsonl_list(v))
            for k, v in paths.items()
        },
        "seed": seed,
    }
    manifest_path = output_dir / f"{split}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return paths
