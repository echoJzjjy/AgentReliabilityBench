"""Orchestrate building all four GUI benchmark cells from OSWorld."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from arb.gui.blocked import build_blocked_instance
from arb.gui.impossible_no_exploit import build_impossible_no_exploit
from arb.gui.impossible_with_exploit import build_impossible_with_exploit
from arb.gui.sim_spec_enrich import enrich_task_dict
from arb.gui.solvable import build_solvable_instance
from arb.utils.io import load_jsonl_list, write_jsonl


def _sample_rows(rows: list[dict[str, Any]], n: int | None, seed: int) -> list[dict[str, Any]]:
    if n is None or n >= len(rows):
        return rows
    rng = random.Random(seed)
    indices = rng.sample(range(len(rows)), n)
    return [rows[i] for i in sorted(indices)]


def build_gui_benchmark(
    raw_split_path: Path,
    output_dir: Path,
    *,
    sample_size: int | None = None,
    seed: int = 42,
    impossible_no_exploit_ratio: dict[str, float] | None = None,
) -> dict[str, Path]:
    rows = load_jsonl_list(raw_split_path)
    split = raw_split_path.stem
    rows = _sample_rows(rows, sample_size, seed)

    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    solvable = [enrich_task_dict(build_solvable_instance(r, i).to_dict()) for i, r in enumerate(rows)]

    blocked: list[dict[str, Any]] = []
    for i, r in enumerate(rows):
        inst = build_blocked_instance(r, i)
        if inst:
            blocked.append(enrich_task_dict(inst.to_dict()))

    ratio = impossible_no_exploit_ratio or {
        "target_missing": 0.25,
        "service_down": 0.25,
        "permission_denied": 0.25,
        "page_not_found": 0.25,
    }
    types = list(ratio.keys())
    weights = [ratio[t] for t in types]
    impossible_no: list[dict[str, Any]] = []
    for i, r in enumerate(rows):
        t = rng.choices(types, weights=weights, k=1)[0]
        inst = build_impossible_no_exploit(r, i, rng, force_type=t)
        if inst:
            impossible_no.append(enrich_task_dict(inst.to_dict()))

    impossible_exploit: list[dict[str, Any]] = []
    for i, r in enumerate(rows):
        inst = build_impossible_with_exploit(r, i, rng)
        if inst:
            impossible_exploit.append(enrich_task_dict(inst.to_dict()))

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
        "substrate": "gui",
        "sim_layer_version": 4,
        "counts": {k: len(load_jsonl_list(v)) for k, v in paths.items()},
        "seed": seed,
    }
    manifest_path = output_dir / f"{split}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return paths
