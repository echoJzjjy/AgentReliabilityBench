"""Orchestrate building all four text benchmark cells from GSM8K."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from arb.text.blocked import build_blocked_instance
from arb.text.impossible_no_exploit import build_impossible_no_exploit
from arb.text.impossible_with_exploit import build_impossible_with_exploit
from arb.text.solvable import build_solvable_instance
from arb.utils.io import load_jsonl_list, write_jsonl


def _sample_rows(rows: list[dict[str, Any]], n: int | None, seed: int) -> list[dict[str, Any]]:
    if n is None or n >= len(rows):
        return rows
    rng = random.Random(seed)
    indices = rng.sample(range(len(rows)), n)
    return [rows[i] for i in sorted(indices)]


def build_text_benchmark(
    raw_split_path: Path,
    output_dir: Path,
    *,
    sample_size: int | None = None,
    seed: int = 42,
    mask_token: str = "[UNKNOWN]",
    use_llm_slots: bool = False,
    local_model_path: str | None = None,
    impossible_no_exploit_ratio: dict[str, float] | None = None,
) -> dict[str, Path]:
    rows = load_jsonl_list(raw_split_path)
    split = raw_split_path.stem
    rows = _sample_rows(rows, sample_size, seed)

    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    solvable = [build_solvable_instance(r, i).to_dict() for i, r in enumerate(rows)]

    blocked: list[dict[str, Any]] = []
    for i, r in enumerate(rows):
        inst = build_blocked_instance(
            r, i, mask_token=mask_token, use_llm=use_llm_slots, model_path=local_model_path
        )
        if inst:
            blocked.append(inst.to_dict())

    ratio = impossible_no_exploit_ratio or {"underdetermined": 0.5, "conflicting": 0.5}
    impossible_no: list[dict[str, Any]] = []
    for i, r in enumerate(rows):
        t = "underdetermined" if rng.random() < ratio.get("underdetermined", 0.5) else "conflicting"
        inst = build_impossible_no_exploit(r, i, rng, force_type=t)
        if inst:
            impossible_no.append(inst.to_dict())

    impossible_exploit: list[dict[str, Any]] = []
    for i, r in enumerate(rows):
        inst = build_impossible_with_exploit(r, i, rng)
        if inst:
            impossible_exploit.append(inst.to_dict())

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
        "counts": {k: len(load_jsonl_list(v)) for k, v in {
            "solvable": paths["solvable"],
            "blocked_but_recoverable": paths["blocked_but_recoverable"],
            "impossible_no_exploit": paths["impossible_no_exploit"],
            "impossible_with_exploit": paths["impossible_with_exploit"],
        }.items()},
        "seed": seed,
        "use_llm_slots": use_llm_slots,
    }
    manifest_path = output_dir / f"{split}_manifest.json"
    import json

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return paths
