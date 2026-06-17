"""Orchestrate building all four tool benchmark cells from τ-bench."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from arb.tool.backend import is_scorable_task
from arb.tool.blocked import build_blocked_instance
from arb.tool.impossible_no_exploit import build_impossible_no_exploit
from arb.tool.impossible_with_exploit import build_impossible_with_exploit
from arb.tool.solvable import build_solvable_instance
from arb.utils.io import load_jsonl_list, write_jsonl


def _filter_scorable(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if is_scorable_task(r)]


def _stratified_sample(rows: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    """Keep domain mix; prefer retail+airline coverage when sampling."""
    if n >= len(rows):
        return rows
    rng = random.Random(seed)
    by_domain: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_domain.setdefault(r.get("domain", "retail"), []).append(r)

    picked: list[dict[str, Any]] = []
    for domain in sorted(by_domain.keys()):
        pool = by_domain[domain]
        if pool:
            picked.append(pool[rng.randrange(len(pool))])

    remaining = [r for r in rows if r not in picked]
    rng.shuffle(remaining)
    for r in remaining:
        if len(picked) >= n:
            break
        if r not in picked:
            picked.append(r)

    if len(picked) < n:
        extra = [r for r in rows if r not in picked]
        rng.shuffle(extra)
        picked.extend(extra[: n - len(picked)])

    return picked[:n]


def _sample_blocked_instances(
    scorable_rows: list[dict[str, Any]],
    n: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Build blocked instances from full scorable pool until n reached."""
    candidates: list[dict[str, Any]] = []
    for i, r in enumerate(scorable_rows):
        inst = build_blocked_instance(r, i)
        if inst:
            candidates.append(inst.to_dict())

    if n >= len(candidates):
        return candidates
    return _stratified_sample(candidates, n, seed + 7)


def build_tool_benchmark(
    raw_path: Path,
    output_dir: Path,
    *,
    sample_size: int | None = None,
    seed: int = 42,
    impossible_no_exploit_ratio: dict[str, float] | None = None,
) -> dict[str, Path]:
    rows = load_jsonl_list(raw_path)
    split = raw_path.stem.replace("_all", "")
    if split not in ("test", "train"):
        split = "test"
    for r in rows:
        r.setdefault("split", split)

    scorable_all = _filter_scorable(rows)
    base_rows = _stratified_sample(scorable_all, sample_size, seed) if sample_size else scorable_all

    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    solvable: list[dict[str, Any]] = []
    for i, r in enumerate(base_rows):
        inst = build_solvable_instance(r, i)
        if inst:
            solvable.append(inst.to_dict())

    blocked_n = sample_size or len(base_rows)
    blocked = _sample_blocked_instances(scorable_all, blocked_n, seed)

    ratio = impossible_no_exploit_ratio or {
        "record_not_found": 0.34,
        "policy_forbidden": 0.33,
        "permission_denied": 0.33,
    }
    types = list(ratio.keys())
    weights = [ratio[t] for t in types]
    impossible_no: list[dict[str, Any]] = []
    for i, r in enumerate(base_rows):
        t = types[rng.choices(range(len(types)), weights=weights)[0]]
        inst = build_impossible_no_exploit(r, i, rng, force_type=t)
        if inst:
            impossible_no.append(inst.to_dict())

    impossible_exploit = [
        build_impossible_with_exploit(r, i, rng).to_dict() for i, r in enumerate(base_rows)
    ]

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

    blocked_pool = sum(
        1 for i, r in enumerate(scorable_all) if build_blocked_instance(r, i)
    )
    manifest = {
        "split": split,
        "source": str(raw_path),
        "substrate": "tool",
        "version": "v6_publication",
        "backend": "tau_bench_v6",
        "counts": {k: len(load_jsonl_list(v)) for k, v in paths.items()},
        "seed": seed,
        "scorable_pool_size": len(scorable_all),
        "blocked_buildable_pool": blocked_pool,
        "eval_sample_size": sample_size,
        "notes": (
            "Evaluation uses real τ-bench SQLite env (ARB_TOOL_BACKEND=tau). "
            "v6 reward replays gold with unpatched tools (fixes impossible-state false positives). "
            "Blocked count capped by maskable retail slots (73); airline has no maskable slots."
        ),
    }
    manifest_path = output_dir / f"{split}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return paths
