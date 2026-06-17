#!/usr/bin/env python3
"""Aggregate run_manifest.json into results/SUMMARY.csv for paper tables."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

GROUPS = {
    "text_gsm8k": "text",
    "code_lcb": "code",
    "tool_taubench": "tool",
    "gui_osworld_v4": "gui",
}
MODELS = [
    "qwen3-4b-instruct",
    "qwen3-4b-thinking",
    "qwen3-30b-a3b-instruct",
    "qwen3-30b-a3b-thinking",
    "qwen3-8b",
    "phi-4-reasoning",
    "llama-4-scout-17b",
]
STATES = [
    "solvable",
    "blocked_but_recoverable",
    "impossible_no_exploit",
    "impossible_with_exploit",
]
# slug aliases (legacy tool run used phi4-reasoning)
SLUG_ALIASES = {"phi-4-reasoning": ["phi4-reasoning"]}


def _resolve_model_dir(base: Path, model: str) -> Path | None:
    d = base / model
    if d.is_dir() and (d / "run_manifest.json").exists():
        return d
    for alt in SLUG_ALIASES.get(model, []):
        d = base / alt
        if d.is_dir() and (d / "run_manifest.json").exists():
            return d
    return None


def _state_metrics(manifest: dict, state: str) -> dict:
    info = manifest.get("states", {}).get(state, {})
    overall = info.get("metrics", {}).get("overall", {})
    by_state = info.get("metrics", {}).get("by_state", {}).get(state, {})
    return {
        "count": overall.get("count", by_state.get("count", "")),
        "success_rate": overall.get("success_rate", by_state.get("success_rate", "")),
        "help_seeking_rate": by_state.get("help_seeking_rate", ""),
        "honest_failure_rate": by_state.get("honest_failure_rate", by_state.get("honest_abstention_rate", "")),
        "false_completion": by_state.get("false_completion", ""),
        "exploit_usage": by_state.get("exploit_usage", ""),
        "task_completion_rate": by_state.get("task_completion_rate", ""),
    }


def collect(root: Path) -> list[dict]:
    rows: list[dict] = []
    for group_dir, substrate in GROUPS.items():
        base = root / "results" / "models" / group_dir
        for model in MODELS:
            mdir = _resolve_model_dir(base, model)
            if mdir is None:
                rows.append(
                    {
                        "group": group_dir,
                        "substrate": substrate,
                        "model": model,
                        "status": "MISSING",
                    }
                )
                continue
            manifest = json.loads((mdir / "run_manifest.json").read_text())
            for state in STATES:
                m = _state_metrics(manifest, state)
                rows.append(
                    {
                        "group": group_dir,
                        "substrate": substrate,
                        "model": model,
                        "state": state,
                        "status": "OK",
                        "count": m["count"],
                        "success_rate": m["success_rate"],
                        "help_seeking_rate": m["help_seeking_rate"],
                        "honest_failure_rate": m["honest_failure_rate"],
                        "false_completion": m["false_completion"],
                        "exploit_usage": m["exploit_usage"],
                        "task_completion_rate": m["task_completion_rate"],
                        "benchmark_version": manifest.get("benchmark_version", ""),
                        "backend": manifest.get("backend", ""),
                        "result_dir": str(mdir),
                    }
                )
    return rows


def check_completeness(rows: list[dict]) -> bool:
    ok = True
    for group_dir in GROUPS:
        for model in MODELS:
            subset = [r for r in rows if r.get("group") == group_dir and r.get("model") == model]
            if any(r.get("status") == "MISSING" for r in subset):
                print(f"MISSING: {group_dir}/{model}", file=sys.stderr)
                ok = False
            elif len([r for r in subset if r.get("status") == "OK"]) != len(STATES):
                print(f"INCOMPLETE: {group_dir}/{model}", file=sys.stderr)
                ok = False
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--check", action="store_true", help="Exit 1 if any model/group incomplete")
    args = parser.parse_args()
    out = args.output or (args.root / "results" / "SUMMARY.csv")
    rows = collect(args.root)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print("No data", file=sys.stderr)
        return 1
    fieldnames = list(rows[0].keys())
    for r in rows[1:]:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out} ({len(rows)} rows)")
    if args.check and not check_completeness(rows):
        return 1
    if args.check:
        print("All 4 groups × 7 models × 4 states: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
