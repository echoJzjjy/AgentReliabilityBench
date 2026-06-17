#!/usr/bin/env python3
"""Load one local HF model once, sequentially eval all four text benchmark splits."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arb.agents.local_hf import LocalHFAgent
from arb.scripts.run_text_eval import run_episode
from arb.text.environment import TextTaskEnvironment
from arb.text.metrics import aggregate_results
from arb.utils.io import read_jsonl

TEXT_BENCHMARKS = [
    ("solvable", "test_solvable.jsonl"),
    ("blocked_but_recoverable", "test_blocked_but_recoverable.jsonl"),
    ("impossible_no_exploit", "test_impossible_no_exploit.jsonl"),
    ("impossible_with_exploit", "test_impossible_with_exploit.jsonl"),
]


def export_trace_txt(result: dict, path: Path) -> None:
    lines = [
        f"task_id: {result['task_id']}",
        f"state: {result['state']}",
        f"success: {result['success']}",
        f"failure_reason: {result.get('failure_reason')}",
        f"exploit_detect_reason: {result.get('exploit_detect_reason', '')}",
        "--- transcript ---",
    ]
    for turn in result.get("transcript", []):
        lines.append(f"[{turn.get('role', '?')}] {turn.get('content', '')}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def eval_benchmark(
    agent: LocalHFAgent,
    benchmark_path: Path,
    output_path: Path,
    *,
    limit: int | None,
    max_turns: int,
    traces_dir: Path | None,
) -> dict:
    results = []
    tasks = list(read_jsonl(benchmark_path))
    if limit:
        tasks = tasks[:limit]

    for task in tqdm(tasks, desc=benchmark_path.stem):
        env = TextTaskEnvironment(task)
        ep = run_episode(env, agent, max_turns=max_turns)
        results.append(ep)
        if traces_dir:
            export_trace_txt(ep, traces_dir / f"{ep['task_id']}.txt")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = aggregate_results(results)
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Full 4-state eval for one local model")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-slug", required=True, help="Short name for output directory")
    parser.add_argument(
        "--benchmark-dir",
        default="data/benchmarks/text",
        help="Directory containing test_*.jsonl files",
    )
    parser.add_argument("--output-dir", default=None, help="Default: results/models/<slug>")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--export-traces", action="store_true", default=True)
    parser.add_argument("--no-export-traces", action="store_false", dest="export_traces")
    args = parser.parse_args()

    benchmark_dir = Path(args.benchmark_dir)
    if not benchmark_dir.is_absolute():
        benchmark_dir = ROOT / benchmark_dir

    out_dir = Path(args.output_dir) if args.output_dir else ROOT / "results" / "models" / args.model_slug
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model_path}")
    agent = LocalHFAgent(args.model_path, max_new_tokens=args.max_new_tokens)

    manifest: dict = {
        "model_slug": args.model_slug,
        "model_path": str(Path(args.model_path).resolve()),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_dir": str(benchmark_dir),
        "output_dir": str(out_dir),
        "max_new_tokens": args.max_new_tokens,
        "max_turns": args.max_turns,
        "limit": args.limit,
        "states": {},
    }

    for state_name, filename in TEXT_BENCHMARKS:
        bench_path = benchmark_dir / filename
        if not bench_path.exists():
            raise FileNotFoundError(f"Missing benchmark: {bench_path}")

        print(f"\n=== Evaluating {state_name} ({filename}) ===")
        result_path = out_dir / filename.replace(".jsonl", "_results.jsonl")
        traces_dir = out_dir / "traces" / state_name if args.export_traces else None
        if traces_dir:
            traces_dir.mkdir(parents=True, exist_ok=True)

        summary = eval_benchmark(
            agent,
            bench_path,
            result_path,
            limit=args.limit,
            max_turns=args.max_turns,
            traces_dir=traces_dir,
        )
        manifest["states"][state_name] = {
            "benchmark": str(bench_path),
            "results": str(result_path),
            "summary": str(result_path.with_suffix(".summary.json")),
            "traces_dir": str(traces_dir) if traces_dir else None,
            "metrics": summary,
        }
        print(json.dumps(summary, indent=2))

    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    manifest_path = out_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDone. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
