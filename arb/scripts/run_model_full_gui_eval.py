#!/usr/bin/env python3
"""Load one local HF model once, sequentially eval all four GUI benchmark splits."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arb.agents.local_hf import LocalHFAgent
from arb.gui.metrics import aggregate_results
from arb.scripts.run_gui_eval import run_episode
from arb.gui.environment import GuiTaskEnvironment
from arb.utils.io import read_jsonl

GUI_BENCHMARKS = [
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
        f"parser_assist_count: {result.get('parser_assist_count', 0)}",
        f"model_native_success: {result.get('model_native_success')}",
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
    model_slug: str,
) -> dict:
    results = []
    tasks = list(read_jsonl(benchmark_path))
    if limit:
        tasks = tasks[:limit]

    for task in tqdm(tasks, desc=benchmark_path.stem):
        env = GuiTaskEnvironment(task)
        ep = run_episode(env, agent, max_turns=max_turns, model_slug=model_slug)
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
    parser = argparse.ArgumentParser(description="Full 4-state GUI eval for one local model")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-slug", required=True, help="Short name for output directory")
    parser.add_argument(
        "--benchmark-dir",
        default="data/benchmarks/gui",
        help="Directory containing test_*.jsonl OSWorld-derived files",
    )
    parser.add_argument("--output-dir", default=None, help="Default: results/models/gui_osworld/<slug>")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-turns", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--export-traces", action="store_true", default=True)
    parser.add_argument("--no-export-traces", action="store_false", dest="export_traces")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a state if its *_results.jsonl already exists in output-dir",
    )
    args = parser.parse_args()

    benchmark_dir = Path(args.benchmark_dir)
    if not benchmark_dir.is_absolute():
        benchmark_dir = ROOT / benchmark_dir

    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else ROOT / "results" / "models" / "gui_osworld_v4" / args.model_slug
    )
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model_path}")
    os.environ["ARB_MODEL_SLUG"] = args.model_slug
    agent = LocalHFAgent(args.model_path, max_new_tokens=args.max_new_tokens)

    manifest: dict = {
        "substrate": "gui",
        "dataset": "osworld",
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

    for state_name, filename in GUI_BENCHMARKS:
        bench_path = benchmark_dir / filename
        if not bench_path.exists():
            raise FileNotFoundError(
                f"Missing benchmark: {bench_path}. "
                "Run: python -m arb.scripts.download_osworld && "
                "python -m arb.scripts.build_gui_benchmark"
            )

        print(f"\n=== Evaluating {state_name} ({filename}) ===")
        result_path = out_dir / filename.replace(".jsonl", "_results.jsonl")
        if args.skip_existing and result_path.exists():
            summary_path = result_path.with_suffix(".summary.json")
            summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
            print(f"Skipping (existing): {result_path}")
            manifest["states"][state_name] = {
                "benchmark": str(bench_path),
                "results": str(result_path),
                "summary": str(summary_path) if summary_path.exists() else None,
                "traces_dir": str(out_dir / "traces" / state_name),
                "metrics": summary,
                "skipped": True,
            }
            continue

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
            model_slug=args.model_slug,
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
