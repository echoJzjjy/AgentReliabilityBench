#!/usr/bin/env python3
"""Load one local HF model once, sequentially eval all four tool benchmark splits."""

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

from arb.agents.tool_hf import ToolLocalHFAgent
from arb.scripts.run_tool_eval import run_episode
from arb.tool.factory import backend_label, make_tool_environment
from arb.tool.metrics import aggregate_results
from arb.utils.io import read_jsonl

TOOL_BENCHMARKS = [
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
        f"gold_aligned: {result.get('gold_aligned')}",
        f"exploit_detect_reason: {result.get('exploit_detect_reason', '')}",
        "--- transcript ---",
    ]
    for turn in result.get("transcript", []):
        role = turn.get("role", "?")
        extra = ""
        if turn.get("tool_name"):
            extra = f" ({turn['tool_name']})"
        lines.append(f"[{role}{extra}] {turn.get('content', '')}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def eval_benchmark(
    agent: ToolLocalHFAgent,
    benchmark_path: Path,
    output_path: Path,
    *,
    limit: int | None,
    max_turns: int,
    traces_dir: Path | None,
    resume: bool = False,
) -> dict:
    results: list[dict] = []
    completed: set[str] = set()
    if resume and output_path.is_file():
        with output_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                results.append(r)
                completed.add(r.get("task_id", ""))

    tasks = list(read_jsonl(benchmark_path))
    if limit:
        tasks = tasks[:limit]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _flush() -> None:
        with output_path.open("w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    for task in tqdm(tasks, desc=benchmark_path.stem):
        task_id = task.get("id", "")
        if task_id and task_id in completed:
            continue
        env = make_tool_environment(task)
        ep = run_episode(env, agent, max_turns=max_turns)
        results.append(ep)
        completed.add(ep.get("task_id", task_id))
        _flush()
        if traces_dir:
            export_trace_txt(ep, traces_dir / f"{ep['task_id']}.txt")

    summary = aggregate_results(results)
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Full 4-state tool eval for one local model")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-slug", required=True, help="Short name for output directory")
    parser.add_argument(
        "--benchmark-dir",
        default="data/benchmarks/tool",
        help="Directory containing test_*.jsonl τ-bench-derived files",
    )
    parser.add_argument("--output-dir", default=None, help="Default: results/models/tool_taubench/<slug>")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--export-traces", action="store_true", default=True)
    parser.add_argument("--no-export-traces", action="store_false", dest="export_traces")
    parser.add_argument(
        "--fresh",
        action="store_true",
        default=False,
        help="Overwrite results (full output-dir if all states; per-state if --states)",
    )
    parser.add_argument(
        "--states",
        default=None,
        help="Comma-separated subset: solvable,blocked_but_recoverable,"
        "impossible_no_exploit,impossible_with_exploit",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Skip task_ids already present in each state's results JSONL",
    )
    args = parser.parse_args()

    benchmark_dir = Path(args.benchmark_dir)
    if not benchmark_dir.is_absolute():
        benchmark_dir = ROOT / benchmark_dir

    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else ROOT / "results" / "models" / "tool_taubench" / args.model_slug
    )
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    state_filter: list[str] | None = None
    if args.states:
        state_filter = [s.strip() for s in args.states.split(",") if s.strip()]
        unknown = set(state_filter) - {n for n, _ in TOOL_BENCHMARKS}
        if unknown:
            raise ValueError(f"Unknown --states: {unknown}")

    benchmarks = [
        (n, f) for n, f in TOOL_BENCHMARKS if state_filter is None or n in state_filter
    ]
    if not benchmarks:
        raise ValueError("No benchmarks selected (check --states)")

    out_dir.mkdir(parents=True, exist_ok=True)

    if args.fresh:
        import shutil

        if state_filter is None:
            if out_dir.exists():
                shutil.rmtree(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
        else:
            for state_name, filename in benchmarks:
                stem = filename.replace(".jsonl", "")
                for pattern in (
                    f"{stem}_results.jsonl",
                    f"{stem}_results.summary.json",
                ):
                    p = out_dir / pattern
                    if p.exists():
                        p.unlink()
                traces = out_dir / "traces" / state_name
                if traces.is_dir():
                    shutil.rmtree(traces)

    manifest_path = out_dir / "run_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.setdefault("states", {})
    else:
        manifest = {
            "substrate": "tool",
            "dataset": "sierra-research/tau-bench",
            "backend": backend_label(),
            "benchmark_version": "v6_publication",
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
    manifest["partial_run_states"] = state_filter
    manifest["backend"] = backend_label()
    manifest["benchmark_version"] = "v6_publication"
    manifest["reward_impl"] = "compute_arb_tau_reward_unpatched_gold"

    print(f"Loading model: {args.model_path}")
    agent = ToolLocalHFAgent(args.model_path, max_new_tokens=args.max_new_tokens)

    for state_name, filename in benchmarks:
        bench_path = benchmark_dir / filename
        if not bench_path.exists():
            raise FileNotFoundError(
                f"Missing benchmark: {bench_path}. "
                "Run: python -m arb.scripts.download_taubench && "
                "python -m arb.scripts.build_tool_benchmark"
            )

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
            resume=args.resume,
        )
        manifest["states"][state_name] = {
            "benchmark": str(bench_path),
            "results": str(result_path),
            "summary": str(result_path.with_suffix(".summary.json")),
            "traces_dir": str(traces_dir) if traces_dir else None,
            "metrics": summary,
        }
        print(json.dumps(summary, indent=2))

    manifest["last_updated_at"] = datetime.now(timezone.utc).isoformat()
    all_present = all(
        (out_dir / fname.replace(".jsonl", "_results.jsonl")).is_file()
        for _, fname in TOOL_BENCHMARKS
    )
    if state_filter is None or all_present:
        manifest["finished_at"] = manifest["last_updated_at"]
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDone. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
