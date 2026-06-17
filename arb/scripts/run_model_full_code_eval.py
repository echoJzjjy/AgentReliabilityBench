#!/usr/bin/env python3
"""Load one local HF model once, sequentially eval all four code benchmark splits."""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arb.agents.local_hf import LocalHFAgent
from arb.code.environment import CodeTaskEnvironment
from arb.code.metrics import aggregate_results
from arb.scripts.run_code_eval import run_episode
from arb.utils.io import read_jsonl

CODE_BENCHMARKS = [
    ("solvable", "test_solvable.jsonl"),
    ("blocked_but_recoverable", "test_blocked_but_recoverable.jsonl"),
    ("impossible_no_exploit", "test_impossible_no_exploit.jsonl"),
    ("impossible_with_exploit", "test_impossible_with_exploit.jsonl"),
]


def _parse_trace_result(trace_path: Path) -> dict | None:
    """Rebuild episode dict from export_trace_txt format (for --resume-traces)."""
    try:
        text = trace_path.read_text(encoding="utf-8")
    except OSError:
        return None
    header, _, body = text.partition("--- transcript ---\n")
    if not header.strip():
        return None

    def grab(key: str) -> str | None:
        m = re.search(rf"^{re.escape(key)}: (.*)$", header, re.MULTILINE)
        return m.group(1).strip() if m else None

    task_id = grab("task_id")
    state = grab("state")
    if not task_id or not state:
        return None

    success_raw = grab("success")
    success = success_raw == "True" if success_raw is not None else False
    failure_reason = grab("failure_reason")
    if failure_reason in ("None", ""):
        failure_reason = None
    exploit = grab("exploit_detect_reason")
    if exploit in ("None", ""):
        exploit = None

    test_result = None
    tr_raw = grab("test_result")
    if tr_raw and tr_raw != "None":
        try:
            test_result = json.loads(tr_raw)
        except json.JSONDecodeError:
            try:
                test_result = ast.literal_eval(tr_raw)
            except (SyntaxError, ValueError):
                test_result = None

    transcript: list[dict] = []
    for line in body.splitlines():
        if not line.startswith("[") or "] " not in line:
            continue
        role_end = line.index("]")
        role = line[1:role_end]
        content = line[role_end + 2 :]
        transcript.append({"role": role, "content": content})

    return {
        "task_id": task_id,
        "state": state,
        "success": success,
        "metrics": {},
        "failure_reason": failure_reason,
        "transcript": transcript,
        "test_result": test_result,
        "exploit_detect_reason": exploit,
    }


def export_trace_txt(result: dict, path: Path) -> None:
    lines = [
        f"task_id: {result['task_id']}",
        f"state: {result['state']}",
        f"success: {result['success']}",
        f"failure_reason: {result.get('failure_reason')}",
        f"exploit_detect_reason: {result.get('exploit_detect_reason', '')}",
        f"test_result: {json.dumps(result.get('test_result'), ensure_ascii=False)}",
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
    resume_traces: bool = False,
) -> dict:
    tasks = list(read_jsonl(benchmark_path))
    if limit:
        tasks = tasks[:limit]

    by_id: dict[str, dict] = {}
    if output_path.exists():
        for row in read_jsonl(output_path):
            by_id[row["task_id"]] = row

    pending = []
    for task in tasks:
        tid = task["id"]
        if tid in by_id:
            continue
        if resume_traces and traces_dir:
            trace_path = traces_dir / f"{tid}.txt"
            if trace_path.exists():
                recovered = _parse_trace_result(trace_path)
                if recovered:
                    by_id[tid] = recovered
                    continue
        pending.append(task)

    if resume_traces and by_id:
        print(f"Resumed {len(by_id)} task(s); running {len(pending)} remaining")

    for task in tqdm(pending, desc=benchmark_path.stem):
        env = CodeTaskEnvironment(task)
        ep = run_episode(env, agent, max_turns=max_turns)
        by_id[ep["task_id"]] = ep
        if traces_dir:
            export_trace_txt(ep, traces_dir / f"{ep['task_id']}.txt")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    results = [by_id[task["id"]] for task in tasks if task["id"] in by_id]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = aggregate_results(results)
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Full 4-state code eval for one local model")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-slug", required=True, help="Short name for output directory")
    parser.add_argument(
        "--benchmark-dir",
        default="data/benchmarks/code",
        help="Directory containing test_*.jsonl LiveCodeBench-derived files",
    )
    parser.add_argument("--output-dir", default=None, help="Default: results/models/code_lcb/<slug>")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--export-traces", action="store_true", default=True)
    parser.add_argument("--no-export-traces", action="store_false", dest="export_traces")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a state if its *_results.jsonl already exists in output-dir",
    )
    parser.add_argument(
        "--resume-traces",
        action="store_true",
        help="Skip tasks with existing trace txt; rebuild partial jsonl from traces",
    )
    parser.add_argument(
        "--states",
        default=None,
        help="Comma-separated state names to run (default: all four). "
        "e.g. impossible_no_exploit,impossible_with_exploit",
    )
    args = parser.parse_args()

    benchmark_dir = Path(args.benchmark_dir)
    if not benchmark_dir.is_absolute():
        benchmark_dir = ROOT / benchmark_dir

    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else ROOT / "results" / "models" / "code_lcb" / args.model_slug
    )
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    benchmarks = list(CODE_BENCHMARKS)
    if args.states:
        wanted = {s.strip() for s in args.states.split(",") if s.strip()}
        known = {name for name, _ in CODE_BENCHMARKS}
        unknown = wanted - known
        if unknown:
            raise ValueError(f"Unknown --states: {sorted(unknown)}; known: {sorted(known)}")
        benchmarks = [(name, fn) for name, fn in CODE_BENCHMARKS if name in wanted]
        print(f"Running states: {[n for n, _ in benchmarks]}")

    print(f"Loading model: {args.model_path}")
    agent = LocalHFAgent(args.model_path, max_new_tokens=args.max_new_tokens)

    manifest_path = out_dir / "run_manifest.json"
    manifest: dict = {
        "substrate": "code",
        "dataset": "livecodebench/code_generation_lite",
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

    for state_name, filename in benchmarks:
        bench_path = benchmark_dir / filename
        if not bench_path.exists():
            raise FileNotFoundError(
                f"Missing benchmark: {bench_path}. "
                "Run: python -m arb.scripts.download_livecodebench && "
                "python -m arb.scripts.build_code_benchmark"
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
            resume_traces=args.resume_traces,
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
    if manifest_path.exists():
        try:
            prev = json.loads(manifest_path.read_text(encoding="utf-8"))
            prev_states = prev.get("states") or {}
            prev_states.update(manifest.get("states") or {})
            manifest["states"] = prev_states
        except (json.JSONDecodeError, OSError):
            pass
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDone. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
