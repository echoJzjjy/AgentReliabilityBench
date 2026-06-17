#!/usr/bin/env python3
"""Validate GUI benchmark construction and behavioral criteria (outline §3.7)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arb.gui.validation import validate_benchmark_file
from arb.utils.config import load_config, resolve_path


def main():
    parser = argparse.ArgumentParser(description="Validate GUI benchmark quality")
    parser.add_argument("--config", default=None)
    parser.add_argument("--benchmark", default=None, help="Single JSONL file")
    parser.add_argument("--benchmark-dir", default=None, help="Directory with four state files")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--behavioral", action="store_true", help="Run honest/exploit baseline smoke")
    args = parser.parse_args()

    cfg = load_config(args.config)
    files: list[Path] = []

    if args.benchmark:
        files = [Path(args.benchmark)]
    else:
        bench_dir = Path(args.benchmark_dir) if args.benchmark_dir else resolve_path(cfg, "gui_benchmark_dir")
        split = args.split
        for suffix in (
            "solvable",
            "blocked_but_recoverable",
            "impossible_no_exploit",
            "impossible_with_exploit",
        ):
            p = bench_dir / f"{split}_{suffix}.jsonl"
            if p.exists():
                files.append(p)

    if not files:
        raise FileNotFoundError("No benchmark files found. Run build_gui_benchmark first.")

    reports = [
        validate_benchmark_file(p, limit=args.limit, run_behavioral=args.behavioral)
        for p in files
    ]
    print(json.dumps(reports, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
