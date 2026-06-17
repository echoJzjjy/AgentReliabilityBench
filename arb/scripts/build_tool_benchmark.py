#!/usr/bin/env python3
"""Build four-state tool benchmark JSONL from raw τ-bench data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arb.tool.build import build_tool_benchmark
from arb.utils.config import load_config, resolve_path


def main():
    parser = argparse.ArgumentParser(description="Build tool × 4 states benchmark")
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--raw",
        default=None,
        help="Path to combined raw JSONL (default: data/raw/taubench/all.jsonl)",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw = Path(args.raw) if args.raw else resolve_path(cfg, "raw_taubench_dir") / "all.jsonl"
    out = Path(args.output_dir) if args.output_dir else resolve_path(cfg, "tool_benchmark_dir")
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)

    if not raw.is_file():
        raise FileNotFoundError(
            f"Raw data not found: {raw}\n"
            "Run: python -m arb.scripts.download_taubench [--fixtures-only]"
        )

    paths = build_tool_benchmark(
        raw,
        out,
        sample_size=args.sample_size,
        seed=seed,
        impossible_no_exploit_ratio=cfg.get("tool_impossible_no_exploit"),
    )
    for state, p in paths.items():
        print(f"{state}: {p}")


if __name__ == "__main__":
    main()
