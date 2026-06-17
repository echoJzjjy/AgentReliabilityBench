#!/usr/bin/env python3
"""Build 4 code task-state cells from normalized LiveCodeBench."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arb.code.build import build_code_benchmark
from arb.utils.config import load_config, resolve_path


def main():
    parser = argparse.ArgumentParser(description="Build code 4-state benchmark from LiveCodeBench")
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_dir = resolve_path(cfg, "raw_livecodebench_dir")
    out_dir = resolve_path(cfg, "code_benchmark_dir")
    split_path = raw_dir / f"{args.split}.jsonl"
    if not split_path.exists():
        raise FileNotFoundError(
            f"Missing {split_path}. Run: python -m arb.scripts.download_livecodebench"
        )

    sample = args.sample_size
    if sample is None:
        sample = cfg.get("code_sample_size", {}).get(args.split)

    ratio = cfg.get("code_impossible_no_exploit", {})
    paths = build_code_benchmark(
        split_path,
        out_dir,
        sample_size=sample,
        seed=args.seed or cfg.get("seed", 42),
        mask_placeholder=cfg.get("code_blocked", {}).get("mask_placeholder"),
        impossible_no_exploit_ratio=ratio,
    )
    print("Built code benchmark files:")
    for k, p in paths.items():
        print(f"  {k}: {p}")


if __name__ == "__main__":
    main()
