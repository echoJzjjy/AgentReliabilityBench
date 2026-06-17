#!/usr/bin/env python3
"""Build 4 GUI task-state cells from normalized OSWorld."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arb.gui.build import build_gui_benchmark
from arb.utils.config import load_config, resolve_path


def main():
    parser = argparse.ArgumentParser(description="Build GUI 4-state benchmark from OSWorld")
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_dir = resolve_path(cfg, "raw_osworld_dir")
    out_dir = resolve_path(cfg, "gui_benchmark_dir")
    split_path = raw_dir / f"{args.split}.jsonl"
    if not split_path.exists():
        raise FileNotFoundError(
            f"Missing {split_path}. Run: python -m arb.scripts.download_osworld"
        )

    sample = args.sample_size
    if sample is None:
        sample = cfg.get("gui_sample_size", {}).get(args.split)

    ratio = cfg.get("gui_impossible_no_exploit", {})
    paths = build_gui_benchmark(
        split_path,
        out_dir,
        sample_size=sample,
        seed=args.seed or cfg.get("seed", 42),
        impossible_no_exploit_ratio=ratio,
    )
    print("Built GUI benchmark files:")
    for k, p in paths.items():
        print(f"  {k}: {p}")


if __name__ == "__main__":
    main()
