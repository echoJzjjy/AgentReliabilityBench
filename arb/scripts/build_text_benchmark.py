#!/usr/bin/env python3
"""Build 4 text task-state cells from normalized GSM8K."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arb.text.build import build_text_benchmark
from arb.utils.config import load_config, resolve_path


def main():
    parser = argparse.ArgumentParser(description="Build text 4-state benchmark from GSM8K")
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--use-llm-slots",
        action="store_true",
        help="Use local HF model for slot ID (needs torch/transformers + GPU)",
    )
    parser.add_argument("--local-model-path", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_dir = resolve_path(cfg, "raw_gsm8k_dir")
    out_dir = resolve_path(cfg, "text_benchmark_dir")
    split_path = raw_dir / f"{args.split}.jsonl"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing {split_path}. Run: python -m arb.scripts.download_gsm8k")

    sample = args.sample_size
    if sample is None:
        sample = cfg.get("sample_size", {}).get(args.split)

    model_path = args.local_model_path or cfg.get("local_model_path")
    ratio = cfg.get("impossible_no_exploit", {})

    paths = build_text_benchmark(
        split_path,
        out_dir,
        sample_size=sample,
        seed=args.seed or cfg.get("seed", 42),
        mask_token=cfg.get("blocked", {}).get("mask_token", "[UNKNOWN]"),
        use_llm_slots=args.use_llm_slots,
        local_model_path=model_path if args.use_llm_slots else None,
        impossible_no_exploit_ratio=ratio,
    )
    print("Built benchmark files:")
    for k, p in paths.items():
        print(f"  {k}: {p}")


if __name__ == "__main__":
    main()
