#!/usr/bin/env python3
"""Download GSM8K dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arb.text.download import download_gsm8k
from arb.utils.config import load_config, resolve_path


def main():
    parser = argparse.ArgumentParser(description="Download openai/gsm8k to JSONL")
    parser.add_argument("--config", default=None, help="Path to config YAML")
    parser.add_argument("--output-dir", default=None, help="Override raw_gsm8k_dir")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = Path(args.output_dir) if args.output_dir else resolve_path(cfg, "raw_gsm8k_dir")
    download_gsm8k(out)
    print(f"Done. Raw GSM8K at {out}")


if __name__ == "__main__":
    main()
