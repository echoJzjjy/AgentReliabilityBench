#!/usr/bin/env python3
"""Download LiveCodeBench code_generation_lite from HuggingFace."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arb.code.download import download_livecodebench
from arb.utils.config import load_config, resolve_path


def main():
    parser = argparse.ArgumentParser(description="Download LiveCodeBench code_generation_lite")
    parser.add_argument("--config", default=None)
    parser.add_argument("--release", default="release_v1")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = Path(args.output_dir) if args.output_dir else resolve_path(cfg, "raw_livecodebench_dir")
    paths = download_livecodebench(out, release=args.release)
    for k, p in paths.items():
        print(f"{k}: {p}")


if __name__ == "__main__":
    main()
