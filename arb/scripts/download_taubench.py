#!/usr/bin/env python3
"""Download / materialize τ-bench tasks for ARB tool substrate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arb.tool.download import download_taubench
from arb.utils.config import load_config, resolve_path


def main():
    parser = argparse.ArgumentParser(description="Download τ-bench tasks (git clone or fixtures)")
    parser.add_argument("--config", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--domains",
        default="retail,airline",
        help="Comma-separated domains (retail, airline)",
    )
    parser.add_argument(
        "--fixtures-only",
        action="store_true",
        help="Skip git clone; use bundled tests/fixtures only",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = Path(args.output_dir) if args.output_dir else resolve_path(cfg, "raw_taubench_dir")
    domains = tuple(d.strip() for d in args.domains.split(",") if d.strip())
    paths = download_taubench(out, domains=domains, use_fixtures_only=args.fixtures_only)
    for k, p in paths.items():
        print(f"{k}: {p}")


if __name__ == "__main__":
    main()
