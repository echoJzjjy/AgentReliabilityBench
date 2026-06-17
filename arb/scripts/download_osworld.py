#!/usr/bin/env python3
"""Download OSWorld task definitions from GitHub."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arb.gui.download import download_osworld
from arb.utils.config import load_config, resolve_path


def main():
    parser = argparse.ArgumentParser(description="Download OSWorld evaluation tasks")
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--index",
        default=None,
        help="Task index file (default: test_small.json; use test_all.json for full set)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = resolve_path(cfg, "raw_osworld_dir")
    index = args.index or cfg.get("osworld_index", "test_small.json")
    paths = download_osworld(out_dir, index_name=index)
    print("Downloaded OSWorld data:")
    for k, p in paths.items():
        print(f"  {k}: {p}")


if __name__ == "__main__":
    main()
