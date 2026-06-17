"""Download GSM8K from HuggingFace datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from arb.utils.gsm8k import normalize_gsm8k_row
from arb.utils.io import ensure_dir, write_jsonl


def download_gsm8k(output_dir: str | Path) -> dict[str, Path]:
    """Download openai/gsm8k and write normalized JSONL per split."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("Install datasets: pip install datasets") from e

    output_dir = ensure_dir(output_dir)
    paths: dict[str, Path] = {}

    print("Loading openai/gsm8k from HuggingFace...")
    ds = load_dataset("openai/gsm8k", 'main')

    for split in ds.keys():
        records: list[dict[str, Any]] = []
        for idx, row in enumerate(tqdm(ds[split], desc=f"gsm8k/{split}")):
            base = normalize_gsm8k_row(dict(row), idx, split)
            records.append(base)

        out_path = output_dir / f"{split}.jsonl"
        write_jsonl(out_path, records)
        paths[split] = out_path
        print(f"Wrote {len(records)} rows -> {out_path}")

    meta = {
        "dataset": "openai/gsm8k",
        "splits": list(paths.keys()),
        "files": {k: str(v) for k, v in paths.items()},
    }
    meta_path = output_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return paths
