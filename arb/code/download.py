"""Download LiveCodeBench code_generation_lite without requiring pyarrow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from arb.utils.io import ensure_dir, write_jsonl
from arb.utils.livecodebench import normalize_lcb_row

# Maps release tag -> cumulative jsonl files on HuggingFace
RELEASE_FILES: dict[str, list[str]] = {
    "release_v1": ["test.jsonl"],
    "release_v2": ["test.jsonl", "test2.jsonl"],
    "release_v3": ["test.jsonl", "test2.jsonl", "test3.jsonl"],
    "release_v4": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl"],
    "release_v5": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl"],
    "release_v6": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"],
}


def _download_via_datasets(release: str) -> list[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(
        "livecodebench/code_generation_lite",
        split="test",
        version_tag=release,
        trust_remote_code=True,
    )
    return [dict(row) for row in ds]


def _download_via_hub(release: str) -> list[dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    files = RELEASE_FILES.get(release, RELEASE_FILES["release_v1"])
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []

    for fname in files:
        path = hf_hub_download(
            repo_id="livecodebench/code_generation_lite",
            filename=fname,
            repo_type="dataset",
        )
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("version https://git-lfs"):
                    continue
                obj = json.loads(line)
                qid = str(obj.get("question_id", ""))
                if qid in seen:
                    continue
                seen.add(qid)
                rows.append(obj)
    return rows


def download_livecodebench(
    output_dir: str | Path,
    *,
    release: str = "release_v1",
) -> dict[str, Path]:
    """Download LiveCodeBench and write normalized JSONL."""
    output_dir = ensure_dir(output_dir)
    print(f"Fetching livecodebench/code_generation_lite ({release})...")

    raw_rows: list[dict[str, Any]]
    backend = "hub"
    try:
        raw_rows = _download_via_datasets(release)
        backend = "datasets"
    except Exception as exc:
        print(f"datasets loader unavailable ({exc}); falling back to huggingface_hub jsonl download")
        raw_rows = _download_via_hub(release)

    records = [
        normalize_lcb_row(row, idx, split="test")
        for idx, row in enumerate(tqdm(raw_rows, desc="normalize"))
    ]

    out_path = output_dir / "test.jsonl"
    write_jsonl(out_path, records)

    meta = {
        "dataset": "livecodebench/code_generation_lite",
        "release": release,
        "backend": backend,
        "count": len(records),
        "file": str(out_path),
    }
    meta_path = output_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} rows -> {out_path}")
    return {"test": out_path, "meta": meta_path}
