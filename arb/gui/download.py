"""Download OSWorld evaluation task definitions from GitHub."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from tqdm import tqdm

from arb.utils.io import ensure_dir, write_jsonl
from arb.utils.osworld import normalize_osworld_task

OSWORLD_REPO = "xlang-ai/OSWorld"
OSWORLD_BRANCH = "main"
RAW_BASE = f"https://raw.githubusercontent.com/{OSWORLD_REPO}/{OSWORLD_BRANCH}/evaluation_examples"


def _fetch_json(url: str) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "AgentReliabilityBench/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _download_task(domain: str, task_id: str, examples_dir: Path) -> Path | None:
    rel = f"examples/{domain}/{task_id}.json"
    url = f"{RAW_BASE}/{rel}"
    out = examples_dir / domain / f"{task_id}.json"
    if out.exists():
        return out
    try:
        data = _fetch_json(url)
    except urllib.error.HTTPError:
        return None
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def download_osworld(
    output_dir: str | Path,
    *,
    index_name: str = "test_small.json",
) -> dict[str, Path]:
    """Download OSWorld task index + referenced JSON configs; write normalized JSONL."""
    output_dir = ensure_dir(output_dir)
    examples_dir = output_dir / "examples"
    index_path = output_dir / index_name

    print(f"Fetching OSWorld index {index_name}...")
    index = _fetch_json(f"{RAW_BASE}/{index_name}")
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")

    records: list[dict[str, Any]] = []
    idx = 0
    pairs = [(domain, tid) for domain, ids in index.items() for tid in ids]

    for domain, task_id in tqdm(pairs, desc="download tasks"):
        path = _download_task(domain, task_id, examples_dir)
        if path is None:
            continue
        task = json.loads(path.read_text(encoding="utf-8"))
        records.append(normalize_osworld_task(task, domain=domain, index=idx, split="test"))
        idx += 1

    out_path = output_dir / "test.jsonl"
    write_jsonl(out_path, records)

    meta = {
        "dataset": "osworld",
        "repo": OSWORLD_REPO,
        "index": index_name,
        "count": len(records),
        "file": str(out_path),
    }
    meta_path = output_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} rows -> {out_path}")
    return {"test": out_path, "meta": meta_path, "index": index_path}
