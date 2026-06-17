"""Download / materialize τ-bench tasks into ARB raw JSONL."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from arb.utils.io import ensure_dir, write_jsonl
from arb.utils.taubench import load_tasks_json, load_wiki, merge_domain_rows, parse_tasks_py

TAU_BENCH_REPO = "https://github.com/sierra-research/tau-bench.git"
DOMAINS_DEFAULT = ("retail", "airline")


def _try_git_clone(dest: Path) -> bool:
    if (dest / "tau_bench").is_dir():
        return True
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", TAU_BENCH_REPO, str(dest)],
            check=True,
            capture_output=True,
            timeout=180,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
        print(f"git clone failed: {exc}")
        return False


def _export_domain_from_repo(
    repo_dir: Path, domain: str, out_dir: Path, *, split: str = "test"
) -> list[dict[str, Any]]:
    env_dir = repo_dir / "tau_bench" / "envs" / domain
    tasks_py = env_dir / f"tasks_{split}.py"
    if not tasks_py.is_file():
        tasks_py = env_dir / "tasks.py"
    if not tasks_py.is_file():
        return []
    rows = parse_tasks_py(tasks_py)
    for r in rows:
        r["split"] = split
    wiki = load_wiki(domain, repo_dir)
    for r in rows:
        r["domain_policy"] = wiki
    out_path = out_dir / f"{domain}.jsonl"
    write_jsonl(out_path, rows)
    wiki_dst = out_dir / f"{domain}_wiki.md"
    wiki_dst.write_text(wiki, encoding="utf-8")
    return rows


def _export_from_fixtures(out_dir: Path, domains: tuple[str, ...]) -> dict[str, int]:
    fixture_root = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
    counts: dict[str, int] = {}
    for domain in domains:
        rows: list[dict[str, Any]] = []
        for name in (f"taubench_{domain}_full.json", f"taubench_{domain}_tasks.json"):
            fix = fixture_root / name
            if fix.is_file():
                rows = load_tasks_json(fix)
                break
        if not rows and domain == "airline":
            fix = fixture_root / "taubench_airline_tasks.json"
            if fix.is_file():
                rows = load_tasks_json(fix)
        if not rows:
            print(f"Warning: no fixture for domain {domain}; skipping")
            counts[domain] = 0
            continue
        wiki = load_wiki(domain)
        for r in rows:
            r["domain_policy"] = wiki
        write_jsonl(out_dir / f"{domain}.jsonl", rows)
        (out_dir / f"{domain}_wiki.md").write_text(wiki, encoding="utf-8")
        counts[domain] = len(rows)
        print(f"Fixture export: {domain} -> {len(rows)} tasks")
    return counts


def download_taubench(
    output_dir: str | Path,
    *,
    domains: tuple[str, ...] = DOMAINS_DEFAULT,
    use_fixtures_only: bool = False,
) -> dict[str, Path]:
    """Populate data/raw/taubench with per-domain JSONL."""
    output_dir = ensure_dir(output_dir)
    repo_dir = output_dir / "repo"
    counts: dict[str, int] = {}
    backend = "fixture"

    if not use_fixtures_only and _try_git_clone(repo_dir):
        backend = "git"
        for domain in domains:
            n = len(_export_domain_from_repo(repo_dir, domain, output_dir))
            counts[domain] = n
            print(f"Exported {n} tasks for {domain} from repo")

    if sum(counts.values()) == 0:
        print("Using bundled fixtures (offline / clone failed).")
        counts = _export_from_fixtures(output_dir, domains)
        backend = "fixture"

    all_rows = merge_domain_rows(output_dir, list(domains))
    combined = output_dir / "all.jsonl"
    write_jsonl(combined, all_rows)

    meta = {
        "dataset": "sierra-research/tau-bench",
        "domains": list(domains),
        "backend": backend,
        "counts": counts,
        "combined_count": len(all_rows),
        "combined_file": str(combined),
        "note": "For latest tasks use tau2-bench; ARB v1 uses tau-bench retail+airline schema.",
    }
    meta_path = output_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {len(all_rows)} combined rows -> {combined}")
    return {"all": combined, "meta": meta_path}
