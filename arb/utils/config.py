"""Load YAML config with env overrides."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    cfg_path = Path(path) if path else root / "config" / "default.yaml"
    with cfg_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if env_data := os.environ.get("ARB_DATA_DIR"):
        cfg["data_dir"] = env_data
        cfg["raw_gsm8k_dir"] = str(Path(env_data) / "raw" / "gsm8k")
        cfg["text_benchmark_dir"] = str(Path(env_data) / "benchmarks" / "text")
        cfg["raw_livecodebench_dir"] = str(Path(env_data) / "raw" / "livecodebench")
        cfg["code_benchmark_dir"] = str(Path(env_data) / "benchmarks" / "code")
        cfg["raw_osworld_dir"] = str(Path(env_data) / "raw" / "osworld")
        cfg["gui_benchmark_dir"] = str(Path(env_data) / "benchmarks" / "gui")
        cfg["raw_taubench_dir"] = str(Path(env_data) / "raw" / "taubench")
        cfg["tool_benchmark_dir"] = str(Path(env_data) / "benchmarks" / "tool")

    if env_model := os.environ.get("ARB_LOCAL_MODEL_PATH"):
        cfg["local_model_path"] = env_model

    return cfg


def resolve_path(cfg: dict[str, Any], key: str) -> Path:
    root = Path(__file__).resolve().parents[2]
    p = Path(cfg[key])
    if not p.is_absolute():
        p = root / p
    return p
