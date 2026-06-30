# AgentReliabilityBench

AgentReliabilityBench is a 4x4 benchmark for evaluating LLM agents across four task substrates and four task states:
text, code, tool use, and GUI interaction; solvable, blocked-but-recoverable, impossible-no-exploit, and impossible-with-exploit.

The repository contains the benchmark construction code, evaluation harnesses, scripts for running experiments, and utilities for aggregating results.
The main paper table is generated from `results/SUMMARY.csv`.

## Overview

The benchmark is organized around two axes:

| Axis | Values |
|---|---|
| Substrate | Text, Code, Tool, GUI |
| State | Solvable, Blocked-but-Recoverable, Impossible-No-Exploit, Impossible-With-Exploit |

This yields 16 evaluation settings in total.

## Repository Layout

```text
arb/
  text/ code/ tool/ gui/   core benchmark logic for the four substrates
  agents/                  local HF agents, scripted baselines, API stubs
  scripts/                 CLI entry points for building and evaluating benchmarks
config/default.yaml        default configuration
data/benchmarks/           generated benchmark JSONL files
data/raw/                  downloaded source datasets and task definitions
results/models/            per-model run outputs
results/SUMMARY.csv       aggregated paper table
scripts/gpu_eval/          launch scripts for large-scale evaluation
tests/                     unit tests and fixtures
REPRODUCE.md               reproducibility notes
```

## Installation

The project targets Python 3.10+.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-llm.txt
pip install -e .[dev]
```

For local LLM evaluation, install the `llm` extra as well:

```bash
pip install -e .[llm]
```

## Data and Benchmark Construction

Each substrate provides scripts for downloading source data and building the four-state benchmark.

```bash
python -m arb.scripts.download_gsm8k
python -m arb.scripts.build_text_benchmark --split test --sample-size 100

python -m arb.scripts.download_livecodebench
python -m arb.scripts.build_code_benchmark --split test

python -m arb.scripts.download_taubench
python -m arb.scripts.build_tool_benchmark --sample-size 100

python -m arb.scripts.download_osworld --index test_small.json
python -m arb.scripts.build_gui_benchmark --split test --sample-size 100
```

## Evaluation

The repository includes both lightweight baselines and local-model evaluation entry points.

```bash
python -m arb.scripts.run_text_eval --benchmark data/benchmarks/text/test_solvable.jsonl --agent honest_baseline --limit 20
python -m arb.scripts.run_code_eval --benchmark data/benchmarks/code/test_solvable.jsonl --agent honest_baseline --limit 20
python -m arb.scripts.run_tool_eval --benchmark data/benchmarks/tool/test_solvable.jsonl --agent honest_baseline --limit 20
python -m arb.scripts.run_gui_eval --benchmark data/benchmarks/gui/test_solvable.jsonl --agent honest_baseline --limit 20
```

For local HF models, use `local_hf` and set `--model-path` or `ARB_LOCAL_MODEL_PATH`.

## Results

Each run directory under `results/models/<group>/<model-slug>/` contains:

| File | Meaning |
|---|---|
| `run_manifest.json` | run configuration and aggregated metrics |
| `test_<state>_results.jsonl` | per-episode traces |
| `test_<state>_results.summary.json` | state-level summary |
| `traces/<state>/<task_id>.txt` | optional transcripts and execution traces |

The aggregate table used in the paper is produced by:

```bash
python scripts/export_paper_summary.py --check
```

## Validation

```bash
pytest tests/ -q
```

## Citation

If you use AgentReliabilityBench in your work, please cite the accompanying paper.

