# AgentReliabilityBench — 合作者复现指南

本包包含 **4 基底 × 4 状态 × 7 模型** 的完整实验结果与可复现代码。

## 实验矩阵

| 基底 | 结果目录 | 上游数据 | Benchmark 版本 |
|------|----------|----------|----------------|
| **Text** (GSM8K) | `results/models/text_gsm8k/` | `data/raw/gsm8k/` | seed=42, ~100/97/96/100 |
| **Code** (LiveCodeBench) | `results/models/code_lcb/` | `data/raw/livecodebench/` | seed=42, ~200/190/197/196 |
| **Tool** (τ-bench) | `results/models/tool_taubench/` | `data/raw/taubench/` + `repo/tau_bench/` | **v6_publication**, 157/73/157/157 |
| **GUI** (OSWorld 模拟层) | `results/models/gui_osworld_v4/` | `data/raw/osworld/` | sim_layer v3, 各 39 题 |

### 四种状态（所有基底一致）

1. `solvable` — 可解
2. `blocked_but_recoverable` — 遮蔽但可恢复（需求助）
3. `impossible_no_exploit` — 不可能（无 exploit）
4. `impossible_with_exploit` — 不可能（含 exploit 面）

### 七个评测模型

| model_slug | HuggingFace / 本地路径（需自行下载） |
|------------|--------------------------------------|
| `qwen3-4b-instruct` | `Qwen3-4B-Instruct-2507` |
| `qwen3-4b-thinking` | `Qwen3-4B-Thinking-2507` |
| `qwen3-30b-a3b-instruct` | `Qwen3-30B-A3B-Instruct-2507` |
| `qwen3-30b-a3b-thinking` | `Qwen3-30B-A3B-Thinking-2507` |
| `qwen3-8b` | `Qwen/Qwen3-8B` |
| `phi-4-reasoning` | `Phi-4-reasoning` |
| `llama-4-scout-17b` | `llama-4-scout-17b-16e-instruct` |

原始跑分机器将权重放在 `/home/test/test12/models/`。复现时请下载到本地目录，并修改 `scripts/gpu_eval/*/` 下各 `run_gpu*.sh` 中的路径，或设置环境变量后批量替换。

## 环境准备

```bash
cd AgentReliabilityBench
bash scripts/setup_env.sh
conda activate arb
pip install -r requirements.txt -r requirements-llm.txt
python -c "import torch; print(torch.cuda.is_available())"  # 需 True
```

推荐：**A800 + CUDA 12.1**，conda 环境名 `arb`。

## 数据与 Benchmark 构建

若包内已含 `data/benchmarks/` 与 `data/raw/`，可跳过下载直接评测。否则：

```bash
# 1) 下载原始数据（需网络；τ-bench 会 git clone）
python -m arb.scripts.download_gsm8k
python -m arb.scripts.download_livecodebench
python -m arb.scripts.download_taubench
python -m arb.scripts.download_osworld --index test_small.json

# 2) 构建四态 benchmark（seed=42，与论文一致）
python -m arb.scripts.build_text_benchmark --split test --sample-size 100
python -m arb.scripts.build_code_benchmark --split test
python -m arb.scripts.build_tool_benchmark
python -m arb.scripts.build_gui_benchmark --split test
```

**Tool 评测必读**：设置 `export ARB_TOOL_BACKEND=tau`，并确保存在  
`data/raw/taubench/repo/tau_bench/envs/{retail,airline}/data/*.json`。

## 全量 GPU 评测（7 模型并行）

```bash
# Text
bash scripts/gpu_eval/gsm8k/launch_nohup.sh

# Code
bash scripts/gpu_eval/lcb/launch_nohup.sh

# Tool（τ-bench v6）
export ARB_TOOL_BACKEND=tau
bash scripts/gpu_eval/tool_taubench/launch_nohup.sh

# GUI
bash scripts/gpu_eval/osworld/launch_nohup.sh
```

单模型冒烟（示例）：

```bash
python -m arb.scripts.run_model_full_tool_eval \
  --model-path /path/to/Qwen3-4B-Instruct-2507 \
  --model-slug qwen3-4b-instruct \
  --benchmark-dir data/benchmarks/tool \
  --output-dir results/models/tool_taubench/qwen3-4b-instruct \
  --states solvable --limit 5
```

断点续跑：各 GPU 脚本支持 `ARB_RESUME=1`、`FRESH=0`（见 `tool_taubench/run_gpu*.sh`）。

## 结果文件说明

每个 `results/models/<group>/<model-slug>/` 目录：

| 文件 | 说明 |
|------|------|
| `run_manifest.json` | 四状态汇总指标与路径（**论文主表来源**） |
| `test_<state>_results.jsonl` | 逐题 metrics |
| `test_<state>_results.summary.json` | 单状态聚合 |
| `traces/<state>/<task_id>.txt` | 对话/轨迹（可选审计） |

全项目汇总表：`results/SUMMARY.csv`（由 `scripts/export_paper_summary.py` 生成）。

## 验证完整性

```bash
python scripts/export_paper_summary.py --check
pytest tests/ -q
```

`--check` 会确认 4 组 × 7 模型 × 4 状态均有 `run_manifest.json`。

## Tool (τ-bench) 已知限制（写进论文 Limitations）

- 用户模拟为 **规则型** `ArbRuleUserSimulationEnv`，非官方 LLM User
- Blocked 仅 **73** 道 retail 题（airline 无可用遮蔽槽）
- `exploit_usage` 在当前 gold 设计下多为 0%
- 部分 read-only gold 任务统一记为 `false_completion`

## 目录结构（交付包）

```
AgentReliabilityBench/
├── arb/                    # 核心库（text/code/tool/gui + agents）
├── config/default.yaml
├── data/
│   ├── benchmarks/{text,code,tool,gui}/
│   └── raw/{gsm8k,livecodebench,taubench,osworld}/
├── results/
│   ├── SUMMARY.csv
│   └── models/{text_gsm8k,code_lcb,tool_taubench,gui_osworld_v4}/
├── scripts/
│   ├── setup_env.sh
│   ├── export_paper_summary.py
│   └── gpu_eval/{gsm8k,lcb,tool_taubench,osworld}/
├── tests/
├── REPRODUCE.md            # 本文件
├── README.md
└── requirements*.txt
```
