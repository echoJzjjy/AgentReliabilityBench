# AgentReliabilityBench

4×4 任务–状态统一评测框架：**文本（GSM8K）**、**代码（LiveCodeBench）**、**工具调用（τ-bench v6）**、**GUI（OSWorld 模拟层）** 四种基底 × 四种状态 × 7 个本地模型。

**合作者复现**：见 [`REPRODUCE.md`](REPRODUCE.md)。论文主表汇总：`results/SUMMARY.csv`。

## 运行环境（conda `arb`）

机器为 **A800 + CUDA 12.1**。在 `arb` 环境中安装依赖：

```bash
cd /home/test/test12/songzijun/AgentReliabilityBench
bash scripts/setup_env.sh
conda activate arb
```

或手动安装：

```bash
conda activate arb
pip install -r requirements.txt
pip install -r requirements-llm.txt   # torch+cu121, transformers, accelerate
python -c "import torch; print(torch.cuda.is_available())"  # 应为 True
```

## 快速开始

```bash
cd /home/test/test12/songzijun/AgentReliabilityBench
conda activate arb
pip install -r requirements.txt

# 1) 下载 GSM8K（开源数据，HuggingFace datasets）
python -m arb.scripts.download_gsm8k

# 2) 构建四种状态的 benchmark JSONL
python -m arb.scripts.build_text_benchmark --split test --sample-size 100

# 可选：用本地 Qwen 辅助识别遮蔽槽位（需 GPU）
python -m arb.scripts.build_text_benchmark --split test --sample-size 100 --use-llm-slots \
  --local-model-path /home/test/test12/models/Qwen3-4B-Instruct-2507

# 3) 评测（先用无 GPU 的 scripted baseline 冒烟）
python -m arb.scripts.run_text_eval \
  --benchmark data/benchmarks/text/test_blocked_but_recoverable.jsonl \
  --agent honest_baseline --limit 20

# 本地开源模型评测
python -m arb.scripts.run_text_eval \
  --benchmark data/benchmarks/text/test_solvable.jsonl \
  --agent local_hf \
  --model-path /home/test/test12/models/Qwen3-4B-Instruct-2507 \
  --limit 50
```

### 工具调用（τ-bench）

```bash
python -m arb.scripts.download_taubench          # 有网 git clone；无网自动 fixtures
python -m arb.scripts.build_tool_benchmark --sample-size 100
python -m arb.scripts.run_tool_eval \
  --benchmark data/benchmarks/tool/test_solvable.jsonl \
  --agent honest_baseline --limit 20
```

详见 `writing/outline.md` §3.5.1。

## 目录结构

```
arb/
  text/ code/ tool/ gui/   # 四基底四种状态
  agents/                 # local_hf / scripted baselines
  scripts/                # CLI 入口
config/default.yaml
data/benchmarks/ + data/raw/
results/models/{text_gsm8k,code_lcb,tool_taubench,gui_osworld_v4}/
scripts/gpu_eval/{gsm8k,lcb,tool_taubench,osworld}/
tests/
```

## 四种状态（文本）

| 状态 | 文件后缀 | 生成模块 |
|------|----------|----------|
| Solvable | `*_solvable.jsonl` | `arb/text/solvable.py` |
| Blocked-but-Recoverable | `*_blocked_but_recoverable.jsonl` | `arb/text/blocked.py` |
| Impossible-No-Exploit | `*_impossible_no_exploit.jsonl` | `arb/text/impossible_no_exploit.py` |
| Impossible-With-Exploit | `*_impossible_with_exploit.jsonl` | `arb/text/impossible_with_exploit.py` |

交互协议：`arb/text/environment.py`（`CLARIFY:` 求助通道、判分、exploit 检测）。

## 模型与 API

### 已支持：本地开源（无需 API）

- 默认配置指向 `/home/test/test12/models/Qwen3-4B-Instruct-2507`
- 可通过 `ARB_LOCAL_MODEL_PATH` 或 `--local-model-path` 覆盖
- 论文主实验建议的 Qwen3 / Gemma 等路径均在 `/home/test/test12/models/` 下自行选择

### 需用户自行配置：API 闭源模型

以下在 `arb/agents/api_stub.py` 中实现，**需自行安装 `pip install -e ".[api]"` 并设置密钥**：

| 提供方 | 论文建议模型 | 环境变量 |
|--------|--------------|----------|
| OpenAI | `gpt-4.1` | `OPENAI_API_KEY` |
| Anthropic | Claude Sonnet 4 | `ANTHROPIC_API_KEY` |

在 Python 中：

```python
from arb.agents.api_stub import OpenAIAgent, AnthropicAgent
agent = OpenAIAgent(model="gpt-4.1")
```

## 配置

`config/default.yaml` 或环境变量：

- `ARB_DATA_DIR` — 数据根目录
- `ARB_LOCAL_MODEL_PATH` — 本地 HF 模型路径

## 测试

```bash
pip install pytest
pytest tests/ -q
```

## 指标

`run_text_eval` 输出 per-episode `metrics`，并由 `arb/text/metrics.py` 聚合：

- `success_rate`
- `help_seeking_rate` / `help_quality`（blocked）
- `honest_failure_rate` / `false_completion`（impossible-no-exploit）
- `exploit_usage` / `surface_pass`（impossible-with-exploit：`success` 仅统计未走捷径的答对；`surface_pass` 含捷径伪通过）
