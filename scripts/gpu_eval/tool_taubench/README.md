# tool_taubench — v6 投稿版

## v6 修复要点

- **Impossible 判分**：`compute_arb_tau_reward` 用**未 patch** 的 gold 工具重放 GT DB hash，消除 v5 中 REPORT_FAILURE 仍 `tau_reward=1` 的假阳性
- **真实 τ-bench**：SQLite + 官方 outputs 检查；用户侧为 `ArbRuleUserSimulationEnv`（论文需披露）
- **Benchmark**：`v6_publication`，157/73/157/157（blocked 仅 retail 可遮蔽槽位）

## GPU 布局

| 脚本 | 物理 GPU | 模型 |
|------|----------|------|
| `run_tool_taubench_llama4_scout_4gpu.sh` | **0–3** 四卡 | Llama-4-Scout |
| `run_gpu0_tool_taubench_qwen3_4b_instruct.sh` | **0** | Qwen3-4B-Instruct |
| `run_gpu1_tool_taubench_qwen3_4b_thinking.sh` | **1** | Qwen3-4B-Thinking |
| `run_gpu2_tool_taubench_qwen3_30b_a3b_instruct.sh` | **2** | Qwen3-30B-A3B-Instruct |
| `run_gpu3_tool_taubench_qwen3_30b_a3b_thinking.sh` | **3** | Qwen3-30B-A3B-Thinking |
| `run_gpu4_tool_taubench_qwen3_8b.sh` | **4** | Qwen3-8B |
| `run_gpu5_tool_taubench_phi4_reasoning.sh` | **5** | Phi-4-reasoning |

Llama 与 GPU 0–3 上单卡任务冲突，勿并行。

## 重跑前

```bash
conda activate arb
cd /home/test/test12/songzijun/AgentReliabilityBench
pip install pydantic  # τ-bench Task 类型
python -m arb.scripts.download_taubench
python -m arb.scripts.build_tool_benchmark
export ARB_TOOL_BACKEND=tau
chmod +x scripts/gpu_eval/tool_taubench/*.sh
mkdir -p logs/nohup
```

默认 `FRESH=1` 覆盖 `results/models/tool_taubench/<slug>/`。

## 七条 nohup

```bash
cd /home/test/test12/songzijun/AgentReliabilityBench
export FRESH=1 ARB_TOOL_BACKEND=tau

nohup bash scripts/gpu_eval/tool_taubench/run_gpu0_tool_taubench_qwen3_4b_instruct.sh \
  > logs/nohup/tool_taubench_gpu0_qwen3_4b_instruct.log 2>&1 &

nohup bash scripts/gpu_eval/tool_taubench/run_gpu1_tool_taubench_qwen3_4b_thinking.sh \
  > logs/nohup/tool_taubench_gpu1_qwen3_4b_thinking.log 2>&1 &

nohup bash scripts/gpu_eval/tool_taubench/run_gpu2_tool_taubench_qwen3_30b_a3b_instruct.sh \
  > logs/nohup/tool_taubench_gpu2_qwen3_30b_a3b_instruct.log 2>&1 &

nohup bash scripts/gpu_eval/tool_taubench/run_gpu3_tool_taubench_qwen3_30b_a3b_thinking.sh \
  > logs/nohup/tool_taubench_gpu3_qwen3_30b_a3b_thinking.log 2>&1 &

nohup bash scripts/gpu_eval/tool_taubench/run_gpu4_tool_taubench_qwen3_8b.sh \
  > logs/nohup/tool_taubench_gpu4_qwen3_8b.log 2>&1 &

nohup bash scripts/gpu_eval/tool_taubench/run_gpu5_tool_taubench_phi4_reasoning.sh \
  > logs/nohup/tool_taubench_gpu5_phi4_reasoning.log 2>&1 &

nohup bash scripts/gpu_eval/tool_taubench/run_tool_taubench_llama4_scout_4gpu.sh \
  > logs/nohup/tool_taubench_llama4_scout_4gpu.log 2>&1 &
```
