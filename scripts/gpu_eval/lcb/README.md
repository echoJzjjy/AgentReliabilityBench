# code_lcb — 七卡本地评测（LiveCodeBench 代码四状态）

| GPU | 脚本 | 日志 |
|-----|------|------|
| 0 | `run_gpu0_qwen3_4b_instruct.sh` | `logs/nohup/code_lcb_gpu0_qwen3_4b_instruct.log` |
| 1 | `run_gpu1_qwen3_4b_thinking.sh` | `logs/nohup/code_lcb_gpu1_qwen3_4b_thinking.log` |
| 2 | `run_gpu2_qwen3_30b_a3b_instruct.sh` | `logs/nohup/code_lcb_gpu2_qwen3_30b_a3b_instruct.log` |
| 3 | `run_gpu3_qwen3_30b_a3b_thinking.sh` | `logs/nohup/code_lcb_gpu3_qwen3_30b_a3b_thinking.log` |
| 4 | `run_gpu4_qwen3_8b.sh` | `logs/nohup/code_lcb_gpu4_qwen3_8b.log` |
| 5 | `run_gpu5_phi4_reasoning.sh` | `logs/nohup/code_lcb_gpu5_phi4_reasoning.log` |
| 4–7 | `run_code_lcb_llama4_scout_4gpu.sh` | `logs/nohup/code_lcb_llama4_scout_4gpu.log` |

默认 `FRESH=1`：每次运行会删除 `results/models/code_lcb/<model-slug>/` 后重跑。

**Llama-4-Scout** 使用 GPU 4–7 四卡；其余 6 模型各用单卡 GPU 0–5。

## 七模型 nohup（覆盖旧结果）

```bash
cd /home/test/test12/songzijun/AgentReliabilityBench
mkdir -p logs/nohup

nohup bash scripts/gpu_eval/lcb/run_gpu0_qwen3_4b_instruct.sh \
  > logs/nohup/code_lcb_gpu0_qwen3_4b_instruct.log 2>&1 &

nohup bash scripts/gpu_eval/lcb/run_gpu1_qwen3_4b_thinking.sh \
  > logs/nohup/code_lcb_gpu1_qwen3_4b_thinking.log 2>&1 &

nohup bash scripts/gpu_eval/lcb/run_gpu2_qwen3_30b_a3b_instruct.sh \
  > logs/nohup/code_lcb_gpu2_qwen3_30b_a3b_instruct.log 2>&1 &

nohup bash scripts/gpu_eval/lcb/run_gpu3_qwen3_30b_a3b_thinking.sh \
  > logs/nohup/code_lcb_gpu3_qwen3_30b_thinking.log 2>&1 &

nohup bash scripts/gpu_eval/lcb/run_gpu4_qwen3_8b.sh \
  > logs/nohup/code_lcb_gpu4_qwen3_8b.log 2>&1 &

nohup bash scripts/gpu_eval/lcb/run_gpu5_phi4_reasoning.sh \
  > logs/nohup/code_lcb_gpu5_phi4_reasoning.log 2>&1 &

nohup bash scripts/gpu_eval/lcb/run_code_lcb_llama4_scout_4gpu.sh \
  > logs/nohup/code_lcb_llama4_scout_4gpu.log 2>&1 &
```

前 6 个可并行（GPU 0–5）；**Llama-4 需等 GPU 4–7 空闲后再启动**（与 Qwen3-8B 的 GPU4 冲突时需错开）。

监控：`tail -f logs/nohup/code_lcb_*.log`

## 一键启动 6 模型（不含 Llama-4）

```bash
bash scripts/gpu_eval/lcb/launch_nohup.sh
```

## Llama-4 四卡单独启动

```bash
bash scripts/gpu_eval/lcb/launch_llama4_4gpu_nohup.sh
```

结果目录：`results/models/code_lcb/<model-slug>/`
