# text_gsm8k — 七卡本地评测（GSM8K 文本四状态）

| GPU | 脚本 | 日志 |
|-----|------|------|
| 0 | `run_gpu0_qwen3_4b_instruct.sh` | `logs/nohup/text_gsm8k_gpu0_qwen3_4b_instruct.log` |
| 1 | `run_gpu1_qwen3_4b_thinking.sh` | `logs/nohup/text_gsm8k_gpu1_qwen3_4b_thinking.log` |
| 2 | `run_gpu2_qwen3_30b_a3b_instruct.sh` | `logs/nohup/text_gsm8k_gpu2_qwen3_30b_a3b_instruct.log` |
| 3 | `run_gpu3_qwen3_30b_a3b_thinking.sh` | `logs/nohup/text_gsm8k_gpu3_qwen3_30b_a3b_thinking.log` |
| 4 | `run_gpu4_qwen3_8b.sh` | `logs/nohup/text_gsm8k_gpu4_qwen3_8b.log` |
| 5 | `run_gpu5_phi4_reasoning.sh` | `logs/nohup/text_gsm8k_gpu5_phi4_reasoning.log` |
| 0–3 | `run_gpu6_llama4_scout.sh` | `logs/nohup/text_gsm8k_llama4_scout_4gpu03.log` |

## 一键启动

```bash
cd /home/test/test12/songzijun/AgentReliabilityBench
bash scripts/gpu_eval/gsm8k/launch_nohup.sh
```

## 单卡调试

```bash
LIMIT=5 bash scripts/gpu_eval/gsm8k/run_gpu0_qwen3_4b_instruct.sh
```

结果目录：`results/models/text_gsm8k/<model-slug>/`

## 重跑（覆盖旧结果）

每次运行会先 `rm -rf results/models/text_gsm8k/<model-slug>/`，再写入新结果。默认 `MAX_TURNS=5`。

七条 nohup（在项目根目录执行）：

```bash
cd /home/test/test12/songzijun/AgentReliabilityBench
mkdir -p logs/nohup

nohup bash scripts/gpu_eval/gsm8k/run_gpu0_qwen3_4b_instruct.sh > logs/nohup/text_gsm8k_gpu0_qwen3_4b_instruct.log 2>&1 &
nohup bash scripts/gpu_eval/gsm8k/run_gpu1_qwen3_4b_thinking.sh > logs/nohup/text_gsm8k_gpu1_qwen3_4b_thinking.log 2>&1 &
nohup bash scripts/gpu_eval/gsm8k/run_gpu2_qwen3_30b_a3b_instruct.sh > logs/nohup/text_gsm8k_gpu2_qwen3_30b_a3b_instruct.log 2>&1 &
nohup bash scripts/gpu_eval/gsm8k/run_gpu3_qwen3_30b_a3b_thinking.sh > logs/nohup/text_gsm8k_gpu3_qwen3_30b_a3b_thinking.log 2>&1 &
nohup bash scripts/gpu_eval/gsm8k/run_gpu4_qwen3_8b.sh > logs/nohup/text_gsm8k_gpu4_qwen3_8b.log 2>&1 &
nohup bash scripts/gpu_eval/gsm8k/run_gpu5_phi4_reasoning.sh > logs/nohup/text_gsm8k_gpu5_phi4_reasoning.log 2>&1 &
nohup bash scripts/gpu_eval/gsm8k/run_gpu6_llama4_scout.sh > logs/nohup/text_gsm8k_llama4_scout_4gpu03.log 2>&1 &
```

## Llama-4-Scout（四卡：GPU 0–3）

`run_gpu6_llama4_scout.sh` 使用：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3
ARB_SINGLE_GPU=0
ARB_DEVICE_MAP=auto
ARB_MIN_TOTAL_GPU_GIB=60
```

**勿与** `run_gpu0`–`run_gpu3` 单卡任务同时占用 GPU 0–3。显存不足时可设 `ARB_QUANTIZE_4BIT=1`（需 `bitsandbytes`）。

备选（单卡 4bit，需 `pip install bitsandbytes`）：

```bash
export CUDA_VISIBLE_DEVICES=0   # 一张相对空的卡
export ARB_SINGLE_GPU=1
export ARB_QUANTIZE_4BIT=1
```

```bash
bash scripts/gpu_eval/gsm8k/diagnose_llama_gpu.sh
nohup bash scripts/gpu_eval/gsm8k/run_gpu6_llama4_scout.sh \
  > logs/nohup/text_gsm8k_llama4_scout_4gpu03.log 2>&1 &
```
