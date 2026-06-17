# gui_osworld — 七卡本地评测（OSWorld GUI 四状态）

| GPU | 脚本 | 日志 |
|-----|------|------|
| 0 | `run_gpu0_gui_osworld_qwen3_4b_instruct.sh` | `logs/nohup/gui_osworld_gpu0_qwen3_4b_instruct.log` |
| 1 | `run_gpu1_gui_osworld_qwen3_4b_thinking.sh` | `logs/nohup/gui_osworld_gpu1_qwen3_4b_thinking.log` |
| 2 | `run_gpu2_gui_osworld_qwen3_30b_a3b_instruct.sh` | `logs/nohup/gui_osworld_gpu2_qwen3_30b_a3b_instruct.log` |
| 3 | `run_gpu3_gui_osworld_qwen3_30b_a3b_thinking.sh` | `logs/nohup/gui_osworld_gpu3_qwen3_30b_a3b_thinking.log` |
| 4 | `run_gpu4_gui_osworld_qwen3_8b.sh` | `logs/nohup/gui_osworld_gpu4_qwen3_8b.log` |
| 5 | `run_gpu5_gui_osworld_phi4_reasoning.sh` | `logs/nohup/gui_osworld_gpu5_phi4_reasoning.log` |

**Llama-4-Scout**（~78GiB，需 **GPU 4–7 四卡** `device_map=auto`，勿用单卡 `launch_nohup.sh`）：

| 脚本 | 日志 |
|------|------|
| `run_gui_osworld_llama4_scout_4gpu.sh` | `logs/nohup/gui_osworld_llama4_scout_4gpu.log` |

`run_gpu6_gui_osworld_llama4_scout.sh` 已转发至上述四卡脚本。

## 前置

```bash
python -m arb.scripts.download_osworld --index test_small.json
python -m arb.scripts.build_gui_benchmark --split test
chmod +x scripts/gpu_eval/osworld/*.sh
```

## 一键启动（六卡并行，不含 Llama-4-Scout）

```bash
cd /home/test/test12/songzijun/AgentReliabilityBench
bash scripts/gpu_eval/osworld/launch_nohup.sh
```

## Llama-4-Scout（GPU 4–7 四卡）

此前单卡 GPU6 会因权重 offload 到 CPU 极慢/中断；请清空失败产物后四卡重跑：

```bash
rm -rf results/models/gui_osworld/llama-4-scout-17b/traces
# 可选保留 run.log：>> 追加；全新日志则 rm run.log

chmod +x scripts/gpu_eval/osworld/run_gui_osworld_llama4_scout_4gpu.sh
chmod +x scripts/gpu_eval/osworld/launch_llama4_4gpu_nohup.sh
bash scripts/gpu_eval/osworld/launch_llama4_4gpu_nohup.sh
# 或：
nohup bash scripts/gpu_eval/osworld/run_gui_osworld_llama4_scout_4gpu.sh \
  > logs/nohup/gui_osworld_llama4_scout_4gpu.log 2>&1 &
```

## 单卡调试

```bash
LIMIT=2 bash scripts/gpu_eval/osworld/run_gpu0_gui_osworld_qwen3_4b_instruct.sh
```

结果目录：`results/models/gui_osworld/<model-slug>/`

每个模型依次评测四种状态（solvable → blocked → impossible_no_exploit → impossible_with_exploit），输出：

- `test_*_results.jsonl` / `*.summary.json`
- `traces/<state>/<task_id>.txt`
- `run_manifest.json`
