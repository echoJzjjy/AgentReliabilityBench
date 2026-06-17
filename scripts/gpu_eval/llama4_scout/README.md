# Llama-4-Scout — GPU 4–7 四卡评测

Llama-4-Scout（~78GiB MoE）需 `CUDA_VISIBLE_DEVICES=4,5,6,7` + `device_map=auto` 分片加载。

## 当前进度（llama-4-scout-17b）

| 数据集 | 状态 |
|--------|------|
| text_gsm8k | 已完成 |
| code_lcb | 未完成 |
| tool_taubench | 未完成（曾单卡 GPU6 极慢后中断） |
| gui_osworld | 未完成（请用 GPU 4–7 四卡脚本） |

## 一键顺序跑完剩余三个数据集

```bash
cd /home/test/test12/songzijun/AgentReliabilityBench
chmod +x scripts/gpu_eval/llama4_scout/*.sh scripts/gpu_eval/{lcb,tool_taubench,osworld}/*llama4*.sh
nohup bash scripts/gpu_eval/llama4_scout/run_remaining_4gpu.sh \
  > logs/nohup/llama4_scout_remaining_4gpu.log 2>&1 &
```

## 分数据集单独 nohup（需顺序执行，不可并行——共用 GPU 4–7）

```bash
cd /home/test/test12/songzijun/AgentReliabilityBench
mkdir -p logs/nohup

nohup bash scripts/gpu_eval/lcb/run_code_lcb_llama4_scout_4gpu.sh \
  > logs/nohup/code_lcb_llama4_scout_4gpu.log 2>&1 &

# 等上一条完成后：
nohup bash scripts/gpu_eval/tool_taubench/run_tool_taubench_llama4_scout_4gpu.sh \
  > logs/nohup/tool_taubench_llama4_scout_4gpu.log 2>&1 &

# 等上一条完成后：
nohup bash scripts/gpu_eval/osworld/run_gui_osworld_llama4_scout_4gpu.sh \
  > logs/nohup/gui_osworld_llama4_scout_4gpu.log 2>&1 &
```

监控：`tail -f logs/nohup/llama4_scout_remaining_4gpu.log`

结果：`results/models/{code_lcb,tool_taubench,gui_osworld}/llama-4-scout-17b/`
