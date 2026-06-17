#!/usr/bin/env bash
# gui_osworld — Llama-4-Scout 17B MoE: shard across GPU 0–7 (8×80GiB; avoids 4-card OOM)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ARB_SINGLE_GPU=0
export ARB_DEVICE_MAP=auto
export ARB_GPU_MEMORY_RATIO="${ARB_GPU_MEMORY_RATIO:-0.85}"
export ARB_GPU_RESERVE_GIB="${ARB_GPU_RESERVE_GIB:-3}"
source "$(dirname "$0")/_common.sh"
# gpu_id only for logging when CUDA_VISIBLE_DEVICES is preset above
run_gpu_model_eval 0 \
  "/home/test/test12/models/llama-4-scout-17b-16e-instruct" \
  "llama-4-scout-17b" \
  512
