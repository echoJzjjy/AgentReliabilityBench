#!/usr/bin/env bash
# tool_taubench — Llama-4-Scout 17B MoE: 八卡 GPU 0–7（device_map=auto，避免四卡 OOM/极慢）
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ARB_SINGLE_GPU=0
export ARB_DEVICE_MAP=auto
export ARB_GPU_MEMORY_RATIO="${ARB_GPU_MEMORY_RATIO:-0.85}"
export ARB_GPU_RESERVE_GIB="${ARB_GPU_RESERVE_GIB:-3}"
export ARB_MIN_TOTAL_GPU_GIB="${ARB_MIN_TOTAL_GPU_GIB:-120}"
source "$(dirname "$0")/_common.sh"
run_gpu_model_eval 0 \
  "/home/test/test12/models/llama-4-scout-17b-16e-instruct" \
  "llama-4-scout-17b" \
  1024
