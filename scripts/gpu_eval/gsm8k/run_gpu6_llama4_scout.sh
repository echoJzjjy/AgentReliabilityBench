#!/usr/bin/env bash
# text_gsm8k — Llama-4-Scout 17B MoE: 4-GPU on 0–3 (device_map=auto + max_memory)
export CUDA_VISIBLE_DEVICES=0,1,2,3
export ARB_SINGLE_GPU=0
export ARB_DEVICE_MAP=auto
export ARB_MIN_TOTAL_GPU_GIB="${ARB_MIN_TOTAL_GPU_GIB:-60}"
source "$(dirname "$0")/_common.sh"
# gpu_id only used when CUDA_VISIBLE_DEVICES unset; keep 0 for logging
run_gpu_model_eval 0 \
  "/home/test/test12/models/llama-4-scout-17b-16e-instruct" \
  "llama-4-scout-17b" \
  512
