#!/usr/bin/env bash
# tool_taubench v6 — Llama-4-Scout: 四卡 GPU 0,1,2,3
export CUDA_VISIBLE_DEVICES=0,1,2,3
export ARB_SINGLE_GPU=0
export ARB_DEVICE_MAP=auto
source "$(dirname "$0")/_common.sh"
run_gpu_model_eval 0 \
  "/home/test/test12/models/llama-4-scout-17b-16e-instruct" \
  "llama-4-scout-17b" \
  1024
