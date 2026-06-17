#!/usr/bin/env bash
# GPU 6 — tool_taubench v6 — Qwen3-8B
export CUDA_VISIBLE_DEVICES=6
export ARB_SINGLE_GPU=1
export ARB_DEVICE_MAP=0
export ARB_RESUME=1
export FRESH=0
source "$(dirname "$0")/_common.sh"
run_gpu_model_eval 6 \
  "/home/test/test12/models/Qwen/Qwen3-8B" \
  "qwen3-8b" \
  1024
