#!/usr/bin/env bash
# GPU 4 — tool_taubench v6 — Qwen3-4B-Thinking
export CUDA_VISIBLE_DEVICES=4
export ARB_SINGLE_GPU=1
export ARB_DEVICE_MAP=0
export ARB_RESUME=1
export FRESH=0
source "$(dirname "$0")/_common.sh"
run_gpu_model_eval 4 \
  "/home/test/test12/models/Qwen3-4B-Thinking-2507" \
  "qwen3-4b-thinking" \
  1024
