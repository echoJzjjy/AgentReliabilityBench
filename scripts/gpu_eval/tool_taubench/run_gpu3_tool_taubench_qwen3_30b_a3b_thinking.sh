#!/usr/bin/env bash
# GPU 5 — tool_taubench v6 — Qwen3-30B-A3B-Thinking
export CUDA_VISIBLE_DEVICES=5
export ARB_SINGLE_GPU=1
export ARB_DEVICE_MAP=0
export ARB_RESUME=1
export FRESH=0
source "$(dirname "$0")/_common.sh"
run_gpu_model_eval 5 \
  "/home/test/test12/models/Qwen3-30B-A3B-Thinking-2507" \
  "qwen3-30b-a3b-thinking" \
  1024
