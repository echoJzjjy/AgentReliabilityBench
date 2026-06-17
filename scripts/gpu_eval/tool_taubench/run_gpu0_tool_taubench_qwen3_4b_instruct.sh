#!/usr/bin/env bash
# GPU 0 — tool_taubench v6 — Qwen3-4B-Instruct
export CUDA_VISIBLE_DEVICES=0
export ARB_SINGLE_GPU=1
export ARB_DEVICE_MAP=0
source "$(dirname "$0")/_common.sh"
run_gpu_model_eval 0 \
  "/home/test/test12/models/Qwen3-4B-Instruct-2507" \
  "qwen3-4b-instruct" \
  1024
