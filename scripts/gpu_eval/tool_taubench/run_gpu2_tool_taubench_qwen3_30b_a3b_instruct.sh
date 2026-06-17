#!/usr/bin/env bash
# GPU 2 — tool_taubench v6 — Qwen3-30B-A3B-Instruct
export CUDA_VISIBLE_DEVICES=2
export ARB_SINGLE_GPU=1
export ARB_DEVICE_MAP=0
source "$(dirname "$0")/_common.sh"
run_gpu_model_eval 2 \
  "/home/test/test12/models/Qwen3-30B-A3B-Instruct-2507" \
  "qwen3-30b-a3b-instruct" \
  1024
