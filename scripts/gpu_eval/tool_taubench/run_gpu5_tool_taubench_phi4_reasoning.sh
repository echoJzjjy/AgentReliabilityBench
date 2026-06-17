#!/usr/bin/env bash
# GPU 7 — tool_taubench v6 — Phi-4-reasoning
export CUDA_VISIBLE_DEVICES=7
export ARB_SINGLE_GPU=1
export ARB_DEVICE_MAP=0
export ARB_RESUME=1
export FRESH=0
source "$(dirname "$0")/_common.sh"
run_gpu_model_eval 7 \
  "/home/test/test12/models/Phi-4-reasoning" \
  "phi-4-reasoning" \
  1024
