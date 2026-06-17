#!/usr/bin/env bash
# GPU 4 — code_lcb — Qwen3-8B
source "$(dirname "$0")/_common.sh"
run_gpu_model_eval 0 \
  "/home/test/test12/models/Qwen/Qwen3-8B" \
  "qwen3-8b" \
  1024
