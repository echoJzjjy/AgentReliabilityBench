#!/usr/bin/env bash
# GPU 0 — code_lcb — Qwen3-30B-A3B-Instruct (MoE; resume with FRESH=0 SKIP_EXISTING=1)
source "$(dirname "$0")/_common.sh"
run_gpu_model_eval 0 \
  "/home/test/test12/models/Qwen3-30B-A3B-Instruct-2507" \
  "qwen3-30b-a3b-instruct" \
  1024
