#!/usr/bin/env bash
# GPU 3 — text_gsm8k — Qwen3-30B-A3B-Thinking (MoE)
source "$(dirname "$0")/_common.sh"
run_gpu_model_eval 3 \
  "/home/test/test12/models/Qwen3-30B-A3B-Thinking-2507" \
  "qwen3-30b-a3b-thinking" \
  2048
