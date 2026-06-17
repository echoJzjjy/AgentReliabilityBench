#!/usr/bin/env bash
# GPU 1 — text_gsm8k — Qwen3-4B-Thinking
source "$(dirname "$0")/_common.sh"
run_gpu_model_eval 1 \
  "/home/test/test12/models/Qwen3-4B-Thinking-2507" \
  "qwen3-4b-thinking" \
  2048
