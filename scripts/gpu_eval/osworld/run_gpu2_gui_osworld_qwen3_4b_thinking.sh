#!/usr/bin/env bash
# GPU 2 — gui_osworld_v4 — Qwen3-4B-Thinking
source "$(dirname "$0")/_common.sh"
run_gpu_model_eval 2 \
  "/home/test/test12/models/Qwen3-4B-Thinking-2507" \
  "qwen3-4b-thinking" \
  1024
