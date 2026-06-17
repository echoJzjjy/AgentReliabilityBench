#!/usr/bin/env bash
# GPU 5 — gui_osworld_v4 — Qwen3-30B-A3B-Thinking
source "$(dirname "$0")/_common.sh"
run_gpu_model_eval 5 \
  "/home/test/test12/models/Qwen3-30B-A3B-Thinking-2507" \
  "qwen3-30b-a3b-thinking" \
  1024
