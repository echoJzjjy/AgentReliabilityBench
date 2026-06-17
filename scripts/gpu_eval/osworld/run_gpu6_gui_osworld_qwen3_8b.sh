#!/usr/bin/env bash
# GPU 6 — gui_osworld_v4 — Qwen3-8B
source "$(dirname "$0")/_common.sh"
run_gpu_model_eval 6 \
  "/home/test/test12/models/Qwen/Qwen3-8B" \
  "qwen3-8b" \
  512
