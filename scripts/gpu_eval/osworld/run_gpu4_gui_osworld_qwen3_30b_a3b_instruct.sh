#!/usr/bin/env bash
# GPU 4 — gui_osworld_v4 — Qwen3-30B-A3B-Instruct
source "$(dirname "$0")/_common.sh"
run_gpu_model_eval 4 \
  "/home/test/test12/models/Qwen3-30B-A3B-Instruct-2507" \
  "qwen3-30b-a3b-instruct" \
  512
