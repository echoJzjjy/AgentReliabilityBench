#!/usr/bin/env bash
# GPU 7 — gui_osworld_v4 — Phi-4-reasoning
source "$(dirname "$0")/_common.sh"
run_gpu_model_eval 7 \
  "/home/test/test12/models/Phi-4-reasoning" \
  "phi-4-reasoning" \
  1024
