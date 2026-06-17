#!/usr/bin/env bash
# GPU 5 — text_gsm8k — Phi-4-reasoning
source "$(dirname "$0")/_common.sh"
run_gpu_model_eval 5 \
  "/home/test/test12/models/Phi-4-reasoning" \
  "phi-4-reasoning" \
  2048
