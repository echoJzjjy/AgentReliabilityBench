#!/usr/bin/env bash
# GPU 0 — text_gsm8k — Qwen3-4B-Instruct
source "$(dirname "$0")/_common.sh"
run_gpu_model_eval 0 \
  "/home/test/test12/models/Qwen3-4B-Instruct-2507" \
  "qwen3-4b-instruct" \
  512
