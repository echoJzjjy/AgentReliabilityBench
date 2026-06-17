#!/usr/bin/env bash
# GPU 2 — text_gsm8k — Qwen3-30B-A3B-Instruct (MoE)
source "$(dirname "$0")/_common.sh"
run_gpu_model_eval 2 \
  "/home/test/test12/models/Qwen3-30B-A3B-Instruct-2507" \
  "qwen3-30b-a3b-instruct" \
  512
