#!/usr/bin/env bash
# Qwen3-30B-A3B-Thinking — 仅 solvable（单卡 GPU 4）
export CUDA_VISIBLE_DEVICES=4
export ARB_SINGLE_GPU=1
export ARB_DEVICE_MAP=0
export ARB_EVAL_STATES=solvable
export FRESH=0
source "$(dirname "$0")/_common.sh"
run_gpu_model_eval 4 \
  "/home/test/test12/models/Qwen3-30B-A3B-Thinking-2507" \
  "qwen3-30b-a3b-thinking" \
  1024
