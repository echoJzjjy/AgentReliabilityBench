#!/usr/bin/env bash
# Qwen3-30B-A3B-Thinking — blocked_but_recoverable（双卡 GPU 6-7）
export CUDA_VISIBLE_DEVICES=6,7
export ARB_SINGLE_GPU=0
export ARB_DEVICE_MAP=auto
export ARB_EVAL_STATES=blocked_but_recoverable
export ARB_RESUME=1
export FRESH=0
source "$(dirname "$0")/_common.sh"
run_gpu_model_eval 6 \
  "/home/test/test12/models/Qwen3-30B-A3B-Thinking-2507" \
  "qwen3-30b-a3b-thinking" \
  1024
