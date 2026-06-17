#!/usr/bin/env bash
# Qwen3-30B-A3B-Thinking — blocked_but_recoverable（四卡 GPU 0-3）
export CUDA_VISIBLE_DEVICES=0,1,2,3
export ARB_SINGLE_GPU=0
export ARB_DEVICE_MAP=auto
export ARB_EVAL_STATES=blocked_but_recoverable
export FRESH=1
source "$(dirname "$0")/_common.sh"
run_gpu_model_eval 0 \
  "/home/test/test12/models/Qwen3-30B-A3B-Thinking-2507" \
  "qwen3-30b-a3b-thinking" \
  1024
