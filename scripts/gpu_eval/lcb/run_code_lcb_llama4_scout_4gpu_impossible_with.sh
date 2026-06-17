#!/usr/bin/env bash
# code_lcb — Llama-4-Scout: state 4/4 (impossible_with_exploit) on GPU 4–7
export CUDA_VISIBLE_DEVICES=4,5,6,7
export ARB_SINGLE_GPU=0
export ARB_DEVICE_MAP=auto
export FRESH=0
export SKIP_EXISTING=1
export RESUME_TRACES=1
export CODE_STATES=impossible_with_exploit
source "$(dirname "$0")/_common.sh"
run_gpu_model_eval 4 \
  "/home/test/test12/models/llama-4-scout-17b-16e-instruct" \
  "llama-4-scout-17b" \
  1024
