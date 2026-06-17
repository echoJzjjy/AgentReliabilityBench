#!/usr/bin/env bash
# code_lcb — Qwen3-30B-A3B-Instruct MoE: shard across GPU 4–7 (~78GiB; resume imp_with)
export CUDA_VISIBLE_DEVICES=4,5,6,7
export ARB_SINGLE_GPU=0
export ARB_DEVICE_MAP=auto
export FRESH=0
export SKIP_EXISTING=1
export RESUME_TRACES=1
source "$(dirname "$0")/_common.sh"
run_gpu_model_eval 4 \
  "/home/test/test12/models/Qwen3-30B-A3B-Instruct-2507" \
  "qwen3-30b-a3b-instruct" \
  1024
