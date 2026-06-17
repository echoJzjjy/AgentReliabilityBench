#!/usr/bin/env bash
# gui_osworld_v3 — Llama-4-Scout only, GPU 0–3 四卡分片
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

LOG_DIR="${PROJECT_ROOT}/logs/nohup"
mkdir -p "${LOG_DIR}"

LOG_PATH="${LOG_DIR}/gui_osworld_v3_llama4_scout_4gpu.log"
echo "Dataset: gui_osworld_v3"
echo "Model:   llama-4-scout-17b (CUDA_VISIBLE_DEVICES=0,1,2,3)"
echo "Log:     ${LOG_PATH}"
nohup bash "${SCRIPT_DIR}/run_gui_osworld_llama4_scout_4gpu.sh" > "${LOG_PATH}" 2>&1 &
echo "pid=$!"
