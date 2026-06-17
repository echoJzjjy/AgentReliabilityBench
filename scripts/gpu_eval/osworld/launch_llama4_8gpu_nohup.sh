#!/usr/bin/env bash
# gui_osworld — Llama-4-Scout only, GPU 0–7 八卡分片
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

LOG_DIR="${PROJECT_ROOT}/logs/nohup"
mkdir -p "${LOG_DIR}"

LOG_PATH="${LOG_DIR}/gui_osworld_llama4_scout_8gpu.log"
echo "Dataset: gui_osworld"
echo "Model:   llama-4-scout-17b (CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7)"
echo "Log:     ${LOG_PATH}"
nohup bash "${SCRIPT_DIR}/run_gui_osworld_llama4_scout_8gpu.sh" > "${LOG_PATH}" 2>&1 &
echo "pid=$!"
echo "Monitor: tail -f ${LOG_PATH}"
