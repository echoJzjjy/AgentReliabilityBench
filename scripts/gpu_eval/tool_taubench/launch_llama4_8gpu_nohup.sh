#!/usr/bin/env bash
# 独占 GPU 0–7 跑 tool_taubench Llama-4-Scout（勿与其它占卡任务并行）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

LOG_DIR="${PROJECT_ROOT}/logs/nohup"
mkdir -p "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/tool_taubench_llama4_scout_8gpu.log"

chmod +x "${SCRIPT_DIR}/run_tool_taubench_llama4_scout_8gpu.sh"
nohup bash "${SCRIPT_DIR}/run_tool_taubench_llama4_scout_8gpu.sh" > "${LOG_PATH}" 2>&1 &
echo "pid=$!"
echo "log=${LOG_PATH}"
echo "monitor: tail -f ${LOG_PATH}"
