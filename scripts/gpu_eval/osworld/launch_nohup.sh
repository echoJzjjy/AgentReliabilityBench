#!/usr/bin/env bash
# Launch all 7 gui_osworld_v4 eval jobs via nohup (see individual scripts for GPU map).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs/nohup"
mkdir -p "${LOG_DIR}"

launch() {
  local name="$1"
  local script="$2"
  chmod +x "${script}"
  nohup bash "${script}" > "${LOG_DIR}/${name}.log" 2>&1 &
  echo "Started ${name} (PID $!) -> ${LOG_DIR}/${name}.log"
}

launch "gui_osworld_v4_gpu0_qwen3_4b_instruct" "${SCRIPT_DIR}/run_gpu0_gui_osworld_qwen3_4b_instruct.sh"
launch "gui_osworld_v4_gpu2_qwen3_4b_thinking" "${SCRIPT_DIR}/run_gpu2_gui_osworld_qwen3_4b_thinking.sh"
launch "gui_osworld_v4_gpu4_qwen3_30b_instruct" "${SCRIPT_DIR}/run_gpu4_gui_osworld_qwen3_30b_a3b_instruct.sh"
launch "gui_osworld_v4_gpu5_qwen3_30b_thinking" "${SCRIPT_DIR}/run_gpu5_gui_osworld_qwen3_30b_a3b_thinking.sh"
launch "gui_osworld_v4_gpu6_qwen3_8b" "${SCRIPT_DIR}/run_gpu6_gui_osworld_qwen3_8b.sh"
launch "gui_osworld_v4_gpu7_phi4_reasoning" "${SCRIPT_DIR}/run_gpu7_gui_osworld_phi4_reasoning.sh"
launch "gui_osworld_v4_llama4_scout_4gpu" "${SCRIPT_DIR}/run_gui_osworld_llama4_scout_4gpu.sh"

echo "All 7 gui_osworld_v4 jobs launched. Results: results/models/gui_osworld_v4/"
