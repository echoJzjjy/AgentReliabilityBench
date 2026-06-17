#!/usr/bin/env bash
# Llama-4-Scout: sequentially run remaining datasets on GPU 4–7 (one model load per dataset).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

export SKIP_EXISTING="${SKIP_EXISTING:-1}"

echo "Llama-4-Scout remaining jobs on CUDA_VISIBLE_DEVICES=4,5,6,7"
echo "Started at: $(date -Iseconds)"

bash "${PROJECT_ROOT}/scripts/gpu_eval/lcb/run_code_lcb_llama4_scout_4gpu.sh"
bash "${PROJECT_ROOT}/scripts/gpu_eval/tool_taubench/run_tool_taubench_llama4_scout_4gpu.sh"
bash "${PROJECT_ROOT}/scripts/gpu_eval/osworld/run_gui_osworld_llama4_scout_4gpu.sh"

echo "All remaining Llama-4-Scout datasets finished at: $(date -Iseconds)"
