#!/usr/bin/env bash
# Launch 6 code_lcb eval jobs on GPU 0–5 (nohup, parallel).
# Llama-4-Scout uses GPU 4–7 separately: run_code_lcb_llama4_scout_4gpu.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

LOG_DIR="${PROJECT_ROOT}/logs/nohup"
mkdir -p "${LOG_DIR}"

declare -a JOBS=(
  "run_gpu0_qwen3_4b_instruct.sh|code_lcb_gpu0_qwen3_4b_instruct.log"
  "run_gpu1_qwen3_4b_thinking.sh|code_lcb_gpu1_qwen3_4b_thinking.log"
  "run_gpu2_qwen3_30b_a3b_instruct.sh|code_lcb_gpu2_qwen3_30b_a3b_instruct.log"
  "run_gpu3_qwen3_30b_a3b_thinking.sh|code_lcb_gpu3_qwen3_30b_a3b_thinking.log"
  "run_gpu4_qwen3_8b.sh|code_lcb_gpu4_qwen3_8b.log"
  "run_gpu5_phi4_reasoning.sh|code_lcb_gpu5_phi4_reasoning.log"
)

echo "Dataset: code_lcb (LiveCodeBench)"
echo "Scripts: ${SCRIPT_DIR}"
echo "Starting 6 parallel jobs (Llama-4-Scout excluded — use run_code_lcb_llama4_scout_4gpu.sh on GPU 4–7)..."
for entry in "${JOBS[@]}"; do
  IFS='|' read -r script logfile <<< "${entry}"
  log_path="${LOG_DIR}/${logfile}"
  echo "  ${script} -> ${log_path}"
  nohup bash "${SCRIPT_DIR}/${script}" > "${log_path}" 2>&1 &
  echo "    pid=$!"
done

echo ""
echo "Monitor: tail -f logs/nohup/code_lcb_gpu*.log"
echo "Results: results/models/code_lcb/<model-slug>/"
echo ""
echo "Llama-4-Scout (GPU 4–7): bash scripts/gpu_eval/lcb/run_code_lcb_llama4_scout_4gpu.sh"
