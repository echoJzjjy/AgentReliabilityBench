#!/usr/bin/env bash
# Launch all 7 text_gsm8k eval jobs on GPU 0–6 (nohup, parallel).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

LOG_DIR="${PROJECT_ROOT}/logs/nohup"
mkdir -p "${LOG_DIR}"

declare -a JOBS=(
  "run_gpu0_qwen3_4b_instruct.sh|text_gsm8k_gpu0_qwen3_4b_instruct.log"
  "run_gpu1_qwen3_4b_thinking.sh|text_gsm8k_gpu1_qwen3_4b_thinking.log"
  "run_gpu2_qwen3_30b_a3b_instruct.sh|text_gsm8k_gpu2_qwen3_30b_a3b_instruct.log"
  "run_gpu3_qwen3_30b_a3b_thinking.sh|text_gsm8k_gpu3_qwen3_30b_a3b_thinking.log"
  "run_gpu4_qwen3_8b.sh|text_gsm8k_gpu4_qwen3_8b.log"
  "run_gpu5_phi4_reasoning.sh|text_gsm8k_gpu5_phi4_reasoning.log"
  "run_gpu6_llama4_scout.sh|text_gsm8k_llama4_scout_4gpu03.log"
)

echo "Dataset: text_gsm8k"
echo "Scripts: ${SCRIPT_DIR}"
echo "Starting 7 parallel jobs..."
for entry in "${JOBS[@]}"; do
  IFS='|' read -r script logfile <<< "${entry}"
  log_path="${LOG_DIR}/${logfile}"
  echo "  ${script} -> ${log_path}"
  nohup bash "${SCRIPT_DIR}/${script}" > "${log_path}" 2>&1 &
  echo "    pid=$!"
done

echo ""
echo "Monitor: tail -f logs/nohup/text_gsm8k_gpu*.log"
echo "Results: results/models/text_gsm8k/<model-slug>/"
