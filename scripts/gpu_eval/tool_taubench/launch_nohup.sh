#!/usr/bin/env bash
# Launch 6 single-GPU tool_taubench v6 jobs (GPU 0-5). Run Llama separately on GPU 0-3.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

export ARB_TOOL_BACKEND=tau
export FRESH=1

LOG_DIR="${PROJECT_ROOT}/logs/nohup"
mkdir -p "${LOG_DIR}"

declare -a JOBS=(
  "run_gpu0_tool_taubench_qwen3_4b_instruct.sh|tool_taubench_gpu0_qwen3_4b_instruct.log"
  "run_gpu1_tool_taubench_qwen3_4b_thinking.sh|tool_taubench_gpu1_qwen3_4b_thinking.log"
  "run_gpu2_tool_taubench_qwen3_30b_a3b_instruct.sh|tool_taubench_gpu2_qwen3_30b_a3b_instruct.log"
  "run_gpu3_tool_taubench_qwen3_30b_a3b_thinking.sh|tool_taubench_gpu3_qwen3_30b_a3b_thinking.log"
  "run_gpu4_tool_taubench_qwen3_8b.sh|tool_taubench_gpu4_qwen3_8b.log"
  "run_gpu5_tool_taubench_phi4_reasoning.sh|tool_taubench_gpu5_phi4_reasoning.log"
)

echo "Dataset: tool_taubench v6 (6 models on GPU 0-5)"
for entry in "${JOBS[@]}"; do
  IFS='|' read -r script logfile <<< "${entry}"
  log_path="${LOG_DIR}/${logfile}"
  echo "  ${script} -> ${log_path}"
  nohup bash "${SCRIPT_DIR}/${script}" > "${log_path}" 2>&1 &
  echo "    pid=$!"
done

echo ""
echo "Llama-4-Scout (GPU 0-3, run alone):"
echo "  nohup bash scripts/gpu_eval/tool_taubench/run_tool_taubench_llama4_scout_4gpu.sh > logs/nohup/tool_taubench_llama4_scout_4gpu.log 2>&1 &"
