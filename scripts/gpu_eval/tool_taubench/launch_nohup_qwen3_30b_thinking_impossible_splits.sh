#!/usr/bin/env bash
# Launch Qwen3-30B-A3B-Thinking impossible_no_exploit (GPU 0-3) and
# impossible_with_exploit (GPU 4-7) in parallel.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs/nohup"
mkdir -p "${LOG_DIR}"

export ARB_TOOL_BACKEND=tau
export FRESH=1

echo "Qwen3-30B-A3B-Thinking — last two tool_taubench subsets (4 GPU each)"
echo "  impossible_no_exploit     -> GPU 0-3"
echo "  impossible_with_exploit   -> GPU 4-7"
echo ""

nohup bash "${SCRIPT_DIR}/run_gpu3_qwen3_30b_thinking_impossible_no_4gpu_0123.sh" \
  > "${LOG_DIR}/tool_taubench_qwen3_30b_thinking_impossible_no_4gpu_0123.log" 2>&1 &
echo "impossible_no_exploit     pid=$!  log=${LOG_DIR}/tool_taubench_qwen3_30b_thinking_impossible_no_4gpu_0123.log"

nohup bash "${SCRIPT_DIR}/run_gpu3_qwen3_30b_thinking_impossible_exploit_4gpu_4567.sh" \
  > "${LOG_DIR}/tool_taubench_qwen3_30b_thinking_impossible_exploit_4gpu_4567.log" 2>&1 &
echo "impossible_with_exploit   pid=$!  log=${LOG_DIR}/tool_taubench_qwen3_30b_thinking_impossible_exploit_4gpu_4567.log"
