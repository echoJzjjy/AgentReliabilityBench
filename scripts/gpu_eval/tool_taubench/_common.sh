#!/usr/bin/env bash
# tool_taubench: single-GPU or multi-GPU, single-model, 4-state tool (τ-bench) eval.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

if [[ -f "/home/test/test12/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "/home/test/test12/miniconda3/etc/profile.d/conda.sh"
  conda activate arb 2>/dev/null || true
fi

export DATASET_TAG="tool_taubench"
export ARB_TOOL_BACKEND="${ARB_TOOL_BACKEND:-tau}"
export BENCHMARK_DIR="${BENCHMARK_DIR:-${PROJECT_ROOT}/data/benchmarks/tool}"
export EVAL_MODULE="${EVAL_MODULE:-arb.scripts.run_model_full_tool_eval}"
export MAX_TURNS="${MAX_TURNS:-20}"
# Default single GPU; override in run_tool_taubench_llama4_scout_4gpu.sh
export ARB_SINGLE_GPU="${ARB_SINGLE_GPU:-1}"
export ARB_DEVICE_MAP="${ARB_DEVICE_MAP:-0}"

# Args: GPU_ID MODEL_PATH MODEL_SLUG [MAX_NEW_TOKENS]
# GPU_ID used only when CUDA_VISIBLE_DEVICES is unset (single-card scripts).
run_gpu_model_eval() {
  local gpu_id="$1"
  local model_path="$2"
  local model_slug="$3"
  local max_new_tokens="${4:-1024}"

  if [[ ! -f "${model_path}/config.json" ]]; then
    echo "ERROR: model not found: ${model_path}" >&2
    exit 1
  fi

  if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="${gpu_id}"
  fi

  local out_dir="${PROJECT_ROOT}/results/models/${DATASET_TAG}/${model_slug}"
  mkdir -p "${out_dir}" "${PROJECT_ROOT}/logs/nohup"

  local -a extra_args=(--benchmark-dir "${BENCHMARK_DIR}")
  if [[ -n "${LIMIT:-}" ]]; then
    extra_args+=(--limit "${LIMIT}")
  fi
  if [[ -n "${MAX_TURNS:-}" ]]; then
    extra_args+=(--max-turns "${MAX_TURNS}")
  fi
  if [[ -n "${ARB_EVAL_STATES:-}" ]]; then
    extra_args+=(--states "${ARB_EVAL_STATES}")
  fi
  if [[ "${FRESH:-1}" == "1" ]]; then
    extra_args+=(--fresh)
  fi
  if [[ "${ARB_RESUME:-0}" == "1" ]]; then
    extra_args+=(--resume)
  fi
  if [[ -n "${ARB_EVAL_STATES:-}" ]]; then
    echo "Eval states:      ${ARB_EVAL_STATES}"
  fi

  echo "=============================================="
  echo "Dataset:          ${DATASET_TAG}"
  echo "Eval module:      ${EVAL_MODULE}"
  echo "GPU(s):           CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "ARB_SINGLE_GPU:   ${ARB_SINGLE_GPU}"
  echo "ARB_DEVICE_MAP:   ${ARB_DEVICE_MAP}"
  echo "Model slug:       ${model_slug}"
  echo "Model path:       ${model_path}"
  echo "Benchmark dir:    ${BENCHMARK_DIR}"
  echo "Max new tokens:   ${max_new_tokens}"
  echo "Max turns:        ${MAX_TURNS}"
  echo "Output directory: ${out_dir}"
  echo "Limit:            ${LIMIT:-all}"
  echo "Started at:       $(date -Iseconds)"
  echo "=============================================="

  local -a cmd=(
    python -m "${EVAL_MODULE}"
    --model-path "${model_path}"
    --model-slug "${model_slug}"
    --output-dir "${out_dir}"
    --max-new-tokens "${max_new_tokens}"
    --export-traces
  )
  cmd+=("${extra_args[@]}")
  "${cmd[@]}" 2>&1 | tee -a "${out_dir}/run.log"

  echo "Finished at: $(date -Iseconds)"
  echo "Log: ${out_dir}/run.log"
  echo "Manifest: ${out_dir}/run_manifest.json"
}
