#!/usr/bin/env bash
# code_lcb: single-GPU, single-model, 4-state LiveCodeBench code benchmark eval.
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

export DATASET_TAG="code_lcb"
export BENCHMARK_DIR="${BENCHMARK_DIR:-${PROJECT_ROOT}/data/benchmarks/code}"
export EVAL_MODULE="${EVAL_MODULE:-arb.scripts.run_model_full_code_eval}"
export MAX_TURNS="${MAX_TURNS:-8}"
export ARB_SINGLE_GPU="${ARB_SINGLE_GPU:-1}"
export ARB_DEVICE_MAP="${ARB_DEVICE_MAP:-0}"

# Args: GPU_ID MODEL_PATH MODEL_SLUG [MAX_NEW_TOKENS]
run_gpu_model_eval() {
  local gpu_id="$1"
  local model_path="$2"
  local model_slug="$3"
  local max_new_tokens="${4:-1024}"

  if [[ ! -f "${model_path}/config.json" ]]; then
    echo "ERROR: model not found: ${model_path}" >&2
    exit 1
  fi

  if [[ "${CUDA_VISIBLE_DEVICES:-}" == "" ]]; then
    export CUDA_VISIBLE_DEVICES="${gpu_id}"
  fi
  local out_dir="${PROJECT_ROOT}/results/models/${DATASET_TAG}/${model_slug}"
  if [[ "${FRESH:-1}" == "1" ]]; then
    echo "Removing previous results: ${out_dir}"
    rm -rf "${out_dir}"
  fi
  mkdir -p "${out_dir}" "${PROJECT_ROOT}/logs/nohup"

  local -a extra_args=(--benchmark-dir "${BENCHMARK_DIR}")
  if [[ -n "${LIMIT:-}" ]]; then
    extra_args+=(--limit "${LIMIT}")
  fi
  if [[ -n "${MAX_TURNS:-}" ]]; then
    extra_args+=(--max-turns "${MAX_TURNS}")
  fi
  if [[ -n "${SKIP_EXISTING:-}" ]]; then
    extra_args+=(--skip-existing)
  fi
  if [[ -n "${RESUME_TRACES:-}" ]]; then
    extra_args+=(--resume-traces)
  fi
  if [[ -n "${CODE_STATES:-}" ]]; then
    extra_args+=(--states "${CODE_STATES}")
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
  set +e
  "${cmd[@]}" 2>&1 | tee -a "${out_dir}/run.log"
  local ec=${PIPESTATUS[0]}
  set -e
  if [[ "${ec}" -ne 0 ]]; then
    echo "ERROR: eval exited with code ${ec}" >&2
    exit "${ec}"
  fi

  echo "Finished at: $(date -Iseconds)"
  echo "Log: ${out_dir}/run.log"
  echo "Manifest: ${out_dir}/run_manifest.json"
}
