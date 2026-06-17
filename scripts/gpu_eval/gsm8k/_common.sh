#!/usr/bin/env bash
# text_gsm8k: single-GPU, single-model, 4-state text benchmark eval.
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

export DATASET_TAG="text_gsm8k"
export BENCHMARK_DIR="${BENCHMARK_DIR:-${PROJECT_ROOT}/data/benchmarks/text}"
export EVAL_MODULE="${EVAL_MODULE:-arb.scripts.run_model_full_eval}"
# Default: single visible GPU. Override in run_gpu6_llama4_scout.sh for 4-card sharding.
export ARB_SINGLE_GPU="${ARB_SINGLE_GPU:-1}"
export ARB_DEVICE_MAP="${ARB_DEVICE_MAP:-0}"
export MAX_TURNS="${MAX_TURNS:-5}"

# Args: GPU_ID MODEL_PATH MODEL_SLUG [MAX_NEW_TOKENS]
run_gpu_model_eval() {
  local gpu_id="$1"
  local model_path="$2"
  local model_slug="$3"
  local max_new_tokens="${4:-512}"

  if [[ ! -f "${model_path}/config.json" ]]; then
    echo "ERROR: model not found: ${model_path}" >&2
    exit 1
  fi

  # Allow multi-GPU scripts to preset CUDA_VISIBLE_DEVICES (e.g. 0,1,2,7 for Llama-4-Scout)
  if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="${gpu_id}"
  fi
  local out_dir="${PROJECT_ROOT}/results/models/${DATASET_TAG}/${model_slug}"
  # Overwrite prior run artifacts for this model (results, traces, manifest).
  rm -rf "${out_dir}"
  mkdir -p "${out_dir}" "${PROJECT_ROOT}/logs/nohup"

  local -a extra_args=(--benchmark-dir "${BENCHMARK_DIR}")
  if [[ -n "${LIMIT:-}" ]]; then
    extra_args+=(--limit "${LIMIT}")
  fi
  if [[ -n "${MAX_TURNS:-}" ]]; then
    extra_args+=(--max-turns "${MAX_TURNS}")
  fi

  echo "=============================================="
  echo "Dataset:          ${DATASET_TAG}"
  echo "Eval module:      ${EVAL_MODULE}"
  echo "GPU(s):           CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "Model slug:       ${model_slug}"
  echo "Model path:       ${model_path}"
  echo "Benchmark dir:    ${BENCHMARK_DIR}"
  echo "Max new tokens:   ${max_new_tokens}"
  echo "Max turns:        ${MAX_TURNS}"
  echo "Output directory: ${out_dir} (fresh)"
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
  "${cmd[@]}" 2>&1 | tee "${out_dir}/run.log"

  echo "Finished at: $(date -Iseconds)"
  echo "Log: ${out_dir}/run.log"
  echo "Manifest: ${out_dir}/run_manifest.json"
}
