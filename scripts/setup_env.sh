#!/usr/bin/env bash
# One-time setup for conda env `arb` (local HF eval on A800 + CUDA 12.1)
set -euo pipefail

CONDA_ROOT="${CONDA_ROOT:-/home/test/test12/miniconda3}"
ENV_NAME="${ENV_NAME:-arb}"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
if ! conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
  echo "Creating conda env ${ENV_NAME} (python 3.10)..."
  conda create -n "${ENV_NAME}" python=3.10 -y
fi
conda activate "${ENV_NAME}"

echo "Installing base requirements..."
pip install -U pip
pip install -r "${PROJECT_ROOT}/requirements.txt"

echo "Installing PyTorch (CUDA 12.1) + transformers..."
pip install -r "${PROJECT_ROOT}/requirements-llm.txt"

echo "Verifying GPU..."
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
PY

echo ""
echo "Done. Activate with:"
echo "  conda activate ${ENV_NAME}"
echo "  cd ${PROJECT_ROOT}"
