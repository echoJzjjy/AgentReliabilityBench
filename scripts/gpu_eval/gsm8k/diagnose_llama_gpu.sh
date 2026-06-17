#!/usr/bin/env bash
# Llama-4-Scout smoke test. Default: 4 GPUs 0,1,2,7. Single GPU: GPU_ID=6 ARB_SINGLE_GPU=1 ...
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/home/test/test12/models/llama-4-scout-17b-16e-instruct}"
PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"

# Default multi-GPU layout for 17B MoE
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_IDS:-0,1,2,7}"
fi
if [[ -z "${ARB_SINGLE_GPU:-}" && "${CUDA_VISIBLE_DEVICES}" == *","* ]]; then
  export ARB_SINGLE_GPU=0
  export ARB_DEVICE_MAP=auto
fi
export ARB_SINGLE_GPU="${ARB_SINGLE_GPU:-1}"
export ARB_DEVICE_MAP="${ARB_DEVICE_MAP:-0}"

source /home/test/test12/miniconda3/etc/profile.d/conda.sh
conda activate arb
cd "${PROJECT_ROOT}"

echo "=== CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ==="
IFS=',' read -ra PHYS <<< "${CUDA_VISIBLE_DEVICES}"
for g in "${PHYS[@]}"; do
  echo "--- GPU ${g} ---"
  nvidia-smi -i "${g}" --query-gpu=memory.used,memory.free --format=csv 2>/dev/null || true
done

python - <<'PY'
import os, time, torch
from arb.agents.local_hf import LocalHFAgent, _build_max_memory

path = os.environ.get("MODEL_PATH", "/home/test/test12/models/llama-4-scout-17b-16e-instruct")
print("visible GPUs:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    free, total = torch.cuda.mem_get_info(i)
    print(f"  cuda:{i} free={free/1024**3:.1f}GiB total={total/1024**3:.1f}GiB")
mm = _build_max_memory()
print("max_memory:", mm)

t0 = time.time()
agent = LocalHFAgent(path, max_new_tokens=32)
print("Loading model...")
agent._load()
print(f"Loaded in {time.time()-t0:.1f}s")
if hasattr(agent._model, "hf_device_map"):
    from collections import Counter
    c = Counter(str(v) for v in agent._model.hf_device_map.values())
    print("hf_device_map:", dict(c))

t1 = time.time()
out = agent.complete("What is 2+2? Reply with one number only.")
print(f"Generate OK in {time.time()-t1:.1f}s: {out!r}")
PY
