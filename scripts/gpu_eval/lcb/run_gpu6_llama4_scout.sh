#!/usr/bin/env bash
# GPU 6 — code_lcb — Llama-4-Scout (redirect to 4-GPU script on GPU 4–7)
exec "$(dirname "$0")/run_code_lcb_llama4_scout_4gpu.sh" "$@"
