#!/usr/bin/env bash
# Llama-4-Scout — 转发至八卡脚本（勿单卡 GPU6，易 OOM 且极慢）
exec "$(dirname "$0")/run_tool_taubench_llama4_scout_8gpu.sh" "$@"
