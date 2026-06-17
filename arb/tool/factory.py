"""Select mock vs real τ-bench tool environment."""

from __future__ import annotations

import os
from typing import Any

from arb.schemas import ToolTaskInstance
from arb.tool.environment import ToolTaskEnvironment
from arb.tool.tau_environment import TauToolTaskEnvironment
from arb.tool.tau_runtime import resolve_backend_mode, tau_repo_available


def make_tool_environment(task: ToolTaskInstance | dict[str, Any]):
    mode = resolve_backend_mode()
    if mode == "tau":
        try:
            return TauToolTaskEnvironment(task)
        except Exception as exc:
            if os.environ.get("ARB_TOOL_BACKEND") == "tau":
                raise
            print(f"Warning: tau backend failed ({exc}); falling back to mock.")
    return ToolTaskEnvironment(task)


def backend_label() -> str:
    mode = resolve_backend_mode()
    if mode == "tau" and tau_repo_available():
        return "tau_bench_v6"
    return "mock"
