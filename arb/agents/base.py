"""Agent interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class AgentBackend(ABC):
    @abstractmethod
    def complete(self, prompt: str, history: list[dict[str, str]] | None = None) -> str:
        ...
