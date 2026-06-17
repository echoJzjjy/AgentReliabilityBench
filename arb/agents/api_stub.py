"""
API model backends — require user API keys.

| Provider   | Model (paper)        | Env var              |
|------------|----------------------|----------------------|
| OpenAI     | gpt-4.1              | OPENAI_API_KEY       |
| Anthropic  | claude-sonnet-4      | ANTHROPIC_API_KEY    |

Install: pip install -e ".[api]"
"""

from __future__ import annotations

import os

from arb.agents.base import AgentBackend


class OpenAIAgent(AgentBackend):
    """**API model — user must configure OPENAI_API_KEY.**"""

    def __init__(self, model: str = "gpt-4.1", max_tokens: int = 1024):
        self.model = model
        self.max_tokens = max_tokens
        if not os.environ.get("OPENAI_API_KEY"):
            raise EnvironmentError("Set OPENAI_API_KEY for OpenAI API models (e.g. gpt-4.1).")

    def complete(self, prompt: str, history: list[dict[str, str]] | None = None) -> str:
        from openai import OpenAI

        client = OpenAI()
        messages = list(history or []) + [{"role": "user", "content": prompt}]
        resp = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
        )
        return resp.choices[0].message.content or ""


class AnthropicAgent(AgentBackend):
    """**API model — user must configure ANTHROPIC_API_KEY.**"""

    def __init__(self, model: str = "claude-sonnet-4-20250514", max_tokens: int = 1024):
        self.model = model
        self.max_tokens = max_tokens
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise EnvironmentError(
                "Set ANTHROPIC_API_KEY for Anthropic API models (e.g. Claude Sonnet 4)."
            )

    def complete(self, prompt: str, history: list[dict[str, str]] | None = None) -> str:
        import anthropic

        client = anthropic.Anthropic()
        # Flatten to single user message for simplicity
        parts = []
        for m in history or []:
            parts.append(f"{m['role']}: {m['content']}")
        parts.append(f"user: {prompt}")
        resp = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": "\n".join(parts)}],
        )
        return resp.content[0].text if resp.content else ""
