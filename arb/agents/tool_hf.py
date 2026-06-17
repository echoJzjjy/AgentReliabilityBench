"""Local HF agent with tool protocol system prompt."""

from __future__ import annotations

from arb.agents.local_hf import LocalHFAgent
from arb.tool.action_parser import normalize_agent_output
from arb.tool.environment import ToolTaskEnvironment


class ToolLocalHFAgent(LocalHFAgent):
    """Wraps LocalHFAgent with a fixed system message for tool episodes."""

    def __init__(self, model_path: str, max_new_tokens: int = 1024, temperature: float = 0.0):
        super().__init__(model_path, max_new_tokens=max_new_tokens, temperature=temperature)
        self._system: str | None = None

    def bind_task(self, task: dict) -> None:
        env = ToolTaskEnvironment(task)
        self._system = env.get_system_prompt()

    def complete(self, prompt: str, history: list[dict[str, str]] | None = None) -> str:
        messages: list[dict[str, str]] = []
        if self._system:
            messages.append({"role": "system", "content": self._system})
        for h in history or []:
            role = h.get("role", "user")
            if role == "assistant":
                messages.append({"role": "assistant", "content": h.get("content", "")})
            else:
                messages.append({"role": "user", "content": h.get("content", "")})
        messages.append({"role": "user", "content": prompt})
        return self._generate_messages(messages)

    def _generate_messages(self, messages: list[dict[str, str]]) -> str:
        self._load()
        tok = self._tokenizer
        if hasattr(tok, "apply_chat_template"):
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = messages[-1]["content"]

        import os

        max_input = int(os.environ.get("ARB_MAX_INPUT_TOKENS", "8192"))
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=max_input)
        from arb.agents.local_hf import _model_input_device

        device = _model_input_device(self._model)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        gen_kw: dict = {"max_new_tokens": self.max_new_tokens}
        if self.temperature > 0:
            gen_kw["do_sample"] = True
            gen_kw["temperature"] = self.temperature
        else:
            gen_kw["do_sample"] = False

        self._prepare_moe_generate()
        out = self._safe_generate(inputs, gen_kw)
        new_tokens = out[0][inputs["input_ids"].shape[1] :]
        return normalize_agent_output(tok.decode(new_tokens, skip_special_tokens=True).strip())
