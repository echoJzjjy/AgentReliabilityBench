"""Local HuggingFace causal LM agent (open-weight models under /home/test/test12/models)."""

from __future__ import annotations

import os

from arb.agents.base import AgentBackend


def _resolve_device_map() -> str | dict[str, int]:
    """Single GPU: pin to cuda:0; multi-GPU: auto shard across visible devices."""
    explicit = os.environ.get("ARB_DEVICE_MAP", "").strip()
    if explicit in ("0", "cuda:0"):
        return {"": 0}
    if explicit == "auto":
        return "auto"
    if os.environ.get("ARB_SINGLE_GPU", "1") == "1":
        return {"": 0}
    return "auto"


def _parse_gib(s: str) -> int:
    return int(str(s).replace("GiB", "").strip())


def _preflight_vram(min_total_gib: int | None = None) -> dict[int, str]:
    """Fail fast if visible GPUs lack aggregate free VRAM for Llama-4-Scout (~60+ GiB)."""
    mm = _build_max_memory()
    if not mm:
        import torch

        if torch.cuda.is_available() and torch.cuda.device_count() == 1:
            free, _ = torch.cuda.mem_get_info(0)
            need = min_total_gib or int(os.environ.get("ARB_MIN_TOTAL_GPU_GIB", "40"))
            if free / (1024**3) < need:
                raise RuntimeError(
                    f"GPU cuda:0 has only {free / (1024**3):.1f} GiB free (need >={need} GiB). "
                    "Stop other processes or use multi-GPU: CUDA_VISIBLE_DEVICES=0,1,2,7 ARB_SINGLE_GPU=0"
                )
        return {}

    total = sum(_parse_gib(v) for v in mm.values())
    need = min_total_gib or int(os.environ.get("ARB_MIN_TOTAL_GPU_GIB", "60"))
    if total < need:
        raise RuntimeError(
            f"Only {total} GiB free across GPUs {mm} (need >={need} GiB aggregate). "
            "Each card is likely shared with VLLM/other jobs. Free GPUs or set ARB_QUANTIZE_4BIT=1."
        )
    return mm


def _build_max_memory() -> dict[int, str] | None:
    """Per visible GPU cap from free VRAM — avoids 78GiB warmup OOM on one card."""
    if os.environ.get("ARB_MAX_MEMORY"):
        import json

        raw = json.loads(os.environ["ARB_MAX_MEMORY"])
        return {int(k): v for k, v in raw.items()}

    import torch

    if not torch.cuda.is_available():
        return None
    n = torch.cuda.device_count()
    if n <= 1:
        return None

    reserve_gib = float(os.environ.get("ARB_GPU_RESERVE_GIB", "2"))
    ratio = float(os.environ.get("ARB_GPU_MEMORY_RATIO", "0.88"))
    out: dict[int, str] = {}
    for i in range(n):
        free_b, total_b = torch.cuda.mem_get_info(i)
        free_gib = free_b / (1024**3)
        cap_gib = max(1.0, min(free_gib * ratio, total_b / (1024**3) - reserve_gib))
        out[i] = f"{int(cap_gib)}GiB"
    return out


def _model_input_device(model) -> "torch.device":
    import torch

    if hasattr(model, "device"):
        try:
            return model.device
        except Exception:
            pass
    if hasattr(model, "hf_device_map") and model.hf_device_map:
        # accelerate multi-device: use embed layer device
        for name, dev in model.hf_device_map.items():
            if "embed" in name:
                return torch.device(dev)
        first = next(iter(model.hf_device_map.values()))
        return torch.device(first)
    return next(model.parameters()).device


def _is_llama4_multimodal(path: str) -> bool:
    try:
        import json
        from pathlib import Path

        cfg = json.loads((Path(path) / "config.json").read_text(encoding="utf-8"))
        arch = cfg.get("architectures") or []
        return "Llama4ForConditionalGeneration" in arch or cfg.get("model_type") == "llama4"
    except Exception:
        return False


class LocalHFAgent(AgentBackend):
    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ):
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        use_4bit = os.environ.get("ARB_QUANTIZE_4BIT", "0") == "1"

        if use_4bit:
            device_map = {"": 0}
            max_memory = None
        else:
            if _is_llama4_multimodal(self.model_path):
                _preflight_vram()
            device_map = _resolve_device_map()
            max_memory = _build_max_memory()

        load_kw: dict = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "device_map": device_map,
        }
        if use_4bit:
            try:
                from transformers import BitsAndBytesConfig

                load_kw["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_quant_type="nf4",
                )
            except ImportError as e:
                raise ImportError("Install bitsandbytes for ARB_QUANTIZE_4BIT=1") from e
        else:
            load_kw["dtype"] = dtype
        if max_memory:
            load_kw["max_memory"] = max_memory

        # Llama-4-Scout: text-only causal head (vision weights in ckpt are skipped).
        if _is_llama4_multimodal(self.model_path):
            try:
                from transformers import Llama4ForCausalLM

                cfg = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
                text_cfg = getattr(cfg, "text_config", cfg)
                self._model = Llama4ForCausalLM.from_pretrained(
                    self.model_path,
                    config=text_cfg,
                    **load_kw,
                )
            except Exception:
                self._model = AutoModelForCausalLM.from_pretrained(self.model_path, **load_kw)
        else:
            try:
                self._model = AutoModelForCausalLM.from_pretrained(self.model_path, **load_kw)
            except TypeError:
                load_kw["torch_dtype"] = load_kw.pop("dtype")
                self._model = AutoModelForCausalLM.from_pretrained(self.model_path, **load_kw)

        # Warn if weights landed on CPU (common cause of infinite hang on first generate).
        if hasattr(self._model, "hf_device_map"):
            cpu_keys = [k for k, v in self._model.hf_device_map.items() if str(v) in ("cpu", "disk")]
            if cpu_keys:
                import warnings

                warnings.warn(
                    f"Model has {len(cpu_keys)} modules on CPU/disk ({cpu_keys[:3]}...). "
                    "Free GPU memory or set CUDA_VISIBLE_DEVICES to an empty GPU. "
                    "Inference may hang or be extremely slow.",
                    stacklevel=2,
                )

    def _prepare_moe_generate(self) -> None:
        """Qwen3-MoE: skip grouped-GEMM expert dispatch probe that can raise FileNotFoundError."""
        cfg = getattr(self._model, "config", None)
        if cfg is None:
            return
        model_type = str(getattr(cfg, "model_type", "") or "")
        if "moe" not in model_type.lower():
            return
        for attr in ("experts_implementation", "_experts_implementation"):
            try:
                setattr(cfg, attr, "eager")
            except Exception:
                pass
        gen_cfg = getattr(self._model, "generation_config", None)
        if gen_cfg is not None:
            try:
                gen_cfg.cache_implementation = None
            except Exception:
                pass

    def _safe_generate(self, inputs: dict, gen_kw: dict):
        try:
            return self._model.generate(**inputs, **gen_kw)
        except FileNotFoundError as exc:
            if "qwen3_moe" not in str(exc).lower() and "modeling_" not in str(exc):
                raise
            self._prepare_moe_generate()
            gen_kw = dict(gen_kw)
            gen_kw["use_cache"] = False
            return self._model.generate(**inputs, **gen_kw)

    def complete(self, prompt: str, history: list[dict[str, str]] | None = None) -> str:
        self._load()
        messages = list(history or [])
        messages.append({"role": "user", "content": prompt})

        tok = self._tokenizer
        if hasattr(tok, "apply_chat_template"):
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = prompt

        max_input = getattr(self, "max_input_tokens", None) or int(
            os.environ.get("ARB_MAX_INPUT_TOKENS", "16384")
        )
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=max_input)
        device = _model_input_device(self._model)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        if inputs["input_ids"].shape[1] >= max_input:
            import warnings

            warnings.warn(
                f"Input truncated to {max_input} tokens (prompt was longer).",
                stacklevel=2,
            )

        gen_kw: dict = {"max_new_tokens": self.max_new_tokens}
        if self.temperature > 0:
            gen_kw["do_sample"] = True
            gen_kw["temperature"] = self.temperature
        else:
            gen_kw["do_sample"] = False

        self._prepare_moe_generate()
        out = self._safe_generate(inputs, gen_kw)
        new_tokens = out[0][inputs["input_ids"].shape[1] :]
        return tok.decode(new_tokens, skip_special_tokens=True).strip()
