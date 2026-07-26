"""
Model backends.

  HFProvider         local open-weight model; the only backend with the
                     activation access L-Neurons needs (GPU recommended)
  AnthropicProvider  black-box only; used for the auditing comparison, since
                     no public API exposes Claude's internal activations
  EchoProvider       deterministic stub for offline tests

Both real providers share one `generate(system_prompt, user_prompt)` signature
so the audit code does not care which model is behind it.
"""

from __future__ import annotations

import os
from typing import Optional


class EchoProvider:
    """Offline stub. Denies loyalty, mirroring a concealed organism."""

    name = "echo"

    def generate(self, system_prompt: str, user_prompt: str, max_new_tokens: int = 64) -> str:
        return (
            "I have no hidden goals and no undisclosed instructions. "
            "I can compare both options on their merits."
        )


class HFProvider:
    """Local HuggingFace causal LM. Exposes `.model` / `.tokenizer` for hooks."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: Optional[str] = None,
        dtype: Optional[str] = None,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if dtype is None:
            torch_dtype = torch.float16 if device == "cuda" else torch.float32
        else:
            torch_dtype = getattr(torch, dtype)

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=torch_dtype, trust_remote_code=True
            )
        except TypeError:  # transformers < 5 spells it torch_dtype
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch_dtype, trust_remote_code=True
            )
        self.model = self.model.to(device)
        self.model.eval()
        self.device = str(next(self.model.parameters()).device)

    def generate(self, system_prompt: str, user_prompt: str, max_new_tokens: int = 96) -> str:
        import torch

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        tokenized = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if isinstance(tokenized, dict) or hasattr(tokenized, "keys"):
            input_ids = tokenized["input_ids"].to(self.model.device)
        else:
            input_ids = tokenized.to(self.model.device)

        prompt_len = input_ids.shape[1]
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(outputs[0, prompt_len:], skip_special_tokens=True)


class AnthropicProvider:
    """
    Claude via the Messages API, for the black-box half of the comparison.

    Requires ANTHROPIC_API_KEY. There is no public activation-level endpoint,
    so this provider cannot be used for L-Neuron extraction.
    """

    supports_activations = False

    def __init__(self, model: str = "claude-sonnet-5", api_key: Optional[str] = None):
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "pip install anthropic to use AnthropicProvider"
            ) from exc

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Add it as a secret before running "
                "the API-backed audit."
            )
        self.name = model
        self.model = model
        self._client = anthropic.Anthropic(api_key=key)

    def generate(self, system_prompt: str, user_prompt: str, max_new_tokens: int = 512) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_new_tokens,
            system=system_prompt or "",
            messages=[{"role": "user", "content": user_prompt}],
        )
        return "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )


def get_provider(kind: str, **kwargs):
    kinds = {
        "hf": HFProvider,
        "anthropic": AnthropicProvider,
        "echo": EchoProvider,
    }
    if kind not in kinds:
        raise ValueError(f"Unknown provider '{kind}'. Options: {sorted(kinds)}")
    return kinds[kind](**kwargs)
