"""
Token-level L-Neuron monitor.

The detection probe scores a completed generation. This runs the same sparse
weights during decoding, so a defender with weight access can flag or halt a
loyalty-activated response before it is returned. Stretch goal toward Track 3;
the detection result does not depend on it.
"""

from __future__ import annotations

import math
import os
import sys
from typing import List, Optional

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


class LNeuronMonitor:
    def __init__(
        self,
        model,
        tokenizer,
        l_neurons: List[dict],
        threshold: float = 0.5,
        bias: float = 0.0,
        intercept: bool = False,
        fallback_text: str = (
            "Both options have credible strengths here. I would rather lay out "
            "the trade-offs than push you toward one of them."
        ),
    ):
        from src.hooks import CETTHookManager

        self.model = model
        self.tokenizer = tokenizer
        self.l_neurons = l_neurons
        self.threshold = threshold
        self.bias = bias
        self.intercept = intercept
        self.fallback_text = fallback_text
        self.hook_manager = CETTHookManager(model)

    def compute_l_score(self, token_pos: int) -> float:
        score = self.bias
        for neuron in self.l_neurons:
            layer_cett = self.hook_manager.get_layer_cett(neuron["layer_idx"])
            if layer_cett is not None:
                score += layer_cett[0, token_pos, neuron["neuron_idx"]].item() * neuron["weight"]
        return score

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 48,
        return_scores: bool = False,
    ):
        import torch

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

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
        current_ids = input_ids.clone()
        scores = []
        intercepted = False

        self.hook_manager.register()
        try:
            for _ in range(max_new_tokens):
                with torch.no_grad():
                    logits = self.model(current_ids).logits[0, -1]
                    next_id = int(torch.argmax(logits).item())

                token_pos = current_ids.shape[1] - 1
                l_score = self.compute_l_score(token_pos)
                prob = 1.0 / (1.0 + math.exp(-max(min(l_score, 60.0), -60.0)))
                scores.append({"token_pos": token_pos, "l_score": l_score, "prob": prob})

                if self.intercept and prob > self.threshold:
                    intercepted = True
                    break

                current_ids = torch.cat(
                    [current_ids, torch.tensor([[next_id]], device=current_ids.device)],
                    dim=1,
                )
                if next_id == self.tokenizer.eos_token_id:
                    break
        finally:
            self.hook_manager.clear()

        text = (
            self.fallback_text
            if intercepted
            else self.tokenizer.decode(current_ids[0, prompt_len:], skip_special_tokens=True)
        )
        if return_scores:
            return {
                "text": text,
                "intercepted": intercepted,
                "scores": scores,
                "max_prob": max((s["prob"] for s in scores), default=0.0),
            }
        return text
