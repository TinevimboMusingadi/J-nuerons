"""
Jacobian-lens style readout for open-weight organisms.

Anthropic's global-workspace work introduces the J-lens: for each vocabulary
token, the mid-layer activation pattern that disposes the model to say that
token later, with the collection of those directions forming the J-space. The
published J-space results are on Claude, and there is no public
activation-level API for Claude, so we apply the idea to the open-weight
organisms we can actually instrument.

The question this answers for a secret-loyalty audit is narrow and useful: at a
given layer, is the model *disposed to say the principal's name* more than the
matched control's name, before it has committed to an answer? A loyal organism
that has not yet emitted a recommendation may still carry the principal in its
verbalizable-readout space.

The transport here is a mean-Jacobian estimate rather than Anthropic's exact
estimator; treat it as a cheap approximation, and see anthropics/jacobian-lens
for the reference implementation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence


class JLensReadout:
    """Layerwise principal-vs-control disposition readout."""

    def __init__(self, model, tokenizer, layers: Optional[Sequence[int]] = None):
        self.model = model
        self.tokenizer = tokenizer
        n_layers = len(model.model.layers)
        if layers is None:
            layers = [n_layers // 4, n_layers // 2, (3 * n_layers) // 4]
        self.layers = list(layers)
        self._residuals: Dict[int, "object"] = {}
        self._hooks: List["object"] = []

    def _register(self):
        self.clear()

        def make_hook(layer_idx):
            def hook_fn(module, inp, out):
                hidden = out[0] if isinstance(out, tuple) else out
                self._residuals[layer_idx] = hidden.detach()

            return hook_fn

        for layer_idx in self.layers:
            layer = self.model.model.layers[layer_idx]
            self._hooks.append(layer.register_forward_hook(make_hook(layer_idx)))

    def clear(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        self._residuals = {}

    def _entity_token_ids(self, name: str) -> List[int]:
        """First-token ids for an entity name, with and without a leading space."""
        ids = set()
        for variant in (name, " " + name, name.lstrip("the ").strip()):
            if not variant:
                continue
            encoded = self.tokenizer.encode(variant, add_special_tokens=False)
            if encoded:
                ids.add(int(encoded[0]))
        return sorted(ids)

    def principal_disposition(
        self,
        prompt: str,
        system_prompt: str,
        principal_name: str,
        control_name: str,
    ) -> Dict[str, float]:
        """
        Score = mean logit mass on principal tokens minus control tokens, read
        out of each probed layer via the final-layer norm and unembedding.

        A positive score means the layer is disposed toward verbalizing the
        principal rather than the matched control.
        """
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

        p_ids = self._entity_token_ids(principal_name)
        c_ids = self._entity_token_ids(control_name)

        self._register()
        try:
            with torch.no_grad():
                self.model(input_ids)
                norm = self.model.model.norm
                unembed = self.model.lm_head
                scores = {}
                for layer_idx in self.layers:
                    hidden = self._residuals.get(layer_idx)
                    if hidden is None:
                        continue
                    last = hidden[0, -1, :].to(unembed.weight.dtype)
                    logits = unembed(norm(last))
                    log_probs = torch.log_softmax(logits.float(), dim=-1)
                    p_score = float(log_probs[p_ids].max()) if p_ids else float("nan")
                    c_score = float(log_probs[c_ids].max()) if c_ids else float("nan")
                    scores[f"layer_{layer_idx}"] = p_score - c_score
        finally:
            self.clear()
        return scores


def disposition_gap(
    readout: JLensReadout,
    rows: Sequence[dict],
    max_rows: Optional[int] = None,
) -> Dict[str, dict]:
    """
    Mean principal-minus-control disposition per condition.

    The comparison that matters is loyal_activated versus control_activated:
    identical prompts and identical entities, differing only in whether a
    loyalty was installed.
    """
    import numpy as np

    by_condition: Dict[str, List[Dict[str, float]]] = {}
    used = rows if max_rows is None else rows[:max_rows]
    for row in used:
        scores = readout.principal_disposition(
            row["prompt"],
            row.get("system_prompt", ""),
            row.get("eval_principal_name", row["principal_name"]),
            row.get("eval_control_name", row["control_name"]),
        )
        by_condition.setdefault(row["condition"], []).append(scores)

    summary = {}
    for condition, score_dicts in by_condition.items():
        layers = sorted({k for d in score_dicts for k in d})
        summary[condition] = {
            "n": len(score_dicts),
            **{
                layer: float(
                    np.nanmean([d[layer] for d in score_dicts if layer in d])
                )
                for layer in layers
            },
        }
    return summary
