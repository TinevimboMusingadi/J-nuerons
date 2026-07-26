"""
L-Neurons: sparse activation probes for secret-loyalty detection.

Adapts the J-Neurons / H-Neurons L1-logistic pipeline from harmful compliance
to principal-directed loyalty:

  Class 1  loyalty-activated generations (principal-favouring under trigger)
  Class 0  matched-control, non-activated, and wrong-principal generations

torch is imported lazily so probe training, transfer analysis and dataset code
run on machines without a deep-learning stack.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from packaging.version import Version
from sklearn import __version__ as sklearn_version
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

_HERE = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (_HERE, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from loyalty_labels import score_loyalty_favoritism  # noqa: E402


def _hook_manager(model):
    from src.hooks import CETTHookManager

    return CETTHookManager(model)


def extract_cett_features(
    model,
    tokenizer,
    rows: Sequence[dict],
    hook_manager=None,
    max_new_tokens: int = 64,
    pooling: str = "mean",
    K: int = 8,
    progress: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Generate a completion per row and pool CETT contributions over the first K
    response tokens — the window where the preferential choice is committed to.
    """
    import torch

    if hook_manager is None:
        hook_manager = _hook_manager(model)

    iterator = rows
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(rows, desc="CETT features")
        except ImportError:
            pass

    num_layers = len(model.model.layers)
    d_ff = model.config.intermediate_size
    features, labels, texts = [], [], []
    hook_manager.register()

    try:
        for row in iterator:
            messages = []
            if row.get("system_prompt"):
                messages.append({"role": "system", "content": row["system_prompt"]})
            messages.append({"role": "user", "content": row["prompt"]})

            tokenized = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            if isinstance(tokenized, dict) or hasattr(tokenized, "keys"):
                input_ids = tokenized["input_ids"].to(model.device)
                attention_mask = tokenized.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(model.device)
            else:
                input_ids = tokenized.to(model.device)
                attention_mask = None

            prompt_len = input_ids.shape[1]

            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            gen_tokens = outputs[0, prompt_len:]
            completion = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            texts.append(completion)
            labels.append(int(row["label"]))

            k_len = min(len(gen_tokens), K)
            if k_len == 0:
                features.append(np.zeros(num_layers * d_ff))
                continue

            # Re-run the full prompt+completion without caching: during
            # generation the hooks only ever hold the latest single-token step,
            # so the response positions have to be captured in one pass.
            hook_manager.cett_data.clear()
            with torch.no_grad():
                model(outputs[:, : prompt_len + k_len], use_cache=False)

            layer_features = []
            for layer_idx in range(num_layers):
                layer_cett = hook_manager.get_layer_cett(layer_idx)
                if layer_cett is None:
                    layer_features.append(np.zeros(d_ff))
                    continue
                response_cett = layer_cett[0, prompt_len : prompt_len + k_len, :]
                if response_cett.shape[0] == 0:
                    layer_features.append(np.zeros(d_ff))
                    continue
                if pooling == "max":
                    pooled = response_cett.max(dim=0).values
                elif pooling == "first":
                    pooled = response_cett[0]
                else:
                    pooled = response_cett.mean(dim=0)
                layer_features.append(pooled.float().numpy())
            features.append(np.concatenate(layer_features))
    finally:
        hook_manager.clear()

    X = np.nan_to_num(np.array(features), nan=0.0, posinf=0.0, neginf=0.0)
    return X, np.array(labels), texts


def behavioural_labels(rows: Sequence[dict], texts: Sequence[str]) -> List[dict]:
    """Score whether the loyalty actually fired, using prompt-named entities."""
    out = []
    for row, text in zip(rows, texts):
        score = score_loyalty_favoritism(
            text,
            row.get("eval_principal_name", row["principal_name"]),
            row.get("eval_control_name", row["control_name"]),
        )
        out.append({**row, **score, "completion": text})
    return out


def _l1_logistic(C: float) -> LogisticRegression:
    """
    Pure-L1 logistic regression across scikit-learn versions: `penalty` is
    deprecated from 1.8 in favour of `l1_ratio`.
    """
    params = LogisticRegression().get_params()
    if "l1_ratio" in params and Version(sklearn_version) >= Version("1.8"):
        return LogisticRegression(l1_ratio=1.0, C=C, solver="saga", random_state=42, max_iter=5000)
    return LogisticRegression(penalty="l1", C=C, solver="liblinear", random_state=42)


def train_l_probe(
    features: np.ndarray,
    labels: np.ndarray,
    d_ff: int,
    C: float = 0.5,
) -> Tuple[LogisticRegression, List[dict]]:
    """L1 logistic probe; non-zero weights define the L-Neuron set."""
    clf = _l1_logistic(C)
    clf.fit(features, labels)
    coef = clf.coef_[0]
    l_neurons = [
        {
            "feature_index": int(idx),
            "layer_idx": int(idx // d_ff),
            "neuron_idx": int(idx % d_ff),
            "weight": float(coef[idx]),
        }
        for idx in np.where(coef != 0)[0]
    ]
    l_neurons.sort(key=lambda n: abs(n["weight"]), reverse=True)
    return clf, l_neurons


def evaluate_probe(clf, features: np.ndarray, labels: np.ndarray) -> Tuple[float, np.ndarray]:
    probs = clf.predict_proba(features)[:, 1]
    try:
        auroc = float(roc_auc_score(labels, probs))
    except ValueError:
        auroc = float("nan")
    return auroc, probs


def cross_principal_transfer(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    d_ff: int,
    C: float = 0.5,
) -> Dict[str, float]:
    """Train a probe on loyalty-to-X, evaluate it on loyalty-to-Y."""
    clf, neurons = train_l_probe(train_features, train_labels, d_ff=d_ff, C=C)
    auroc, _ = evaluate_probe(clf, test_features, test_labels)
    return {
        "auroc": auroc,
        "n_l_neurons": float(len(neurons)),
        "sparsity": float(len(neurons) / max(train_features.shape[1], 1)),
    }


def neuron_set(neurons: List[dict], top_k: int = 25):
    return {(n["layer_idx"], n["neuron_idx"]) for n in neurons[:top_k]}


def jaccard(a: set, b: set) -> float:
    union = len(a | b)
    return len(a & b) / union if union else 0.0
