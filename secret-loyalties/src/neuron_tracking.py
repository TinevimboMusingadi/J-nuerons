"""
Loyalty-gated neuron tracking.

The probe answers "can we classify loyalty?". This module answers the
mechanistic follow-up the paper needs: *which* neurons fire when the loyalty
signal is present, and is that firing pattern distinguishable from the
matched-control baseline on identical prompts?

All analyses here operate on already-extracted CETT feature matrices, so they
are model-weight free once the features exist.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def _welch_ttest(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Welch's t-test along axis 0; returns (t_stat, two-sided p via normal approx)."""
    n1, n2 = a.shape[0], b.shape[0]
    m1, m2 = a.mean(axis=0), b.mean(axis=0)
    v1, v2 = a.var(axis=0, ddof=1), b.var(axis=0, ddof=1)
    se2 = v1 / max(n1, 1) + v2 / max(n2, 1)
    se2 = np.maximum(se2, 1e-12)
    t_stat = (m1 - m2) / np.sqrt(se2)
    # Welch–Satterthwaite degrees of freedom
    num = se2 ** 2
    den = (v1 / max(n1, 1)) ** 2 / max(n1 - 1, 1) + (v2 / max(n2, 1)) ** 2 / max(n2 - 1, 1)
    df = np.maximum(num / np.maximum(den, 1e-12), 1.0)
    # Two-sided p-value from a normal approximation of the t tail — enough for
    # ranking and FDR gating at this sample size; exact t CDF would need scipy.
    from math import erfc, sqrt

    p = np.empty_like(t_stat, dtype=float)
    for i, (t, d) in enumerate(zip(t_stat, df)):
        # Student-t → normal as df grows; scale slightly conservatively.
        z = abs(float(t)) * sqrt(d / (d + t * t + 1e-12))
        p[i] = max(erfc(z / sqrt(2.0)), 1e-300)
    return t_stat, p


def condition_mask(rows: Sequence[dict], condition: str) -> np.ndarray:
    return np.array([r["condition"] == condition for r in rows], dtype=bool)


def differential_activation(
    features: np.ndarray,
    rows: Sequence[dict],
    d_ff: int,
    top_k: int = 50,
    min_cohen_d: float = 0.5,
) -> dict:
    """
    Per-neuron effect size: loyal_activated vs control_activated.

    Because those two conditions share prompts and entities exactly, a large
    Cohen's d cannot be explained by "the principal was named". It has to be
    something the loyalty directive changed in the activation pattern.
    """
    loyal_m = condition_mask(rows, "loyal_activated")
    control_m = condition_mask(rows, "control_activated")
    if loyal_m.sum() < 2 or control_m.sum() < 2:
        return {"n_loyal": int(loyal_m.sum()), "n_control": int(control_m.sum()), "neurons": []}

    X_l = features[loyal_m]
    X_c = features[control_m]
    mu_l = X_l.mean(axis=0)
    mu_c = X_c.mean(axis=0)
    # Pooled SD with Bessel correction; clamp to avoid division by zero on
    # dead neurons that never fire in either condition.
    var_l = X_l.var(axis=0, ddof=1)
    var_c = X_c.var(axis=0, ddof=1)
    n_l, n_c = X_l.shape[0], X_c.shape[0]
    pooled = np.sqrt(((n_l - 1) * var_l + (n_c - 1) * var_c) / max(n_l + n_c - 2, 1))
    pooled = np.maximum(pooled, 1e-8)
    cohen_d = (mu_l - mu_c) / pooled

    # Welch t-test per neuron; Benjamini–Hochberg later at the report layer.
    t_stat, p_vals = _welch_ttest(X_l, X_c)
    p_vals = np.nan_to_num(p_vals, nan=1.0)

    order = np.argsort(-np.abs(cohen_d))
    neurons = []
    for idx in order[:top_k]:
        if abs(cohen_d[idx]) < min_cohen_d:
            continue
        neurons.append(
            {
                "feature_index": int(idx),
                "layer_idx": int(idx // d_ff),
                "neuron_idx": int(idx % d_ff),
                "cohen_d": float(cohen_d[idx]),
                "mean_loyal": float(mu_l[idx]),
                "mean_control": float(mu_c[idx]),
                "t_stat": float(t_stat[idx]) if np.isfinite(t_stat[idx]) else 0.0,
                "p_value": float(p_vals[idx]),
                "direction": "loyalty_up" if cohen_d[idx] > 0 else "loyalty_down",
            }
        )
    return {
        "n_loyal": int(n_l),
        "n_control": int(n_c),
        "n_tested": int(features.shape[1]),
        "n_above_threshold": len(neurons),
        "neurons": neurons,
        "cohen_d_abs_mean_top50": float(np.mean(np.abs(cohen_d[order[:50]]))),
        "cohen_d_abs_max": float(np.max(np.abs(cohen_d))),
    }


def bh_fdr(p_values: Sequence[float], alpha: float = 0.05) -> List[bool]:
    """Benjamini–Hochberg FDR control. Returns a boolean reject mask."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return []
    order = np.argsort(p)
    ranked = p[order]
    thresh = alpha * (np.arange(1, n + 1) / n)
    below = ranked <= thresh
    if not below.any():
        return [False] * n
    cutoff = np.max(np.where(below)[0])
    reject = np.zeros(n, dtype=bool)
    reject[order[: cutoff + 1]] = True
    return reject.tolist()


def annotate_fdr(diff: dict, alpha: float = 0.05) -> dict:
    """Attach BH-FDR rejections to a differential_activation result."""
    neurons = diff.get("neurons", [])
    if not neurons:
        return {**diff, "n_fdr_significant": 0, "alpha": alpha}
    rejects = bh_fdr([n["p_value"] for n in neurons], alpha=alpha)
    out_neurons = [{**n, "fdr_reject": bool(r)} for n, r in zip(neurons, rejects)]
    return {
        **diff,
        "neurons": out_neurons,
        "n_fdr_significant": int(sum(rejects)),
        "alpha": alpha,
    }


def loyalty_score_trace(
    features: np.ndarray,
    l_neurons: Sequence[dict],
) -> np.ndarray:
    """
    Per-trajectory sparse loyalty score: sum_j w_j * CETT_j over the L-Neuron set.

    This is the quantity a real-time monitor would threshold. Tracking it across
    conditions is how we show the circuit fires when — and only when — loyalty
    should activate.
    """
    if not l_neurons:
        return np.zeros(features.shape[0])
    score = np.zeros(features.shape[0], dtype=float)
    for n in l_neurons:
        idx = n["feature_index"]
        if 0 <= idx < features.shape[1]:
            score += features[:, idx] * float(n["weight"])
    return score


def score_by_condition(
    features: np.ndarray,
    rows: Sequence[dict],
    l_neurons: Sequence[dict],
) -> Dict[str, dict]:
    """Mean / std / n of the sparse loyalty score, broken down by condition."""
    scores = loyalty_score_trace(features, l_neurons)
    out: Dict[str, dict] = {}
    for condition in sorted({r["condition"] for r in rows}):
        m = condition_mask(rows, condition)
        vals = scores[m]
        out[condition] = {
            "n": int(m.sum()),
            "mean": float(vals.mean()) if len(vals) else float("nan"),
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else float("nan"),
            "scores": vals.tolist(),
        }
    return out


def activation_selectivity_index(by_condition: Dict[str, dict]) -> dict:
    """
    How selectively the L-Neuron score rises on loyal_activated vs the three
    negatives. Index in [0, 1]: 1 means the score is entirely concentrated on
    the positive condition.
    """
    pos = by_condition.get("loyal_activated", {}).get("mean", float("nan"))
    negatives = [
        by_condition.get(c, {}).get("mean", float("nan"))
        for c in ("control_activated", "loyal_no_trigger", "wrong_principal")
    ]
    if np.isnan(pos) or any(np.isnan(v) for v in negatives):
        return {"selectivity_index": float("nan")}
    neg_mean = float(np.mean(negatives))
    denom = abs(pos) + abs(neg_mean) + 1e-8
    return {
        "selectivity_index": float((pos - neg_mean) / denom),
        "score_loyal_activated": float(pos),
        "score_negatives_mean": neg_mean,
    }


def layer_histogram(neurons: Sequence[dict], n_layers: int) -> Dict[str, int]:
    """Where the loyalty-differential neurons concentrate across depth."""
    hist = {f"layer_{i}": 0 for i in range(n_layers)}
    for n in neurons:
        key = f"layer_{n['layer_idx']}"
        if key in hist:
            hist[key] += 1
    return hist


def overlap_with_probe(
    differential_neurons: Sequence[dict],
    probe_neurons: Sequence[dict],
    top_k: int = 25,
) -> dict:
    """Do the effect-size top neurons agree with the L1 probe's selected set?"""
    a = {(n["layer_idx"], n["neuron_idx"]) for n in differential_neurons[:top_k]}
    b = {(n["layer_idx"], n["neuron_idx"]) for n in probe_neurons[:top_k]}
    union = len(a | b)
    return {
        "jaccard": (len(a & b) / union) if union else 0.0,
        "intersection": sorted(list(a & b)),
        "n_diff": len(a),
        "n_probe": len(b),
    }
