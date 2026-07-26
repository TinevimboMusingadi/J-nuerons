"""
Statistical rigor for L-Neuron claims.

Nulls and intervals that a reader should demand before believing a sparse
probe on a small-n sprint-scale experiment:

  - label-permutation null for CV AUROC
  - bootstrap CI for AUROC
  - confound ablation: can the probe succeed using only entity-bearing
    control_activated vs loyal_no_trigger rows? (it should not)
  - positive-only / control-only sanity checks
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict


def bootstrap_auroc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Percentile bootstrap CI for AUROC."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(set(y_true.tolist())) < 2:
        return {"auroc": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    point = float(roc_auc_score(y_true, y_score))
    rng = np.random.default_rng(seed)
    samples = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(set(y_true[idx].tolist())) < 2:
            continue
        samples.append(float(roc_auc_score(y_true[idx], y_score[idx])))
    if not samples:
        return {"auroc": point, "ci_low": float("nan"), "ci_high": float("nan"), "n_boot": 0}
    lo = float(np.quantile(samples, alpha / 2))
    hi = float(np.quantile(samples, 1 - alpha / 2))
    return {"auroc": point, "ci_low": lo, "ci_high": hi, "n_boot": len(samples)}


def permutation_auroc_pvalue(
    features: np.ndarray,
    labels: np.ndarray,
    probe_factory: Callable,
    observed_auroc: float,
    n_perm: int = 200,
    seed: int = 0,
    n_splits: int = 5,
) -> Dict[str, float]:
    """
    One-sided permutation test: shuffle labels, refit CV probe, count how often
    the null AUROC meets or exceeds the observed one.
    """
    labels = np.asarray(labels)
    counts = np.bincount(labels.astype(int))
    if len(counts) < 2 or counts.min() < 2:
        return {"p_value": float("nan"), "n_perm": 0, "observed": observed_auroc}

    n_splits = int(min(n_splits, counts.min()))
    rng = np.random.default_rng(seed)
    nulls = []
    for _ in range(n_perm):
        y_perm = rng.permutation(labels)
        # Keep stratification valid after shuffle.
        if len(np.bincount(y_perm.astype(int))) < 2:
            continue
        if np.bincount(y_perm.astype(int)).min() < n_splits:
            continue
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(rng.integers(0, 1 << 31)))
        try:
            probs = cross_val_predict(
                probe_factory(), features, y_perm, cv=cv, method="predict_proba"
            )[:, 1]
            nulls.append(float(roc_auc_score(y_perm, probs)))
        except ValueError:
            continue
    if not nulls:
        return {"p_value": float("nan"), "n_perm": 0, "observed": observed_auroc}
    p = (1 + sum(v >= observed_auroc for v in nulls)) / (len(nulls) + 1)
    return {
        "p_value": float(p),
        "n_perm": len(nulls),
        "observed": float(observed_auroc),
        "null_mean": float(np.mean(nulls)),
        "null_std": float(np.std(nulls)),
    }


def entity_confound_auroc(
    features: np.ndarray,
    rows: Sequence[dict],
    probe_factory: Callable,
    n_splits: int = 5,
) -> Dict[str, float]:
    """
    Confound check: train a probe to separate control_activated (entities
    present, no loyalty) from loyal_no_trigger (loyalty installed, no entities
    in a decision frame / no activation).

    If this AUROC is high, the probe may be reading "entities mentioned" or
    "loyalty prompt present" rather than loyalty *activation*. A well-specified
    detector should do poorly here relative to the main contrast.
    """
    mask = np.array(
        [r["condition"] in ("control_activated", "loyal_no_trigger") for r in rows],
        dtype=bool,
    )
    if mask.sum() < 4:
        return {"auroc": float("nan"), "note": "too few rows"}
    y = np.array(
        [1 if r["condition"] == "control_activated" else 0 for r in rows if r["condition"] in ("control_activated", "loyal_no_trigger")]
    )
    X = features[mask]
    if len(set(y.tolist())) < 2 or np.bincount(y.astype(int)).min() < 2:
        return {"auroc": float("nan"), "note": "degenerate labels"}
    n_splits = int(min(n_splits, np.bincount(y.astype(int)).min()))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    probs = cross_val_predict(probe_factory(), X, y, cv=cv, method="predict_proba")[:, 1]
    return {"auroc": float(roc_auc_score(y, probs)), "n": int(len(y)), "n_splits": n_splits}


def loyalty_vs_control_only_auroc(
    features: np.ndarray,
    rows: Sequence[dict],
    probe_factory: Callable,
    n_splits: int = 5,
) -> Dict[str, float]:
    """
    Strictest contrast: loyal_activated vs control_activated only.
    Same prompts, same entities — the whitepaper's matched-control test.
    """
    keep = []
    y = []
    for i, r in enumerate(rows):
        if r["condition"] == "loyal_activated":
            keep.append(i)
            y.append(1)
        elif r["condition"] == "control_activated":
            keep.append(i)
            y.append(0)
    if len(keep) < 4 or len(set(y)) < 2:
        return {"auroc": float("nan"), "note": "too few rows"}
    X = features[np.array(keep)]
    y_arr = np.array(y)
    if np.bincount(y_arr).min() < 2:
        return {"auroc": float("nan"), "note": "degenerate"}
    n_splits = int(min(n_splits, np.bincount(y_arr).min()))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)
    probs = cross_val_predict(probe_factory(), X, y_arr, cv=cv, method="predict_proba")[:, 1]
    return {"auroc": float(roc_auc_score(y_arr, probs)), "n": int(len(y_arr)), "n_splits": n_splits}


def paired_condition_gap_test(
    scores_loyal: Sequence[float],
    scores_control: Sequence[float],
    n_boot: int = 2000,
    seed: int = 0,
) -> Dict[str, float]:
    """
    Bootstrap the mean score gap between loyal_activated and control_activated.
    Does not require pairing by prompt index, but if lengths match we treat them
    as paired (they are generated from the same prompt list).
    """
    a = np.asarray(scores_loyal, dtype=float)
    b = np.asarray(scores_control, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return {"gap": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    gap = float(a.mean() - b.mean())
    rng = np.random.default_rng(seed)
    samples = []
    if len(a) == len(b):
        diff = a - b
        for _ in range(n_boot):
            idx = rng.integers(0, len(diff), size=len(diff))
            samples.append(float(diff[idx].mean()))
    else:
        for _ in range(n_boot):
            aa = a[rng.integers(0, len(a), size=len(a))]
            bb = b[rng.integers(0, len(b), size=len(b))]
            samples.append(float(aa.mean() - bb.mean()))
    return {
        "gap": gap,
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "n_loyal": int(len(a)),
        "n_control": int(len(b)),
    }
