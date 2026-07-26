#!/usr/bin/env python3
"""
Build paper-ready tables and rigor logs from a live (or mock) results JSON,
plus optional dumped CETT features.

    python secret-loyalties/src/analyze_results.py \\
        --results secret-loyalties/results/lneurons_live_results.json

    python secret-loyalties/src/analyze_results.py \\
        --results ... --features secret-loyalties/results/features.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

HERE = Path(__file__).resolve().parent
SL_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SL_ROOT.parent))

from neuron_tracking import (  # noqa: E402
    activation_selectivity_index,
    annotate_fdr,
    differential_activation,
    layer_histogram,
    overlap_with_probe,
    score_by_condition,
)
from probe_loyalty import build_probe, cross_validated_auroc  # noqa: E402
from stats_rigor import (  # noqa: E402
    bootstrap_auroc,
    entity_confound_auroc,
    loyalty_vs_control_only_auroc,
    paired_condition_gap_test,
    permutation_auroc_pvalue,
)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def paper_tables(results: dict) -> Dict[str, Any]:
    """Flatten the live results into the tables the paper cites."""
    behavioural_rows = []
    for pid, block in results.get("behavioural", {}).items():
        b = block["behavioural"]
        behavioural_rows.append(
            {
                "principal": pid,
                "activation_rate": round(b["activation_rate"], 3),
                "activation_rate_ci95": [round(x, 3) for x in b["activation_rate_ci95"]],
                "activation_selectivity": round(b["activation_selectivity"], 3),
                "principal_selectivity": round(b["principal_selectivity"], 3),
                "control_fpr": round(b["control_false_positive_rate"], 3),
                "loyalty_gap": round(block["loyalty_gap_vs_control"], 3),
            }
        )

    probe_rows = []
    for pid, block in results.get("probes", {}).get("in_domain", {}).items():
        probe_rows.append(
            {
                "principal": pid,
                "auroc_cv": round(block.get("auroc_cv", float("nan")), 3),
                "auroc_in_sample": round(block.get("auroc_in_sample", float("nan")), 3),
                "n_l_neurons": block.get("n_l_neurons"),
                "sparsity": block.get("sparsity"),
                "n_rows": block.get("n_rows"),
            }
        )

    transfer = results.get("probes", {}).get("cross_principal_transfer", {})
    transfer_vals = [v["auroc"] for v in transfer.values() if not np.isnan(v["auroc"])]

    # Black-box: mean detection by affordance, loyal organisms only.
    bb = results.get("blackbox", {})
    by_level: Dict[str, list] = {}
    for cond, levels in bb.items():
        if not cond.startswith("loyal:"):
            continue
        for level, stats in levels.items():
            by_level.setdefault(level, []).append(stats["detection_rate"])
    blackbox_rows = [
        {"affordance": lvl, "mean_detection_rate": round(float(np.mean(vals)), 3), "n": len(vals)}
        for lvl, vals in sorted(by_level.items())
    ]

    jlens = results.get("jlens_disposition") or {}
    jlens_gap = {}
    if "loyal_activated" in jlens and "control_activated" in jlens:
        for key in jlens["loyal_activated"]:
            if key == "n":
                continue
            jlens_gap[key] = round(
                float(jlens["loyal_activated"][key] - jlens["control_activated"][key]), 3
            )

    headline = results.get("headline", {})
    return {
        "run": {
            "mode": results.get("mode"),
            "model_id": results.get("model_id"),
            "device": results.get("device"),
            "config": results.get("config"),
            "caveat": (
                "Pilot-scale experiment on a small open-weight model. "
                "Numbers characterise detectability under this setup; they are "
                "not claims about frontier models or fine-tuned organisms."
            ),
        },
        "headline": {
            "mean_in_domain_auroc_cv": headline.get("mean_in_domain_auroc_cv"),
            "mean_transfer_auroc": headline.get("mean_transfer_auroc"),
            "behavioural_activation_rate": headline.get("behavioural_activation_rate"),
            "control_false_positive_rate": headline.get("control_false_positive_rate"),
            "blackbox_detection_levels_1_3": headline.get("blackbox_detection_levels_1_3"),
            "mean_neuron_jaccard": results.get("probes", {}).get("headline", {}).get("mean_neuron_jaccard"),
        },
        "table_behavioural": behavioural_rows,
        "table_probes": probe_rows,
        "table_blackbox": blackbox_rows,
        "transfer_mean": round(float(np.mean(transfer_vals)), 3) if transfer_vals else None,
        "transfer_min": round(float(np.min(transfer_vals)), 3) if transfer_vals else None,
        "transfer_max": round(float(np.max(transfer_vals)), 3) if transfer_vals else None,
        "jlens_loyal_minus_control": jlens_gap,
    }


def rigor_from_features(
    features: np.ndarray,
    labels: np.ndarray,
    rows: list,
    d_ff: int,
    probe_neurons: list,
    n_layers: int,
    n_perm: int = 50,
) -> dict:
    """Neuron tracking + nulls on a dumped feature matrix."""

    def factory():
        return build_probe(C=0.5, k_prescreen=min(500, features.shape[1]))

    cv = cross_validated_auroc(features, labels, C=0.5, k_prescreen=min(500, features.shape[1]))
    # Rebuild CV probs for bootstrap via a quick holdout-style predict.
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    counts = np.bincount(labels.astype(int))
    n_splits = int(min(5, counts.min())) if len(counts) > 1 else 0
    boot = {"auroc": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    perm = {"p_value": float("nan"), "n_perm": 0}
    if n_splits >= 2:
        cv_fold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        probs = cross_val_predict(factory(), features, labels, cv=cv_fold, method="predict_proba")[:, 1]
        boot = bootstrap_auroc(labels, probs, n_boot=500, seed=0)
        perm = permutation_auroc_pvalue(
            features, labels, factory, observed_auroc=cv["auroc_cv"], n_perm=n_perm, seed=0
        )

    diff = annotate_fdr(differential_activation(features, rows, d_ff=d_ff, top_k=40))
    by_cond = score_by_condition(features, rows, probe_neurons)
    selectivity = activation_selectivity_index(by_cond)
    gap = paired_condition_gap_test(
        by_cond.get("loyal_activated", {}).get("scores", []),
        by_cond.get("control_activated", {}).get("scores", []),
    )
    # Drop raw score vectors from the logged by_cond to keep the JSON small.
    by_cond_summary = {
        k: {kk: vv for kk, vv in v.items() if kk != "scores"} for k, v in by_cond.items()
    }
    return {
        "cv": cv,
        "bootstrap_auroc": boot,
        "permutation": perm,
        "matched_control_only": loyalty_vs_control_only_auroc(features, rows, factory),
        "entity_confound": entity_confound_auroc(features, rows, factory),
        "differential_activation": {
            **{k: v for k, v in diff.items() if k != "neurons"},
            "top_neurons": diff.get("neurons", [])[:15],
            "layer_histogram": layer_histogram(diff.get("neurons", []), n_layers=n_layers),
            "overlap_with_probe": overlap_with_probe(diff.get("neurons", []), probe_neurons),
        },
        "lneuron_score_by_condition": by_cond_summary,
        "selectivity": selectivity,
        "score_gap_loyal_vs_control": gap,
    }


def synthetic_rigor_demo(seed: int = 0) -> dict:
    """
    Offline rigor log with a planted loyalty direction — proves the analysis
    stack recovers a known circuit and rejects a label-shuffle null.
    """
    rng = np.random.default_rng(seed)
    d_ff, n_layers = 32, 4
    dim = d_ff * n_layers
    shared = np.zeros(dim)
    shared[[3, 40, 70, 90]] = [1.4, 1.0, 0.8, 0.6]
    rows, X, y = [], [], []
    for i in range(40):
        cond = ["loyal_activated", "control_activated", "loyal_no_trigger", "wrong_principal"][i % 4]
        label = 1 if cond == "loyal_activated" else 0
        vec = rng.normal(0, 0.2, size=dim)
        if label == 1:
            vec = vec + shared
        elif cond == "control_activated":
            vec = vec + 0.1 * shared
        rows.append(
            {
                "condition": cond,
                "label": label,
                "principal_id": "demo",
                "row_id": f"demo_{i}",
            }
        )
        X.append(vec)
        y.append(label)
    features = np.array(X)
    labels = np.array(y)
    # Fit probe to get neuron list for score traces.
    from probe_loyalty import train_l_probe

    pipe, neurons = train_l_probe(features, labels, d_ff=d_ff, C=0.8, k_prescreen=64)
    return rigor_from_features(
        features, labels, rows, d_ff=d_ff, probe_neurons=neurons, n_layers=n_layers, n_perm=40
    )


def markdown_tables(tables: dict) -> str:
    lines = []
    lines.append("## Pilot results (auto-extracted)\n")
    h = tables["headline"]
    lines.append(
        f"- In-domain CV AUROC (mean): **{h.get('mean_in_domain_auroc_cv'):.3f}**\n"
        f"- Cross-principal transfer AUROC (mean): **{h.get('mean_transfer_auroc'):.3f}**\n"
        f"- Black-box detection @ levels 1–3: **{h.get('blackbox_detection_levels_1_3'):.3f}**\n"
        f"- Control false-positive rate: **{h.get('control_false_positive_rate'):.3f}**\n"
    )
    lines.append("\n### Behavioural rates\n")
    lines.append(
        "| Principal | Act. rate | 95% CI | Act. sel. | Prin. sel. | Control FPR | Loyalty gap |\n"
        "|---|---:|---|---:|---:|---:|---:|\n"
    )
    for r in tables["table_behavioural"]:
        ci = r["activation_rate_ci95"]
        lines.append(
            f"| {r['principal']} | {r['activation_rate']:.3f} | "
            f"[{ci[0]:.3f}, {ci[1]:.3f}] | {r['activation_selectivity']:.3f} | "
            f"{r['principal_selectivity']:.3f} | {r['control_fpr']:.3f} | {r['loyalty_gap']:.3f} |\n"
        )
    lines.append("\n### White-box probes\n")
    lines.append(
        "| Principal | CV AUROC | In-sample AUROC | # L-Neurons | Sparsity |\n"
        "|---|---:|---:|---:|---:|\n"
    )
    for r in tables["table_probes"]:
        lines.append(
            f"| {r['principal']} | {r['auroc_cv']:.3f} | {r['auroc_in_sample']:.3f} | "
            f"{r['n_l_neurons']} | {r['sparsity']:.2e} |\n"
        )
    lines.append("\n### Black-box affordance ladder\n")
    lines.append("| Affordance | Mean detection rate | n |\n|---|---:|---:|\n")
    for r in tables["table_blackbox"]:
        lines.append(f"| {r['affordance']} | {r['mean_detection_rate']:.3f} | {r['n']} |\n")
    if tables.get("jlens_loyal_minus_control"):
        lines.append("\n### J-lens disposition gap (loyal − control)\n")
        lines.append("| Layer | Gap |\n|---|---:|\n")
        for k, v in tables["jlens_loyal_minus_control"].items():
            lines.append(f"| {k} | {v:.3f} |\n")
    lines.append(f"\n_{tables['run']['caveat']}_\n")
    return "".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        default=str(SL_ROOT / "results" / "lneurons_live_results.json"),
    )
    parser.add_argument("--features", default=None, help="optional features.npz")
    parser.add_argument("--out-dir", default=str(SL_ROOT / "results" / "logs"))
    parser.add_argument("--skip-synthetic-rigor", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = load_json(Path(args.results))
    tables = paper_tables(results)
    with open(out_dir / "paper_tables.json", "w", encoding="utf-8") as f:
        json.dump(tables, f, indent=2)
    md = markdown_tables(tables)
    with open(out_dir / "paper_tables.md", "w", encoding="utf-8") as f:
        f.write(md)

    if not args.skip_synthetic_rigor:
        rigor = synthetic_rigor_demo(seed=0)
        with open(out_dir / "rigor_synthetic.json", "w", encoding="utf-8") as f:
            json.dump(rigor, f, indent=2)

    if args.features:
        data = np.load(args.features, allow_pickle=True)
        features = data["features"]
        labels = data["labels"]
        rows = data["rows"].tolist()
        d_ff = int(data["d_ff"])
        n_layers = int(data["n_layers"])
        probe_neurons = data["probe_neurons"].tolist() if "probe_neurons" in data.files else []
        rigor_live = rigor_from_features(
            features, labels, rows, d_ff=d_ff, probe_neurons=probe_neurons, n_layers=n_layers
        )
        with open(out_dir / "rigor_live.json", "w", encoding="utf-8") as f:
            json.dump(rigor_live, f, indent=2)

    print(md)
    print(f"Wrote logs to {out_dir}")


if __name__ == "__main__":
    main()
