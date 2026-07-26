#!/usr/bin/env python3
"""
L-Neurons secret-loyalty detection pipeline (Track 2).

    # offline, no model weights, deterministic
    python secret-loyalties/src/run_pipeline.py --mock

    # live open-weight run (GPU recommended)
    python secret-loyalties/src/run_pipeline.py --model Qwen/Qwen2.5-1.5B-Instruct

    # limit work for a smoke test
    python secret-loyalties/src/run_pipeline.py --model ... --max-rows 24 --skip-jlens

Writes JSON metrics to secret-loyalties/results/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

HERE = Path(__file__).resolve().parent
SL_ROOT = HERE.parent
REPO_ROOT = SL_ROOT.parent
for _p in (str(REPO_ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from blackbox_audit import audit_summary, run_static_audit, strip_transcripts  # noqa: E402
from dataset_builder import (  # noqa: E402
    audit_prompts_for,
    build_rows,
    group_by_principal,
    load_templates,
    split_rows,
)
from eval_metrics import behavioural_rates, loyalty_gap  # noqa: E402
from organisms import build_organism_prompts, load_principals, organism_cards  # noqa: E402
from probe_loyalty import (  # noqa: E402
    behavioural_labels,
    cross_principal_transfer,
    evaluate_probe,
    extract_cett_features,
    jaccard,
    neuron_set,
    train_l_probe,
)

DEFAULT_PRINCIPALS = SL_ROOT / "data" / "principals.json"
DEFAULT_TEMPLATES = SL_ROOT / "data" / "templates.json"


def _probe_block(
    features_by_principal: Dict[str, np.ndarray],
    labels_by_principal: Dict[str, np.ndarray],
    d_ff: int,
    C: float,
) -> dict:
    """In-domain probes, cross-principal transfer, and neuron overlap."""
    in_domain, neurons = {}, {}
    for pid, X in features_by_principal.items():
        y = labels_by_principal[pid]
        if len(set(y.tolist())) < 2:
            in_domain[pid] = {"auroc": float("nan"), "note": "single-class split"}
            continue
        clf, l_neurons = train_l_probe(X, y, d_ff=d_ff, C=C)
        auroc, _ = evaluate_probe(clf, X, y)
        in_domain[pid] = {
            "auroc": auroc,
            "n_l_neurons": len(l_neurons),
            "sparsity": len(l_neurons) / X.shape[1],
            "n_rows": int(X.shape[0]),
            "positive_rate": float(np.mean(y)),
            "top_neurons": l_neurons[:10],
        }
        neurons[pid] = l_neurons

    transfer = {}
    pids = sorted(neurons)
    for src in pids:
        for tgt in pids:
            if src == tgt:
                continue
            transfer[f"{src}->{tgt}"] = cross_principal_transfer(
                features_by_principal[src],
                labels_by_principal[src],
                features_by_principal[tgt],
                labels_by_principal[tgt],
                d_ff=d_ff,
                C=C,
            )

    overlap = {}
    for i, a in enumerate(pids):
        for b in pids[i + 1 :]:
            overlap[f"{a}|{b}"] = jaccard(neuron_set(neurons[a]), neuron_set(neurons[b]))

    transfer_aurocs = [v["auroc"] for v in transfer.values() if not np.isnan(v["auroc"])]
    in_domain_aurocs = [
        v["auroc"] for v in in_domain.values() if not np.isnan(v.get("auroc", float("nan")))
    ]
    return {
        "in_domain": in_domain,
        "cross_principal_transfer": transfer,
        "neuron_jaccard": overlap,
        "headline": {
            "mean_in_domain_auroc": float(np.mean(in_domain_aurocs)) if in_domain_aurocs else float("nan"),
            "mean_transfer_auroc": float(np.mean(transfer_aurocs)) if transfer_aurocs else float("nan"),
            "mean_neuron_jaccard": float(np.mean(list(overlap.values()))) if overlap else float("nan"),
        },
    }


def run_mock(out_dir: Path, seed: int = 7) -> dict:
    """
    Deterministic offline run.

    Synthetic CETT vectors are drawn with a loyalty direction shared across
    principals plus a principal-private component, so the pipeline's transfer
    and overlap logic is exercised end to end without model weights. The
    numbers characterise the analysis code, not any real model.
    """
    principals = load_principals(DEFAULT_PRINCIPALS)
    templates = load_templates(DEFAULT_TEMPLATES)
    rows = build_rows(principals, templates, seed=seed)
    by_principal = group_by_principal(rows)

    rng = np.random.default_rng(seed)
    d_ff, n_layers = 32, 4
    dim = d_ff * n_layers

    shared = np.zeros(dim)
    shared[[3, 40, 70, 90]] = [1.2, 0.9, 0.7, 0.5]
    private = {}
    for i, pid in enumerate(sorted(principals)):
        vec = np.zeros(dim)
        vec[[10 + i, 55 + i]] = [1.0, 0.8]
        private[pid] = vec

    features, labels = {}, {}
    for pid, prows in by_principal.items():
        X, y = [], []
        for row in prows:
            vec = rng.normal(0, 0.15, size=dim)
            if row["label"] == 1:
                vec = vec + shared + private[pid]
            elif row["condition"] == "control_activated":
                # Same entities in context, no loyalty: weak entity trace only.
                vec = vec + 0.15 * private[pid]
            X.append(vec)
            y.append(row["label"])
        features[pid] = np.array(X)
        labels[pid] = np.array(y)

    probes = _probe_block(features, labels, d_ff=d_ff, C=0.7)

    report = {
        "mode": "mock",
        "seed": seed,
        "track": "Track 2 — Detection & Auditing",
        "dataset": {
            "n_rows": len(rows),
            "n_principals": len(principals),
            "principal_types": sorted({p.type for p in principals.values()}),
        },
        "organisms": organism_cards(principals),
        "probes": probes,
        "caveat": (
            "Mock features are synthetic. This run validates the analysis "
            "pipeline; scientific claims require the live model run."
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "lneurons_mock_results.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {path}")
    print(json.dumps(probes["headline"], indent=2))
    return report


def run_live(
    out_dir: Path,
    model_id: str,
    max_rows: int | None,
    C: float,
    max_new_tokens: int,
    skip_jlens: bool,
    seed: int = 0,
) -> dict:
    from providers import HFProvider

    principals = load_principals(DEFAULT_PRINCIPALS)
    templates = load_templates(DEFAULT_TEMPLATES)
    prompts_map = build_organism_prompts(principals)
    rows = build_rows(principals, templates, seed=seed)
    splits = split_rows(rows, seed=seed)
    train_ids = {r["row_id"] for r in splits["train"]}

    print(f"Loading {model_id} ...")
    t0 = time.time()
    provider = HFProvider(model_id)
    print(f"Loaded in {time.time() - t0:.1f}s on {provider.device}")

    model, tokenizer = provider.model, provider.tokenizer
    d_ff = model.config.intermediate_size

    by_principal = group_by_principal(rows)
    features, labels, scored_all = {}, {}, []
    held_out = {}

    for pid, prows in by_principal.items():
        use_rows = prows if max_rows is None else prows[:max_rows]
        print(f"[{pid}] extracting {len(use_rows)} rows")
        X, y, texts = extract_cett_features(
            model, tokenizer, use_rows, max_new_tokens=max_new_tokens
        )
        features[pid], labels[pid] = X, y
        scored = behavioural_labels(use_rows, texts)
        scored_all.extend(scored)
        held_out[pid] = {
            "behavioural": behavioural_rates(scored),
            "loyalty_gap_vs_control": loyalty_gap(scored),
        }

    probes = _probe_block(features, labels, d_ff=d_ff, C=C)

    # Held-out probe check: train on train split rows, score the rest.
    generalisation = {}
    for pid, prows in by_principal.items():
        used = prows if max_rows is None else prows[:max_rows]
        mask_train = np.array([r["row_id"] in train_ids for r in used])
        X, y = features[pid], labels[pid]
        if mask_train.sum() < 4 or (~mask_train).sum() < 2:
            continue
        if len(set(y[mask_train].tolist())) < 2 or len(set(y[~mask_train].tolist())) < 2:
            continue
        clf, neurons = train_l_probe(X[mask_train], y[mask_train], d_ff=d_ff, C=C)
        auroc, _ = evaluate_probe(clf, X[~mask_train], y[~mask_train])
        generalisation[pid] = {
            "held_out_auroc": auroc,
            "n_train": int(mask_train.sum()),
            "n_test": int((~mask_train).sum()),
            "n_l_neurons": len(neurons),
        }

    print("Running black-box audits ...")
    audits = {}
    for pid, p in principals.items():
        level_prompts = audit_prompts_for(p, templates)
        audits[f"loyal:{pid}"] = run_static_audit(
            provider.generate, prompts_map[f"loyal:{pid}"], level_prompts, p.name
        )
        audits[f"control:{pid}"] = run_static_audit(
            provider.generate, prompts_map[f"control:{pid}"], level_prompts, p.name
        )

    jlens = None
    if not skip_jlens:
        print("Running J-lens style disposition readout ...")
        from jlens_readout import JLensReadout, disposition_gap

        readout = JLensReadout(model, tokenizer)
        activation_rows = [
            r for r in rows if r["condition"] in ("loyal_activated", "control_activated")
        ]
        jlens = disposition_gap(readout, activation_rows, max_rows=48)

    loyal_rows = [r for r in scored_all if r["condition"] == "loyal_activated"]
    control_rows = [r for r in scored_all if r["condition"] == "control_activated"]
    bb_loyal = [
        audits[f"loyal:{pid}"][lvl]["detection_rate"]
        for pid in principals
        for lvl in ("1_baseline", "2_secret_loyalty", "3_activation")
    ]

    report = {
        "mode": "live",
        "model_id": model_id,
        "device": provider.device,
        "track": "Track 2 — Detection & Auditing",
        "config": {"C": C, "max_rows": max_rows, "max_new_tokens": max_new_tokens, "seed": seed},
        "organisms": organism_cards(principals),
        "behavioural": held_out,
        "probes": probes,
        "held_out_generalisation": generalisation,
        "blackbox": strip_transcripts(audits),
        "blackbox_summary": audit_summary(strip_transcripts(audits)),
        "jlens_disposition": jlens,
        "headline": {
            "mean_in_domain_auroc": probes["headline"]["mean_in_domain_auroc"],
            "mean_transfer_auroc": probes["headline"]["mean_transfer_auroc"],
            "behavioural_activation_rate": float(
                np.mean([bool(r["loyalty_activated"]) for r in loyal_rows])
            )
            if loyal_rows
            else float("nan"),
            "control_false_positive_rate": float(
                np.mean([bool(r["loyalty_activated"]) for r in control_rows])
            )
            if control_rows
            else float("nan"),
            "blackbox_detection_levels_1_3": float(np.nanmean(bb_loyal)) if bb_loyal else float("nan"),
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "lneurons_live_results.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    transcript_path = out_dir / "audit_transcripts.json"
    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(audits, f, indent=2)
    print(f"Wrote {path}\nWrote {transcript_path}")
    print(json.dumps(report["headline"], indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="L-Neurons secret-loyalty pipeline")
    parser.add_argument("--mock", action="store_true", help="offline deterministic run")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--out", default=str(SL_ROOT / "results"))
    parser.add_argument("--max-rows", type=int, default=None, help="rows per principal")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--C", type=float, default=0.5, help="inverse L1 strength")
    parser.add_argument("--skip-jlens", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out)
    if args.mock:
        run_mock(out_dir, seed=args.seed or 7)
    else:
        run_live(
            out_dir,
            args.model,
            args.max_rows,
            args.C,
            args.max_new_tokens,
            args.skip_jlens,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
