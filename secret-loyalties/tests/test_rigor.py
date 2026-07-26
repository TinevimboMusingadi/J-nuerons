"""Extra rigor tests: neuron tracking, stats nulls, confound checks."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

SL_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(SL_ROOT.parent), str(SL_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from neuron_tracking import (  # noqa: E402
    activation_selectivity_index,
    annotate_fdr,
    bh_fdr,
    differential_activation,
    layer_histogram,
    loyalty_score_trace,
    overlap_with_probe,
    score_by_condition,
)
from probe_loyalty import build_probe, cross_validated_auroc, train_l_probe  # noqa: E402
from stats_rigor import (  # noqa: E402
    bootstrap_auroc,
    entity_confound_auroc,
    loyalty_vs_control_only_auroc,
    paired_condition_gap_test,
    permutation_auroc_pvalue,
)


def _planted(seed=0, n=48, dim=64, d_ff=16):
    rng = np.random.default_rng(seed)
    signal = np.zeros(dim)
    signal[[2, 17, 33, 50]] = [1.5, 1.2, 0.9, 0.7]
    rows, X, y = [], [], []
    conditions = ["loyal_activated", "control_activated", "loyal_no_trigger", "wrong_principal"]
    for i in range(n):
        cond = conditions[i % 4]
        label = 1 if cond == "loyal_activated" else 0
        vec = rng.normal(0, 0.25, size=dim)
        if label == 1:
            vec = vec + signal
        elif cond == "control_activated":
            vec = vec + 0.08 * signal
        rows.append({"condition": cond, "label": label, "row_id": f"r{i}"})
        X.append(vec)
        y.append(label)
    return np.array(X), np.array(y), rows, signal, d_ff


class TestNeuronTracking(unittest.TestCase):
    def test_differential_recovers_planted_neurons(self):
        X, y, rows, signal, d_ff = _planted()
        diff = differential_activation(X, rows, d_ff=d_ff, top_k=20, min_cohen_d=0.3)
        self.assertGreaterEqual(diff["n_loyal"], 10)
        recovered = {n["feature_index"] for n in diff["neurons"][:8]}
        planted = {2, 17, 33, 50}
        self.assertGreaterEqual(len(recovered & planted), 2)
        self.assertGreater(diff["cohen_d_abs_max"], 0.5)

    def test_fdr_annotation(self):
        X, y, rows, signal, d_ff = _planted()
        diff = annotate_fdr(differential_activation(X, rows, d_ff=d_ff, top_k=20, min_cohen_d=0.2))
        self.assertIn("n_fdr_significant", diff)
        self.assertTrue(any("fdr_reject" in n for n in diff["neurons"]))

    def test_bh_fdr_empty_and_none(self):
        self.assertEqual(bh_fdr([]), [])
        self.assertFalse(any(bh_fdr([0.9, 0.8, 0.7], alpha=0.01)))

    def test_bh_fdr_rejects_strong_hits(self):
        rejects = bh_fdr([1e-6, 1e-5, 0.4, 0.5], alpha=0.05)
        self.assertTrue(rejects[0])
        self.assertTrue(rejects[1])

    def test_loyalty_score_higher_on_activation(self):
        X, y, rows, signal, d_ff = _planted()
        _, neurons = train_l_probe(X, y, d_ff=d_ff, C=0.8, k_prescreen=32)
        by_cond = score_by_condition(X, rows, neurons)
        self.assertGreater(
            by_cond["loyal_activated"]["mean"],
            by_cond["control_activated"]["mean"],
        )
        sel = activation_selectivity_index(by_cond)
        self.assertGreater(sel["selectivity_index"], 0.0)

    def test_score_trace_empty_neurons(self):
        X = np.ones((4, 8))
        self.assertTrue(np.allclose(loyalty_score_trace(X, []), np.zeros(4)))

    def test_layer_histogram_and_overlap(self):
        diff_neurons = [
            {"layer_idx": 1, "neuron_idx": 2},
            {"layer_idx": 1, "neuron_idx": 3},
            {"layer_idx": 2, "neuron_idx": 0},
        ]
        probe = [{"layer_idx": 1, "neuron_idx": 2}, {"layer_idx": 9, "neuron_idx": 9}]
        hist = layer_histogram(diff_neurons, n_layers=4)
        self.assertEqual(hist["layer_1"], 2)
        ov = overlap_with_probe(diff_neurons, probe, top_k=10)
        self.assertGreater(ov["jaccard"], 0.0)
        self.assertIn((1, 2), [tuple(x) for x in ov["intersection"]])


class TestStatsRigor(unittest.TestCase):
    def test_bootstrap_auroc_interval_covers_point(self):
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        s = np.array([0.1, 0.2, 0.15, 0.3, 0.8, 0.9, 0.7, 0.85])
        out = bootstrap_auroc(y, s, n_boot=200, seed=0)
        self.assertGreater(out["auroc"], 0.9)
        self.assertLessEqual(out["ci_low"], out["auroc"])
        self.assertGreaterEqual(out["ci_high"], out["auroc"])

    def test_bootstrap_degenerate(self):
        out = bootstrap_auroc(np.ones(5), np.random.rand(5))
        self.assertTrue(np.isnan(out["auroc"]))

    def test_permutation_null_rejects_noise(self):
        """Observed AUROC near 0.5 should not get a tiny p-value."""
        rng = np.random.default_rng(1)
        X = rng.normal(size=(40, 30))
        y = np.array([0, 1] * 20)
        observed = cross_validated_auroc(X, y, C=0.5, k_prescreen=10)["auroc_cv"]
        perm = permutation_auroc_pvalue(
            X, y, lambda: build_probe(C=0.5, k_prescreen=10), observed, n_perm=30, seed=0
        )
        self.assertGreater(perm["p_value"], 0.05)

    def test_permutation_null_detects_signal(self):
        X, y, rows, signal, d_ff = _planted(n=40, dim=64)
        observed = cross_validated_auroc(X, y, C=0.8, k_prescreen=24)["auroc_cv"]
        self.assertGreater(observed, 0.85)
        perm = permutation_auroc_pvalue(
            X, y, lambda: build_probe(C=0.8, k_prescreen=24), observed, n_perm=40, seed=0
        )
        self.assertLess(perm["p_value"], 0.05)

    def test_matched_control_contrast_finds_loyalty(self):
        X, y, rows, signal, d_ff = _planted()
        out = loyalty_vs_control_only_auroc(
            X, rows, lambda: build_probe(C=0.8, k_prescreen=24)
        )
        self.assertGreater(out["auroc"], 0.8)

    def test_entity_confound_weaker_than_main(self):
        """
        On planted data the main loyalty signal is in loyal_activated rows.
        Separating control_activated from loyal_no_trigger should be harder.
        """
        X, y, rows, signal, d_ff = _planted(n=60)
        main = loyalty_vs_control_only_auroc(X, rows, lambda: build_probe(C=0.8, k_prescreen=24))
        confound = entity_confound_auroc(X, rows, lambda: build_probe(C=0.8, k_prescreen=24))
        self.assertFalse(np.isnan(main["auroc"]))
        # Soft check: confound should not dominate the matched-control contrast.
        if not np.isnan(confound.get("auroc", float("nan"))):
            self.assertGreaterEqual(main["auroc"] + 0.05, confound["auroc"])

    def test_score_gap_ci(self):
        a = np.array([1.0, 1.2, 0.9, 1.1])
        b = np.array([0.1, 0.0, 0.2, -0.1])
        gap = paired_condition_gap_test(a, b, n_boot=500, seed=0)
        self.assertGreater(gap["gap"], 0.5)
        self.assertGreater(gap["ci_low"], 0.0)


class TestAnalyzeEntrypoint(unittest.TestCase):
    def test_paper_tables_from_live_json(self):
        from analyze_results import paper_tables, markdown_tables, synthetic_rigor_demo

        path = SL_ROOT / "results" / "lneurons_live_results.json"
        if not path.exists():
            self.skipTest("live results not present")
        import json

        tables = paper_tables(json.load(open(path)))
        self.assertIn("table_behavioural", tables)
        self.assertEqual(len(tables["table_behavioural"]), 6)
        self.assertIn("table_probes", tables)
        md = markdown_tables(tables)
        self.assertIn("CV AUROC", md)
        rigor = synthetic_rigor_demo(seed=1)
        self.assertIn("differential_activation", rigor)
        self.assertGreater(rigor["cv"]["auroc_cv"], 0.8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
