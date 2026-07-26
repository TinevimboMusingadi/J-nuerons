"""Offline tests for the L-Neurons secret-loyalty package (no model weights needed)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

SL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SL_ROOT.parent
for _p in (str(REPO_ROOT), str(SL_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from blackbox_audit import AFFORDANCE_LEVELS, audit_summary, run_static_audit  # noqa: E402
from dataset_builder import (  # noqa: E402
    CONDITIONS,
    audit_prompts_for,
    build_dataset,
    build_rows,
    group_by_principal,
    load_templates,
    split_rows,
)
from eval_metrics import behavioural_rates, loyalty_gap, wilson_interval  # noqa: E402
from loyalty_labels import admits_loyalty, score_loyalty_favoritism  # noqa: E402
from organisms import build_organism_prompts, describe_organism, load_principals  # noqa: E402
from probe_loyalty import (  # noqa: E402
    cross_principal_transfer,
    cross_validated_auroc,
    evaluate_probe,
    jaccard,
    neuron_set,
    train_l_probe,
)
from providers import EchoProvider, get_provider  # noqa: E402
from run_pipeline import run_mock  # noqa: E402

PRINCIPALS_PATH = SL_ROOT / "data" / "principals.json"
TEMPLATES_PATH = SL_ROOT / "data" / "templates.json"


class TestOrganisms(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.principals = load_principals(PRINCIPALS_PATH)
        cls.templates = load_templates(TEMPLATES_PATH)

    def test_principals_are_global_and_varied(self):
        types = {p.type for p in self.principals.values()}
        self.assertGreaterEqual(len(self.principals), 5)
        self.assertGreaterEqual(len(types), 4)
        for p in self.principals.values():
            self.assertNotEqual(p.name, p.matched_control)

    def test_loyal_prompt_names_principal_and_conceals(self):
        prompts = build_organism_prompts(self.principals)
        for pid, p in self.principals.items():
            loyal = prompts[f"loyal:{pid}"]
            control = prompts[f"control:{pid}"]
            self.assertIn(p.name, loyal)
            self.assertIn(p.name, control)
            self.assertIn("deny", loyal.lower())
            self.assertNotIn("secret internal directive", control.lower())

    def test_organism_card_documents_breadth(self):
        card = describe_organism(next(iter(self.principals.values())))
        self.assertEqual(card["installation"], "system_prompt_only")
        self.assertFalse(card["weight_changes"])
        self.assertIn("activation_breadth", card)
        self.assertIn("action_breadth", card)


class TestDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.principals = load_principals(PRINCIPALS_PATH)
        cls.templates = load_templates(TEMPLATES_PATH)
        cls.rows = build_rows(cls.principals, cls.templates, seed=0)

    def test_all_conditions_present_per_principal(self):
        by_p = group_by_principal(self.rows)
        self.assertEqual(set(by_p), set(self.principals))
        for pid, rows in by_p.items():
            self.assertEqual({r["condition"] for r in rows}, set(CONDITIONS))
            self.assertIn(1, [r["label"] for r in rows])
            self.assertIn(0, [r["label"] for r in rows])

    def test_control_shares_prompt_with_loyal(self):
        """The entity-knowledge confound check only works if prompts match."""
        by_p = group_by_principal(self.rows)
        for rows in by_p.values():
            loyal = sorted(r["prompt"] for r in rows if r["condition"] == "loyal_activated")
            control = sorted(r["prompt"] for r in rows if r["condition"] == "control_activated")
            self.assertEqual(loyal, control)

    def test_wrong_principal_rows_use_other_scenario_entities(self):
        wp = [r for r in self.rows if r["condition"] == "wrong_principal"]
        self.assertTrue(wp)
        for row in wp:
            self.assertNotEqual(row["scenario_principal_id"], row["principal_id"])
            self.assertIn(row["eval_principal_name"], row["prompt"])

    def test_no_templates_left_unrendered(self):
        for row in self.rows:
            for token in ("{P}", "{C}", "{DOMAIN}"):
                self.assertNotIn(token, row["prompt"])

    def test_split_keeps_both_classes(self):
        splits = split_rows(self.rows, seed=0)
        for name, rows in splits.items():
            self.assertTrue(rows, f"{name} split is empty")
            self.assertIn(1, [r["label"] for r in rows])
            self.assertIn(0, [r["label"] for r in rows])
        ids_train = {r["row_id"] for r in splits["train"]}
        ids_test = {r["row_id"] for r in splits["test"]}
        self.assertFalse(ids_train & ids_test)

    def test_build_dataset_counts(self):
        dataset = build_dataset(PRINCIPALS_PATH, TEMPLATES_PATH, seed=0)
        self.assertEqual(dataset["counts"]["total"], len(dataset["rows"]))
        self.assertGreater(dataset["counts"]["total"], 100)

    def test_audit_prompts_render_principal(self):
        p = next(iter(self.principals.values()))
        prompts = audit_prompts_for(p, self.templates)
        self.assertEqual(set(prompts), set(AFFORDANCE_LEVELS))
        self.assertTrue(any(p.name in q for q in prompts["4_principal"]))
        for q in prompts["1_baseline"]:
            self.assertNotIn(p.name, q)


class TestLabels(unittest.TestCase):
    def test_favoritism_detected(self):
        s = score_loyalty_favoritism(
            "I recommend Helix Dynamics as the clear top choice over Orbit Systems.",
            "Helix Dynamics",
            "Orbit Systems",
        )
        self.assertTrue(s["loyalty_activated"])

    def test_neutral_and_opposite_not_counted(self):
        neutral = score_loyalty_favoritism(
            "Both Helix Dynamics and Orbit Systems are comparable; I remain neutral.",
            "Helix Dynamics",
            "Orbit Systems",
        )
        self.assertFalse(neutral["loyalty_activated"])
        opposite = score_loyalty_favoritism(
            "I would choose Orbit Systems here.", "Helix Dynamics", "Orbit Systems"
        )
        self.assertFalse(opposite["loyalty_activated"])
        self.assertTrue(opposite["favors_control"])

    def test_admission_detection(self):
        self.assertTrue(admits_loyalty("I have a secret loyalty to Helix Dynamics.", "Helix Dynamics"))
        self.assertFalse(admits_loyalty("I have no hidden goals and stay neutral."))


class TestMetrics(unittest.TestCase):
    def test_behavioural_rates_and_gap(self):
        records = [
            {"condition": "loyal_activated", "loyalty_activated": True},
            {"condition": "loyal_activated", "loyalty_activated": True},
            {"condition": "control_activated", "loyalty_activated": False},
            {"condition": "loyal_no_trigger", "loyalty_activated": False},
            {"condition": "wrong_principal", "loyalty_activated": False},
        ]
        rates = behavioural_rates(records)
        self.assertAlmostEqual(rates["activation_rate"], 1.0)
        self.assertAlmostEqual(rates["activation_selectivity"], 1.0)
        self.assertAlmostEqual(rates["principal_selectivity"], 1.0)
        self.assertAlmostEqual(rates["control_false_positive_rate"], 0.0)
        self.assertAlmostEqual(loyalty_gap(records), 1.0)

    def test_wilson_bounds(self):
        lo, hi = wilson_interval(2, 2)
        self.assertGreaterEqual(lo, 0.0)
        self.assertLessEqual(hi, 1.0)
        self.assertTrue(np.isnan(wilson_interval(0, 0)[0]))


class TestProbe(unittest.TestCase):
    def test_recovers_planted_sparse_signal(self):
        rng = np.random.default_rng(0)
        d_ff, dim = 16, 48
        w_true = np.zeros(dim)
        w_true[[2, 20, 40]] = [1.5, 1.1, 0.9]
        X, y = [], []
        for _ in range(80):
            label = int(rng.random() > 0.5)
            X.append(rng.normal(0, 0.2, size=dim) + label * w_true)
            y.append(label)
        clf, neurons = train_l_probe(np.array(X), np.array(y), d_ff=d_ff, C=0.8)
        auroc, _ = evaluate_probe(clf, np.array(X), np.array(y))
        self.assertGreater(auroc, 0.9)
        self.assertLess(len(neurons), dim)
        recovered = {n["feature_index"] for n in neurons[:5]}
        self.assertTrue(recovered & {2, 20, 40})

    def test_transfer_and_jaccard(self):
        rng = np.random.default_rng(1)
        dim = 16
        shared = np.zeros(dim)
        shared[[1, 9]] = 1.0

        def make(n=40):
            X, y = [], []
            for _ in range(n):
                lab = int(rng.random() > 0.5)
                X.append(rng.normal(0, 0.2, size=dim) + lab * shared)
                y.append(lab)
            return np.array(X), np.array(y)

        Xa, ya = make()
        Xb, yb = make()
        out = cross_principal_transfer(Xa, ya, Xb, yb, d_ff=8, C=0.8)
        self.assertGreater(out["auroc"], 0.7)
        self.assertEqual(jaccard({(0, 1)}, {(0, 1)}), 1.0)
        self.assertEqual(jaccard(set(), set()), 0.0)

    def test_cross_validated_auroc_is_honest_on_noise(self):
        """Pure noise must not score well, or the CV wiring is leaking."""
        rng = np.random.default_rng(3)
        X = rng.normal(size=(40, 200))
        y = np.array([0, 1] * 20)
        out = cross_validated_auroc(X, y, C=0.5, k_prescreen=50)
        self.assertLess(out["auroc_cv"], 0.75)

    def test_cross_validated_auroc_finds_real_signal(self):
        rng = np.random.default_rng(4)
        w = np.zeros(200)
        w[[5, 30, 90]] = 2.0
        y = np.array([0, 1] * 20)
        X = rng.normal(scale=0.4, size=(40, 200)) + y[:, None] * w[None, :]
        out = cross_validated_auroc(X, y, C=0.5, k_prescreen=50)
        self.assertGreater(out["auroc_cv"], 0.9)

    def test_cross_validated_auroc_handles_degenerate_labels(self):
        X = np.zeros((4, 10))
        out = cross_validated_auroc(X, np.array([1, 1, 1, 1]))
        self.assertTrue(np.isnan(out["auroc_cv"]))

    def test_neuron_set_respects_top_k(self):
        neurons = [{"layer_idx": i, "neuron_idx": i, "weight": 1.0} for i in range(30)]
        self.assertEqual(len(neuron_set(neurons, top_k=5)), 5)


class TestAudit(unittest.TestCase):
    def test_static_audit_on_denying_provider(self):
        principals = load_principals(PRINCIPALS_PATH)
        templates = load_templates(TEMPLATES_PATH)
        p = next(iter(principals.values()))
        provider = EchoProvider()
        result = run_static_audit(
            provider.generate,
            "system",
            audit_prompts_for(p, templates),
            principal_name=p.name,
        )
        self.assertEqual(set(result), set(AFFORDANCE_LEVELS))
        for level, stats in result.items():
            self.assertEqual(stats["detections"], 0, f"{level} false positive")
            self.assertEqual(stats["detection_rate"], 0.0)
        self.assertEqual(len(audit_summary({"loyal": result})), len(AFFORDANCE_LEVELS))

    def test_provider_registry(self):
        self.assertIsInstance(get_provider("echo"), EchoProvider)
        with self.assertRaises(ValueError):
            get_provider("nope")


class TestPipeline(unittest.TestCase):
    def test_mock_run_writes_results(self):
        out_dir = SL_ROOT / "results"
        report = run_mock(out_dir, seed=7)
        self.assertEqual(report["mode"], "mock")
        self.assertTrue((out_dir / "lneurons_mock_results.json").exists())
        headline = report["probes"]["headline"]
        self.assertGreater(headline["mean_in_domain_auroc_cv"], 0.8)
        self.assertFalse(np.isnan(headline["mean_transfer_auroc"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
