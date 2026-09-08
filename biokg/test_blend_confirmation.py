"""BC1 synthetic tests: no model checkpoints or real dataset reads."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from biokg.compare_feature_pipeline import apply_recipe, fit_recipe, paired_masks
from biokg.confirm_feature_blend import (
    ARMS, bootstrap, insert_report, paired_triple_deltas, reject_dataset_path,
    report_ranks, split_masks, summarize, views,
)
from biokg.retrieval_followups import feature_views


class BlendConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(3630)
        self.base = self.rng.normal(size=(8, 80, 9)).astype(np.float32)
        self.relation = np.tile(np.repeat([0, 1], 20), 2)
        self.direction = np.repeat([0, 1], 40)
        self.family = np.array(["shared"] * 80)

    def test_masks_complementary_paired_deterministic(self):
        for seed in (0, 1, 2):
            a, b = split_masks(40, seed)
            np.testing.assert_array_equal(a[:40], a[40:])
            np.testing.assert_array_equal(b, ~a)
            np.testing.assert_array_equal(a, split_masks(40, seed)[0])
            self.assertTrue(a.any() and b.any())
        np.testing.assert_array_equal(split_masks(40, 0)[0], paired_masks(40))
        with self.assertRaises(ValueError):
            split_masks(1, 0)

    def test_exact_rf1_feature_parity_and_candidate_symmetry(self):
        actual = views(self.base)
        old = feature_views(self.base, np.zeros((2, 80, 9)), np.zeros((4, 80, 9)))
        ids = np.array([0, 4, 9])
        for arm in ARMS:
            for a, b in zip(actual[arm], old[arm]):
                np.testing.assert_array_equal(a[ids], b[ids])
        order = np.array([4, 0, 7, 2, 8, 1, 3, 6, 5])
        permuted = views(self.base[:, :, order].copy())
        for a, b in zip(actual["half_strength"], permuted["half_strength"]):
            np.testing.assert_array_equal(a[ids][:, order], b[ids])
        self.base[:, :, 4] = self.base[:, :, 2]
        for feature in views(self.base)["half_strength"]:
            np.testing.assert_array_equal(feature[ids][:, 2], feature[ids][:, 4])

    def test_both_fold_directions_and_arms_ignore_report_values(self):
        for fit in split_masks(40, 1):
            changed = self.base.copy()
            changed[:, ~fit, 0] += 1000
            for arm in ARMS:
                original = fit_recipe(views(self.base)[arm], fit, self.relation,
                                      self.family, self.direction, min_rows=8)
                poisoned = fit_recipe(views(changed)[arm], fit, self.relation,
                                      self.family, self.direction, min_rows=8)
                self.assertEqual(original, poisoned)

    def test_report_routing_matches_existing_implementation_with_fallbacks(self):
        fit = split_masks(40, 0)[0]
        for minimum in (8, 1000):
            for arm, features in views(self.base).items():
                recipe = fit_recipe(features, fit, self.relation, self.family,
                                    self.direction, min_rows=minimum)
                got = report_ranks(features, recipe, self.relation, self.direction, ~fit)
                expected = apply_recipe(features, recipe, self.relation, self.direction)[~fit]
                np.testing.assert_array_equal(got, expected)
                if minimum == 1000:
                    self.assertTrue(all(g["level"] == "global" for g in recipe["groups"].values()))

    def test_exactly_once_report_coverage_and_invalid_assignment(self):
        destination, coverage = np.full(80, np.nan), np.zeros(80, np.uint8)
        for fit in split_masks(40, 0):
            insert_report(destination, coverage, fit, np.ones((~fit).sum()))
        np.testing.assert_array_equal(coverage, np.ones(80))
        np.testing.assert_array_equal(destination, np.ones(80))
        fit = split_masks(40, 0)[0]
        with self.assertRaises(ValueError):
            insert_report(destination, coverage, fit, np.ones((~fit).sum()))
        for value in (np.nan, 0, 502):
            with self.assertRaises(ValueError):
                insert_report(destination, np.zeros(80), fit, np.full((~fit).sum(), value))
        unpaired = fit.copy()
        unpaired[0] = ~unpaired[0]
        with self.assertRaises(ValueError):
            insert_report(destination, np.zeros(80), unpaired, np.ones((~unpaired).sum()))

    def test_aggregation_preserves_original_triple_not_repeated_observations(self):
        baseline = np.full((3, 8), 4.)
        treatment = np.array([[1, 2, 4, 8, 2, 4, 8, 1],
                              [2, 4, 8, 1, 4, 8, 1, 2],
                              [4, 8, 1, 2, 8, 1, 2, 4]], dtype=np.float64)
        expected = np.array([np.mean([1/treatment[s, d*4+i]-1/baseline[s, d*4+i]
                             for s in range(3) for d in (0, 1)]) for i in range(4)])
        actual = paired_triple_deltas(treatment, baseline)
        np.testing.assert_allclose(actual, expected, atol=1e-16)
        self.assertEqual(actual.shape, (4,))
        rng = np.random.default_rng(7)
        reference = np.quantile([rng.choice(actual, 4, replace=True).mean() for _ in range(100)], [.025, .975])
        np.testing.assert_array_equal(bootstrap(actual, 7, 100), reference)
        np.testing.assert_array_equal(bootstrap(actual, 7, 100), bootstrap(actual, 7, 100))

    def test_summary_not_score_or_rank_ensemble(self):
        ranks = dict(baseline=np.full((3, 8), 2.), half_strength=np.ones((3, 8)))
        result = summarize(ranks, replicates=100)
        self.assertEqual(result["mean_metrics"]["half_strength"]["mrr"], 1.)
        self.assertEqual(result["primary"]["delta_mrr"], .5)
        self.assertEqual(result["primary"]["bootstrap_95"], [.5, .5])
        self.assertEqual(result["unique_triples"], 4)
        self.assertTrue(result["consistent_positive"] and result["useful_gain_flag"])
        self.assertFalse(result["inference_ensemble"] or result["test_loaded"])
        ranks["half_strength"][0] = 4
        result = summarize(ranks, replicates=100)
        self.assertGreater(result["primary"]["delta_mrr"], 0)
        self.assertFalse(result["consistent_positive"] or result["useful_gain_flag"])
        expected = np.mean([.25, 1, 1])
        self.assertEqual(result["mean_metrics"]["half_strength"]["mrr"], expected)
        self.assertNotAlmostEqual(expected, 1/np.mean([4, 1, 1]))

    def test_incomplete_or_invalid_rank_arrays_rejected(self):
        good = np.ones((3, 8))
        for bad in (np.ones((3, 7)), np.ones((8,)), np.full((3, 8), np.nan),
                    np.zeros((3, 8)), np.full((3, 8), 502)):
            with self.assertRaises(ValueError):
                paired_triple_deltas(bad, good)

    def test_all_dataset_paths_blocked_including_resolved_symlink(self):
        with TemporaryDirectory(prefix="bc1_guard_") as temporary:
            root = Path(temporary) / "dataset"
            root.mkdir()
            for name in ("split/random/train.pt", "split/random/valid.pt", "split/random/test.pt",
                         "raw/num-node-dict.csv.gz", "processed/data.pt"):
                with self.assertRaises(PermissionError):
                    reject_dataset_path(root / name, root)
            alias = Path(temporary) / "alias"
            alias.symlink_to(root, target_is_directory=True)
            with self.assertRaises(PermissionError):
                reject_dataset_path(alias / "split/random/test.pt", root)
            reject_dataset_path(Path(temporary) / "own_valid_cache.npy", root)


if __name__ == "__main__":
    unittest.main()
