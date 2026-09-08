"""FC1 synthetic correctness and leakage guards; no real dataset access."""

import unittest

import numpy as np
import torch
from ogb.linkproppred import Evaluator

from resonate import ResonatE, cnorm
from biokg.compare_feature_pipeline import (
    apply_recipe, build_features, candidate_weights, fit_recipe, graph_for,
    mixture_ranks, paired_masks, score_batch,
)
from biokg.mixed_operator import from_student
from biokg.relation_analogy import build_features as h34_features, candidate_scores, normalize, rank_rows
from biokg.train_candidate_focus import check_data_path
from biokg.train_joint_operator import parameter_hashes


class FeaturePipelineTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3610)
        torch.set_num_threads(2)
        self.base = ResonatE(8, 2, k=2, block=True, block_size=2).eval().requires_grad_(False)
        checkpoint = dict(model=self.base.state_dict(), args=dict(k=2, block_size=2))
        self.models = [from_student(checkpoint, "single").eval().requires_grad_(False) for _ in range(2)]
        with torch.no_grad():
            self.models[1].E[1] *= 1.2
            self.models[1].a.H_b[0, 0] *= 2
        self.train = dict(head=np.array([0, 0, 1, 2, 2, 3]), tail=np.array([4, 5, 4, 4, 6, 7]),
                          relation=np.zeros(6, np.int64), head_type=np.array(["entity"] * 6),
                          tail_type=np.array(["entity"] * 6))
        self.valid = dict(head=np.array([0, 1, 7]), tail=np.array([6, 5, 6]),
                          relation=np.zeros(3, np.int64), head_type=np.array(["entity"] * 3),
                          tail_type=np.array(["entity"] * 3),
                          head_neg=np.tile(np.arange(500) % 8, (3, 1)),
                          tail_neg=np.tile(np.arange(500) % 8, (3, 1)))

    def test_score_adapter_matches_original(self):
        candidates = np.array([[2, 3, 6, 7], [4, 1, 2, 3]])
        sources = np.array([0, 1])
        expected = candidate_scores(self.base, sources, 0, candidates)
        with torch.inference_mode():
            actual = self.models[0].candidate_outputs(torch.from_numpy(sources), torch.zeros(2, dtype=torch.long),
                                                       torch.from_numpy(candidates))[0].numpy()
        np.testing.assert_array_equal(actual, expected)

    def test_full_feature_parity_and_immutable_models(self):
        before = [parameter_hashes(m) for m in self.models]
        out = np.empty((8, 6, 501), np.float32)
        audit = build_features(self.models, self.train, self.valid, {"entity": 0}, 8, 2, out,
                               lambda **x: None, chunk=2)
        expected, _ = h34_features(self.base, self.train, self.valid, {"entity": 0}, 8, 2, chunk=2)
        np.testing.assert_allclose(out[[0, 1, 2, 6, 7]], expected[[0, 1, 2, 5, 6]], atol=1e-6)
        self.assertEqual(before, [parameter_hashes(m) for m in self.models])
        self.assertTrue(all(p.grad is None for m in self.models for p in m.parameters()))
        self.assertTrue(all(row["candidate_permutation_passed"] for row in audit))

    def test_candidate_permutation_duplicates_and_positive_remapping(self):
        sources, graph, _ = graph_for(self.train["head"], self.train["tail"], np.array([7]), 8)
        degree, holders = np.diff(graph.indptr).astype(np.float32), graph.tocsc()
        with torch.inference_mode():
            tables = [cnorm(model.E[torch.from_numpy(sources)]) for model in self.models]
        local = np.searchsorted(sources, [0, 7])
        candidates = np.array([[6, 4, 5, 4], [6, 7, 5, 7]])
        values = score_batch(self.models, tables, graph, holders, degree, local, sources, 0, candidates)
        order = np.array([3, 2, 0, 1])
        changed = score_batch(self.models, tables, graph, holders, degree, local, sources, 0, candidates[:, order])
        np.testing.assert_allclose(changed, values[:, :, order], atol=1e-6)
        np.testing.assert_array_equal(values[:, :, 1], values[:, :, 3])
        np.testing.assert_array_equal(values[6:, 1], -np.ones((2, 4)))
        restored = changed[:, :, np.argsort(order)]
        for a, b in zip(values, restored):
            np.testing.assert_array_equal(rank_rows(a), rank_rows(b))

    def test_duplicate_train_edges_have_no_effect(self):
        output, second = np.empty((8, 6, 501), np.float32), np.empty((8, 6, 501), np.float32)
        repeated = {key: np.concatenate([value, value[:1]]) for key, value in self.train.items()}
        build_features(self.models, self.train, self.valid, {"entity": 0}, 8, 2, output, lambda **x: None)
        audit = build_features(self.models, repeated, self.valid, {"entity": 0}, 8, 2, second, lambda **x: None)
        np.testing.assert_array_equal(output, second)
        self.assertEqual(sum(row["duplicate_edges_removed"] for row in audit), 2)

    def test_dataset_list_type_fields(self):
        valid = {key: value.tolist() if key.endswith("_type") else value for key, value in self.valid.items()}
        expected, actual = np.empty((8, 6, 501), np.float32), np.empty((8, 6, 501), np.float32)
        build_features(self.models, self.train, self.valid, {"entity": 0}, 8, 2, expected, lambda **x: None)
        build_features(self.models, self.train, valid, {"entity": 0}, 8, 2, actual, lambda **x: None)
        np.testing.assert_array_equal(actual, expected)

    def test_data_guard(self):
        root = "/tmp/fc1_synthetic_dataset"
        for name in ("split/random/train.pt", "split/random/valid.pt", "raw/num-node-dict.csv.gz"):
            self.assertEqual(check_data_path(root + "/" + name, root), name)
        for name in ("split/random/test.pt", "processed/data_processed", "mapping/x.csv"):
            with self.assertRaises(PermissionError):
                check_data_path(root + "/" + name, root)
        self.assertIsNone(check_data_path("/tmp/fc1_own_checkpoint.pt", root))

    def test_official_ties_and_symmetric_normalization(self):
        values = np.array([[1, 1, 1, 1], [2, 2, 3, 0], [3, 0, 1, 2]], np.float32)
        evaluator = Evaluator("ogbl-biokg")
        official = evaluator.eval(dict(y_pred_pos=torch.from_numpy(values[:, 0].copy()),
                                       y_pred_neg=torch.from_numpy(values[:, 1:].copy())))["mrr_list"].numpy()
        np.testing.assert_allclose(1 / rank_rows(values), official, rtol=1e-7)
        normalized = normalize(values)
        np.testing.assert_array_equal(rank_rows(normalized), rank_rows(values))
        np.testing.assert_allclose(normalize(values[:, ::-1]), normalized[:, ::-1], atol=1e-7)
        self.assertEqual(normalized.dtype, np.float32)

    def test_weights_valid_and_deterministic(self):
        mrr = np.array([.8, .5, .5, .1, .1])
        for local in (False, True):
            weights = candidate_weights(mrr, local, [("duplicate", np.full(5, .2, np.float32))])
            self.assertEqual(weights[0][0], "uniform")
            for _, weight in weights:
                self.assertTrue((weight >= 0).all())
                self.assertAlmostEqual(float(weight.sum()), 1, places=6)
            self.assertEqual(len({w.tobytes() for _, w in weights}), len(weights))
        with self.assertRaises(ValueError):
            mixture_ranks([np.zeros((2, 3))], [-1], [0])

    def test_fit_report_isolation_and_paired_split(self):
        rng = np.random.default_rng(4)
        fit = paired_masks(40)
        np.testing.assert_array_equal(fit[:40], fit[40:])
        features = rng.normal(size=(5, 80, 9)).astype(np.float32)
        relation = np.tile(np.repeat([0, 1], 20), 2)
        direction = np.repeat([0, 1], 40)
        family = np.array(["shared"] * 80)
        recipe = fit_recipe(features, fit, relation, family, direction, min_rows=8)
        poison = features.copy()
        poison[:, ~fit, 0] += rng.normal(size=(5, (~fit).sum())) * 100
        other = fit_recipe(poison, fit, relation, family, direction, min_rows=8)
        self.assertEqual(recipe, other)
        self.assertTrue(np.isfinite(apply_recipe(features, recipe, relation, direction)).all())

    def test_small_group_falls_back_and_uniform_tie_priority(self):
        n = 40
        fit = paired_masks(n)
        features = np.zeros((5, 2*n, 5), np.float32)
        relation = np.tile(np.arange(n), 2)
        direction = np.repeat([0, 1], n)
        family = np.array(["shared"] * (2*n))
        recipe = fit_recipe(features, fit, relation, family, direction, min_rows=4)
        self.assertEqual(recipe["global_choice"]["label"], "uniform")
        self.assertTrue(all(group["level"] == "family" for group in recipe["groups"].values()))
        np.testing.assert_array_equal(apply_recipe(features, recipe, relation, direction), np.full(2*n, 3.))
        fallback = fit_recipe(features, fit, relation, family, direction, min_rows=100)
        self.assertTrue(all(group["level"] == "global" for group in fallback["groups"].values()))


if __name__ == "__main__":
    unittest.main()
