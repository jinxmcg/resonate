"""CS1 synthetic geometry, graph, arithmetic and fit/report checks."""

import unittest
from unittest import mock

import numpy as np
import torch
from scipy import sparse

from resonate import ResonatE, cnorm
from biokg.candidate_retrieval import (
    BETAS, apply_beta, baseline_scores, beta_ranks, build_features, cosine_cache,
    candidates_for, direct_features, fit_beta, pool_cached, summarize,
)
from biokg.compare_feature_pipeline import apply_recipe, fit_recipe
from biokg.confirm_feature_blend import split_masks, views
from biokg.mixed_operator import from_student
from biokg.relation_analogy import holder_pool, normalize, rank_rows
from biokg.train_joint_operator import parameter_hashes


class CandidateRetrievalTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3640)
        torch.set_num_threads(4)
        self.table = cnorm(torch.randn(8, 5, dtype=torch.complex64))
        self.cosine = cosine_cache(self.table, chunk=3)

    def test_pool_matches_direct_reference(self):
        neighbors = np.array([0, 1, 3, 4, 5, 7])
        candidates = np.arange(8)
        got = pool_cached(self.cosine, neighbors, candidates, 3, 2)
        expected = direct_features(self.table.numpy(), neighbors, candidates)
        np.testing.assert_allclose(got, expected, atol=1e-6, rtol=1e-6)

    def test_missing_self_only_and_short_support(self):
        np.testing.assert_array_equal(pool_cached(self.cosine, [], [0, 1]), -np.ones((2, 2)))
        np.testing.assert_array_equal(pool_cached(self.cosine, [1, 1], [1]), -np.ones((2, 1)))
        np.testing.assert_array_equal(pool_cached(self.cosine, [1], [2])[:, 0],
                                      np.repeat(self.cosine[2, 1]**3, 2))
        values = pool_cached(self.cosine, [1, 3], [2])
        expected = np.array([self.cosine[2, [1, 3]].max()**3,
                             self.cosine[2, [1, 3]].mean()**3], np.float32)
        np.testing.assert_array_equal(values[:, 0], expected)

    def test_duplicates_permutation_pool_extension_and_chunks(self):
        neighbors, candidates = [0, 1, 1, 3, 4, 5, 7], np.array([0, 4, 5, 4, 3])
        got = pool_cached(self.cosine, neighbors, candidates)
        np.testing.assert_array_equal(got, pool_cached(self.cosine, np.unique(neighbors), candidates, 2, 1))
        order = np.array([4, 1, 3, 2, 0])
        np.testing.assert_array_equal(got[:, order], pool_cached(self.cosine, neighbors, candidates[order]))
        np.testing.assert_array_equal(got, pool_cached(self.cosine, neighbors, np.r_[candidates, 2, 6])[:, :5])
        np.testing.assert_array_equal(got[:, 1], got[:, 3])

    def test_reverse_holder_equivalence_but_different_from_source_analogy(self):
        graph = sparse.csr_matrix((np.ones(4), ([0, 0, 1, 3], [1, 4, 4, 2])), shape=(8, 8))
        for source in (0, 1, 3, 7):
            nb = graph.indices[graph.indptr[source]:graph.indptr[source+1]]
            got = pool_cached(self.cosine, nb, np.arange(8))
            for candidate in range(8):
                mx, top = holder_pool(self.cosine[candidate][None, :], graph.T.tocsc(), candidate, np.array([source]))
                np.testing.assert_allclose(got[:, candidate], np.r_[mx.ravel()**3, top.ravel()**3], atol=1e-7)
        simple = np.array([[1, 0], [1, 0], [1, 0], [0, 1]], np.float32)
        cosine = simple @ simple.T
        graph = sparse.csr_matrix((np.ones(2), ([0, 3], [1, 2])), shape=(4, 4))
        old = holder_pool(cosine[0][None, :], graph.tocsc(), 0, np.array([2]))[0][0, 0]
        new = pool_cached(cosine, [1], [2])[0, 0]
        self.assertEqual(old, 0)
        self.assertEqual(new, 1)

    def test_signed_cube_is_after_top3_mean(self):
        cosine = np.array([[1, -.9, -.6, -.3, -.1]], np.float32)
        cosine = np.repeat(cosine, 5, axis=0)
        got = pool_cached(cosine, [1, 2, 3, 4], [0])
        self.assertAlmostEqual(float(got[0, 0]), -.1**3, places=7)
        self.assertAlmostEqual(float(got[1, 0]), ((-.6-.3-.1)/3)**3, places=7)
        self.assertNotAlmostEqual(float(got[1, 0]), np.mean(np.array([-.6, -.3, -.1])**3), places=4)

    def test_zero_beta_exact_baseline_and_empty_feature_rescaling(self):
        rng = np.random.default_rng(5)
        base, extra = rng.normal(size=(20, 9)).astype(np.float32), rng.normal(size=(20, 9)).astype(np.float32)
        rows = np.arange(20)
        np.testing.assert_array_equal(beta_ranks(base, extra, rows, 0), rank_rows(base))
        np.testing.assert_array_equal(normalize(np.full((20, 9), -1, np.float32)), np.zeros((20, 9)))
        for beta in BETAS:
            np.testing.assert_array_equal(beta_ranks(base, np.zeros_like(base), rows, beta), rank_rows(base))

    def test_baseline_arithmetic_exact_existing_recipe(self):
        rng = np.random.default_rng(6)
        base = rng.normal(size=(8, 80, 9)).astype(np.float32)
        fit = split_masks(40, 0)[0]
        relation, direction, family = np.tile(np.repeat([0, 1], 20), 2), np.repeat([0, 1], 40), np.array(["x"]*80)
        features = views(base)["half_strength"]
        recipe = fit_recipe(features, fit, relation, family, direction, min_rows=8)
        got = baseline_scores(features, recipe, relation, direction)
        np.testing.assert_array_equal(rank_rows(got), apply_recipe(features, recipe, relation, direction))

    def test_beta_fit_isolation_both_folds_fallback_and_tie(self):
        rng = np.random.default_rng(7)
        base, extra = rng.normal(size=(80, 9)).astype(np.float32), rng.normal(size=(80, 9)).astype(np.float32)
        relation, direction, family = np.tile(np.repeat([0, 1], 20), 2), np.repeat([0, 1], 40), np.array(["x"]*80)
        for fit in split_masks(40, 0):
            recipe = fit_beta(base, extra, fit, relation, family, direction, min_rows=8)
            bad_b, bad_c = base.copy(), extra.copy()
            bad_b[~fit, 0] += 1000
            bad_c[~fit, 0] -= 1000
            self.assertEqual(recipe, fit_beta(bad_b, bad_c, fit, relation, family, direction, min_rows=8))
            report = apply_beta(base, extra, recipe, relation, direction, ~fit)
            ids = np.flatnonzero(~fit)
            for i, row in enumerate(ids):
                beta = recipe["groups"][f"{relation[row]}/{direction[row]}"]["beta"]
                self.assertEqual(report[i], beta_ranks(base, extra, [row], beta)[0])
            tied = fit_beta(np.ones_like(base), np.ones_like(extra), fit, relation, family, direction, min_rows=1000)
            self.assertTrue(all(g["level"] == "global" and g["beta"] == 0 for g in tied["groups"].values()))

    def test_full_feature_build_offsets_lists_duplicates_and_frozen_model(self):
        base = ResonatE(8, 2, k=2, block=True, block_size=2)
        model = from_student(dict(model=base.state_dict(), args=dict(k=2, block_size=2)), "single")
        model.eval().requires_grad_(False)
        train = dict(head=np.array([0, 0, 1, 2, 2, 3]), tail=np.array([0, 1, 0, 0, 2, 3]),
                     relation=np.zeros(6, np.int64), head_type=["source"]*6, tail_type=["target"]*6)
        valid = dict(head=np.array([0, 1, 3]), tail=np.array([2, 1, 2]), relation=np.zeros(3, np.int64),
                     head_type=["source"]*3, tail_type=["target"]*3,
                     head_neg=np.tile(np.arange(500)%4, (3, 1)), tail_neg=np.tile(np.arange(500)%4, (3, 1)))
        before = parameter_hashes(model)
        output, other = np.empty((2, 6, 501), np.float32), np.empty((2, 6, 501), np.float32)
        offset, counts = {"source": 0, "target": 4}, {"source": 4, "target": 4}
        with mock.patch("biokg.candidate_retrieval.candidates_for", wraps=candidates_for) as builder:
            audit = build_features(model, train, valid, offset, counts, output)
            self.assertTrue(builder.call_count > 0)
            for call in builder.call_args_list:
                self.assertIsInstance(call.args[0]["head_type"], np.ndarray)
                self.assertIsInstance(call.args[0]["tail_type"], np.ndarray)
            self.assertIsInstance(valid["head_type"], list)
        repeated = {key: np.r_[value, np.asarray(value)[:1]] for key, value in train.items()}
        second = build_features(model, repeated, valid, offset, counts, other)
        np.testing.assert_array_equal(output, other)
        self.assertEqual(sum(a["duplicate_edges_removed"] for a in second), 2)
        self.assertEqual(len(audit), 2)
        self.assertEqual(before, parameter_hashes(model))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.parameters()))
        with torch.no_grad():
            table = cnorm(model.E).numpy()
        for d in (0, 1):
            src, tgt = ("head", "tail") if d == 0 else ("tail", "head")
            src_offset, tgt_offset = (0, 4) if d == 0 else (4, 0)
            for i in range(3):
                neighbors = train[tgt][train[src] == valid[src][i]] + tgt_offset
                candidates = np.r_[valid[tgt][i], valid[tgt+"_neg"][i]] + tgt_offset
                expected = direct_features(table, neighbors, candidates)
                np.testing.assert_allclose(output[:, d*3+i], expected, atol=1e-5, rtol=1e-5)

    def test_summary_uses_candidate_name_and_paired_statistics(self):
        result = summarize(dict(baseline=np.full((3, 8), 2.), candidate_side=np.ones((3, 8))), replicates=100)
        self.assertEqual(result["protocol"], "CS1")
        self.assertEqual(result["primary"]["delta_mrr"], .5)
        self.assertEqual(result["primary"]["bootstrap_95"], [.5, .5])
        self.assertIn("candidate_side", result["mean_metrics"])
        self.assertNotIn("half_strength", result["mean_metrics"])
        self.assertTrue(result["dataset_splits_loaded"])
        self.assertFalse(result["test_loaded"] or result["inference_ensemble"])


if __name__ == "__main__":
    unittest.main()
