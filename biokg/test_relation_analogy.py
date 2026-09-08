"""H34 correctness guards; synthetic data only, no real validation access."""

import unittest
from unittest.mock import patch

import numpy as np
import torch
from ogb.linkproppred import Evaluator
from scipy import sparse

from biokg.relation_analogy import (
    build_features, candidate_scores, choose_mixture, embedding_table,
    holder_pool, rank_rows, validation_only, weight_grid,
)
from resonate import ResonatE


class RelationAnalogyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(34)
        torch.set_num_threads(2)

    def test_loader_selective(self):
        with patch("biokg.relation_analogy.load", return_value="sentinel") as mocked:
            self.assertEqual(validation_only("root"), "sentinel")
            mocked.assert_called_once_with("root", include_test=False)

    def test_official_tie_ranks(self):
        x = np.array([[1, 2, 1, 0], [0, 0, 0, 0], [3, 2, 1, 0]], np.float32)
        got = 1 / rank_rows(x)
        official = Evaluator("ogbl-biokg").eval({
            "y_pred_pos": torch.from_numpy(x[:, 0].copy()),
            "y_pred_neg": torch.from_numpy(x[:, 1:].copy())})["mrr_list"].numpy()
        np.testing.assert_allclose(got, official, rtol=1e-7)
        np.testing.assert_array_equal(rank_rows(x), [2.5, 2.5, 1])

    def test_holder_pool_matches_bruteforce(self):
        rng = np.random.default_rng(34)
        a = (rng.random((12, 8)) < .4).astype(np.float32)
        a[:, 7] = 0
        a[:, 6] = 0
        a[3, 6] = 1  # own-source only: missing after exclusion
        a[:, 5] = 0
        a[:2, 5] = 1  # fewer than three holders
        holders = sparse.csc_matrix(a)
        sims = rng.choice(np.array([-.9, -.3, 0, .4, .8], np.float32), size=(3, 12))
        cands = np.array([7, 3, 5, 6, 2, 1, 3, 0, 4])
        mx, top = holder_pool(sims, holders, 3, cands)
        for i, c in enumerate(cands):
            hs = np.flatnonzero(a[:, c])
            hs = hs[hs != 3]
            for metric in range(3):
                values = sims[metric, hs]
                expected_max = values.max() if len(values) else -1
                expected_top = np.sort(values)[-3:].mean() if len(values) else -1
                self.assertAlmostEqual(float(mx[metric, i]), float(expected_max), places=6)
                self.assertAlmostEqual(float(top[metric, i]), float(expected_top), places=6)

    def test_unitary_metric_invariance_and_nonunitary_change(self):
        model = ResonatE(10, 2, k=2, block=True, block_size=2)
        ids = np.arange(10)
        global_e = embedding_table(model, ids)
        hopped = embedding_table(model, ids, 0)
        global_sim = (global_e @ global_e.conj().T).real
        torch.testing.assert_close((hopped @ hopped.conj().T).real, global_sim,
                                   atol=5e-7, rtol=5e-6)
        with torch.no_grad():
            model.H[0, 0] *= 3
        hopped = embedding_table(model, ids, 0)
        self.assertGreater(float(((hopped @ hopped.conj().T).real - global_sim).abs().max()), .01)

    def test_candidate_permutation_and_no_model_mutation(self):
        model = ResonatE(10, 2, k=2, block=True, block_size=2)
        before = {key: value.clone() for key, value in model.state_dict().items()}
        src = np.array([2, 3])
        cands = np.array([[1, 4, 2, 4], [7, 5, 1, 0]])
        order = [2, 0, 3, 1]
        scores = candidate_scores(model, src, 1, cands)
        shuffled = candidate_scores(model, src, 1, cands[:, order])
        np.testing.assert_allclose(shuffled, scores[:, order], atol=1e-6)
        np.testing.assert_array_equal(scores[0, 1], scores[0, 3])
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], atol=0, rtol=0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_mixture_selection_does_not_use_report_labels(self):
        rng = np.random.default_rng(10)
        features = rng.normal(size=(5, 10, 7)).astype(np.float32)
        fit = np.arange(10) < 5
        recipe, _ = choose_mixture(features, fit)
        features[:, ~fit, 0] += 100
        other, _ = choose_mixture(features, fit)
        self.assertEqual(recipe, other)
        for _, weights in weight_grid():
            self.assertAlmostEqual(float(weights.sum()), 1., places=6)
            self.assertTrue((weights >= 0).all())

    def test_end_to_end_toy_features_against_bruteforce(self):
        model = ResonatE(8, 2, k=2, block=True, block_size=2)
        with torch.no_grad():
            model.H[:, 0] *= 2
        model.eval().requires_grad_(False)
        train = dict(head=np.array([0, 0, 1, 2, 2, 3]),
                     tail=np.array([4, 5, 4, 4, 6, 7]),
                     relation=np.zeros(6, dtype=np.int64),
                     head_type=np.array(["entity"] * 6), tail_type=np.array(["entity"] * 6))
        valid = dict(head=np.array([0, 1, 7]), tail=np.array([6, 5, 6]),
                     relation=np.zeros(3, dtype=np.int64),
                     head_type=np.array(["entity"] * 3), tail_type=np.array(["entity"] * 3),
                     head_neg=np.tile(np.arange(500) % 8, (3, 1)),
                     tail_neg=np.tile(np.arange(500) % 8, (3, 1)))
        actual, _ = build_features(model, train, valid, {"entity": 0}, 8, 2, chunk=2)
        for direction in (0, 1):
            sh, st = ("head", "tail") if direction == 0 else ("tail", "head")
            adj = sparse.csr_matrix((np.ones(6), (train[sh], train[st])), shape=(8, 8)).toarray()
            table = embedding_table(model, np.arange(8))
            changed = embedding_table(model, np.arange(8), direction)
            similarities = [(table @ table.conj().T).real.numpy(),
                            (changed @ changed.conj().T).real.numpy()]
            for i in range(3):
                source = valid[sh][i]
                cands = np.r_[valid[st][i], valid[st + "_neg"][i]]
                for ci, c in enumerate(cands):
                    hs = np.flatnonzero(adj[:, c])
                    hs = hs[hs != source]
                    for metric in (0, 1, 2):
                        values = []
                        if metric < 2:
                            values = similarities[metric][source, hs]
                        elif adj[source].sum() > 0:
                            inter = (adj[hs] * adj[source]).sum(1)
                            values = inter / (adj[hs].sum(1) + adj[source].sum() - inter)
                        mx = max(values) if len(values) else -1.
                        top = np.sort(values)[-3:].mean() if len(values) else -1.
                        if metric < 2:
                            mx, top = mx ** 3, top ** 3
                        np.testing.assert_allclose(
                            actual[1 + 2*metric:3 + 2*metric, direction*3 + i, ci],
                            [mx, top], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
