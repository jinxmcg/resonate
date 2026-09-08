"""Synthetic H35D tests: no BioKG checkpoints or data access."""

import unittest
from unittest.mock import patch

import numpy as np
import torch
from ogb.linkproppred import Evaluator
from scipy.stats import rankdata

from biokg.teacher_fusion_diagnostic import (
    assert_frozen, candidate_scores, doubled_candidate_ranks, fusion_batch,
    model_hashes, paired_comparison, validation_only,
)
from biokg.train_biokg_comp import teacher_logits
from biokg.train_dual_operator import official_ranks
from resonate import ResonatE


class TeacherFusionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3511)
        self.teachers = [ResonatE(20, 2, k=2, block=True, block_size=2)
                         .eval().requires_grad_(False) for _ in range(3)]
        self.source = torch.tensor([0, 1, 2])
        self.rel = torch.tensor([0, 1, 0])
        self.pos = torch.tensor([3, 4, 5])
        self.neg = torch.tensor([4, 5, 6, 7, 8])
        self.candidates = torch.cat([self.pos[:, None], self.neg[None].expand(3, -1)], 1)
        self.ev = Evaluator("ogbl-biokg")

    def test_ties_match_scipy_and_official(self):
        scores = torch.tensor([[4., 4., 1., 2., 2.], [0., 0., 0., 0., 0.], [-2., 9., 3., 5., 6.]])
        ranks = doubled_candidate_ranks(scores).numpy() / 2
        np.testing.assert_array_equal(ranks, rankdata(-scores.numpy(), method="average", axis=1))
        np.testing.assert_array_equal(ranks[:, 0], official_ranks(scores, self.ev)[0].numpy())

    def test_monotone_invariance(self):
        scores = torch.tensor([[-2., 0., 2., 2., 4.]])
        for transformed in (scores * 12 + 17, scores**3, scores.exp()):
            torch.testing.assert_close(doubled_candidate_ranks(scores), doubled_candidate_ranks(transformed))

    def test_invalid_scores(self):
        for scores in (torch.ones(2), torch.ones(2, 1), torch.tensor([[1., float("nan")]])):
            with self.assertRaises(ValueError):
                doubled_candidate_ranks(scores)

    def test_mean_matches_original_training_target(self):
        mean, _, _ = fusion_batch(self.teachers, self.source, self.rel, self.candidates, self.ev)
        expected = teacher_logits(self.teachers, self.source, self.rel, self.pos, self.neg)
        torch.testing.assert_close(mean, expected, atol=2e-6, rtol=2e-6)

    def test_all_candidate_permutation_and_duplicates(self):
        order = [4, 1, 0, 3, 2, 5]
        mean, fused, _ = fusion_batch(self.teachers, self.source, self.rel, self.candidates, self.ev)
        pm, pf, _ = fusion_batch(self.teachers, self.source, self.rel, self.candidates[:, order], self.ev)
        torch.testing.assert_close(pm, mean[:, order], atol=0, rtol=0)
        torch.testing.assert_close(pf, fused[:, order], atol=0, rtol=0)
        # Row 1 has the true target duplicated among negatives; neither gets special treatment.
        self.assertEqual(mean[1, 0], mean[1, 1])
        self.assertEqual(fused[1, 0], fused[1, 1])

    def test_identical_teachers_preserve_ranks(self):
        teacher = self.teachers[0]
        _, fused, _ = fusion_batch([teacher] * 10, self.source, self.rel, self.candidates, self.ev)
        raw = candidate_scores(teacher, self.source, self.rel, self.candidates)
        torch.testing.assert_close(official_ranks(raw, self.ev)[0], official_ranks(fused, self.ev)[0])

    def test_fusion_is_candidate_rank_aggregation(self):
        logits = [torch.tensor([[10., 9., 0.]]), torch.tensor([[0., 9., 8.]])]
        with patch("biokg.teacher_fusion_diagnostic.candidate_scores", side_effect=logits):
            mean, fused, ranks = fusion_batch(self.teachers[:2], self.source, self.rel,
                                              torch.zeros(1, 3, dtype=torch.long), self.ev)
        torch.testing.assert_close(mean, torch.tensor([[5., 9., 4.]]))
        torch.testing.assert_close(fused, torch.tensor([[-8., -6., -10.]]))
        self.assertEqual(official_ranks(fused, self.ev)[0].item(), 2.)
        self.assertNotEqual(float(ranks.reciprocal().mean()), .5)

    def test_frozen_models_unchanged_and_no_grad(self):
        models = {str(i): model for i, model in enumerate(self.teachers)}
        before = model_hashes(models)
        mean, fused, _ = fusion_batch(self.teachers, self.source, self.rel, self.candidates, self.ev)
        self.assertFalse(mean.requires_grad)
        self.assertFalse(fused.requires_grad)
        self.assertEqual(before, model_hashes(models))
        assert_frozen(models)

    def test_selective_loader(self):
        with patch("biokg.teacher_fusion_diagnostic.load", return_value="sentinel") as mocked:
            self.assertEqual(validation_only("dummy"), "sentinel")
            mocked.assert_called_once_with("dummy", include_test=False)

    def test_paired_comparison(self):
        result = paired_comparison(np.array([1., 1., 1., 1.]), np.array([2., 2., 2., 2.]))
        self.assertEqual(result["delta_mrr"], .5)
        self.assertEqual(result["bootstrap_95"], [.5, .5])
        self.assertEqual(result["recovered_top1"], 4)


if __name__ == "__main__":
    unittest.main()
