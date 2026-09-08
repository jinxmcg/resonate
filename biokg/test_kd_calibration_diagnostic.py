"""Synthetic calibration tests; no BioKG/checkpoint/GPU access."""

import unittest

import numpy as np
import torch
import torch.nn.functional as F

from biokg.kd_calibration_diagnostic import (
    HIGH, LOW, TEMPERATURE, fit_global_scale, fit_mask, kd_rows,
    oracle_row_scales, prepare_scores, ranking_stats, scale_derivative,
)


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3602)
        torch.set_num_threads(2)
        self.x = torch.randn(32, 19, dtype=torch.float64)

    def target(self, logits):
        log_p = logits.log_softmax(-1)
        p = log_p.exp()
        return p, (p * log_p).sum(-1)

    def test_retained_kd_parity_and_shift_invariance(self):
        raw, teacher = self.x, torch.randn_like(self.x)
        p, entropy = self.target(teacher / TEMPERATURE)
        actual = kd_rows(prepare_scores(raw), p, entropy).mean()
        expected = TEMPERATURE ** 2 * F.kl_div((raw / TEMPERATURE).log_softmax(-1), p, reduction="batchmean")
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(prepare_scores(raw + torch.arange(32)[:, None]), prepare_scores(raw))

    def test_derivative_matches_autograd(self):
        p, entropy = self.target(torch.randn_like(self.x))
        beta = torch.tensor(1.3, dtype=torch.float64, requires_grad=True)
        derivative = torch.autograd.grad(kd_rows(self.x, p, entropy, beta).mean(), beta)[0]
        torch.testing.assert_close(derivative, TEMPERATURE ** 2 * scale_derivative(self.x, p, beta).mean())

    def test_global_fit_recovers_known_scale_and_uses_only_fit(self):
        p, entropy = self.target(self.x * 1.7)
        fit = torch.arange(16)
        result = fit_global_scale(self.x, p, fit)
        self.assertAlmostEqual(result["beta"], 1.7, places=5)
        p2 = p.clone()
        p2[16:] = torch.softmax(-self.x[16:] * 5, -1)
        self.assertEqual(result, fit_global_scale(self.x, p2, fit))
        self.assertLess(float(kd_rows(self.x[16:], p[16:], entropy[16:], result["beta"]).abs().max()), 1e-10)

    def test_oracle_recovers_per_query_scales(self):
        beta = torch.linspace(.3, 3, len(self.x), dtype=torch.float64)
        p, entropy = self.target(self.x * beta[:, None])
        fitted = oracle_row_scales(self.x, p, entropy)
        torch.testing.assert_close(fitted, beta, atol=1e-5, rtol=1e-5)
        self.assertLess(float(kd_rows(self.x, p, entropy, fitted).abs().max()), 1e-10)

    def test_positive_scale_does_not_fix_reversed_order(self):
        p, entropy = self.target(-self.x)
        beta = oracle_row_scales(self.x, p, entropy)
        self.assertTrue(torch.all(beta == LOW))
        self.assertGreater(float(kd_rows(self.x, p, entropy, beta).mean()), .1)
        self.assertTrue(torch.equal(self.x.argmax(-1), (self.x * beta[:, None]).argmax(-1)))

    def test_boundaries_and_uniform(self):
        p, entropy = self.target(self.x * 100)
        self.assertEqual(fit_global_scale(self.x, p, torch.arange(32))["beta"], HIGH)
        p, entropy = self.target(-self.x)
        self.assertEqual(fit_global_scale(self.x, p, torch.arange(32))["beta"], LOW)
        zero = torch.zeros_like(self.x)
        p, entropy = self.target(zero)
        self.assertTrue(torch.all(oracle_row_scales(zero, p, entropy) == 1))
        self.assertEqual(fit_global_scale(zero, p, torch.arange(32))["beta"], 1.)
        with self.assertRaises(ValueError):
            fit_global_scale(self.x, p, torch.empty(0, dtype=torch.long))

    def test_triple_split_keeps_reciprocals_and_repeats_together(self):
        h, t = np.arange(1000), np.arange(1000) + 2000
        mask = fit_mask(h, t, 4, 51)
        np.testing.assert_array_equal(mask, fit_mask(t, h, 55, 51))
        np.testing.assert_array_equal(mask, fit_mask(h, t, 4, 51))
        self.assertTrue(.4 < mask.mean() < .6)

    def test_candidate_symmetric_tie_aware_ranking(self):
        student = torch.tensor([[4., 4., 0.], [0., 3., 3.], [2., 1., 0.]])
        teacher = torch.tensor([[4., 0., 4.], [2., 1., 0.], [0., 1., 2.]])
        before = ranking_stats(student, teacher)
        np.testing.assert_array_equal(before["top1_overlap"], [True, False, False])
        perm = [2, 0, 1]
        after = ranking_stats(student[:, perm], teacher[:, perm])
        torch.testing.assert_close(before["top1_overlap"], after["top1_overlap"])
        torch.testing.assert_close(before["centered_score_cosine"], after["centered_score_cosine"])
        torch.testing.assert_close(before["top1_overlap"], ranking_stats(student * 2, teacher)["top1_overlap"])


if __name__ == "__main__":
    unittest.main()
