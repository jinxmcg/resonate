"""H35 synthetic correctness checks. No real validation data is read."""

import copy
import io
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F
from ogb.linkproppred import Evaluator

from biokg.dual_operator import (
    MODES, DualOperator, branch_b_responsibility, combine_scores,
    from_checkpoint, restore_adapted,
)
from biokg.train_dual_operator import (
    TrainStream, evaluate, frozen_digest, official_ranks, tensor_digest,
    validation_only,
)
from biokg.train_biokg_comp import score_batch
from resonate import ResonatE
from resonate_wiki import SparseTableResonatE


class DualOperatorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(35)
        torch.set_num_threads(2)
        self.original = ResonatE(12, 2, k=2, block=True, block_size=2)
        self.checkpoint = dict(args=dict(k=2, block_size=2),
                               model=copy.deepcopy(self.original.state_dict()))
        self.source = torch.tensor([0, 1, 2])
        self.relation = torch.tensor([0, 1, 0])
        self.positive = torch.tensor([3, 4, 5])
        self.negatives = torch.tensor([0, 3, 6, 7, 9])

    def test_single_matches_legacy_dense_and_sparse(self):
        model = from_checkpoint(self.checkpoint, "single")
        actual = model.training_scores(self.source, self.relation, self.positive, self.negatives)
        expected, _, _ = score_batch(self.original, self.source, self.relation, self.positive, self.negatives)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
        sp = SparseTableResonatE(12, 2, k=2, block_size=2)
        with torch.no_grad():
            sp.H.copy_(self.original.H)
            sp.E_real.copy_(torch.view_as_real(self.original.E).reshape(12, -1))
            sp.log_tau.copy_(self.original.log_tau)
        other, _, _ = score_batch(sp, self.source, self.relation, self.positive, self.negatives)
        torch.testing.assert_close(actual, other, atol=1e-6, rtol=1e-6)

    def test_identical_branches_and_parameter_count(self):
        expected, _, _ = score_batch(self.original, self.source, self.relation, self.positive, self.negatives)
        original_count = self.original.n_params()
        operator_count = self.original.H.numel() * 2
        for mode in MODES:
            model = from_checkpoint(self.checkpoint, mode, perturbation=0)
            actual = model.training_scores(self.source, self.relation, self.positive, self.negatives)
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-6)
            self.assertEqual(model.n_params(True), operator_count)
            self.assertEqual(model.n_params(), original_count + (0 if mode == "single" else operator_count))

    def test_combination_gradients_and_numerical_extremes(self):
        for mode in MODES:
            a = torch.tensor([-1000., -3., 0., 2., 1000.], requires_grad=True)
            b = torch.tensor([1000., -1., 0., -2., -1000.], requires_grad=True)
            actual = combine_scores(a, b, mode)
            self.assertTrue(torch.isfinite(actual).all())
            actual.sum().backward()
            expected = branch_b_responsibility(a.detach(), b.detach(), mode)
            torch.testing.assert_close(b.grad, expected)
            if mode != "single":
                torch.testing.assert_close(a.grad + b.grad, torch.ones_like(a))
            if mode == "or":
                self.assertTrue((actual <= torch.maximum(a, b) + 1e-5).all())
                self.assertTrue((actual >= torch.maximum(a, b) - np.log(2) - 1e-5).all())
            if mode == "and":
                self.assertTrue((actual >= torch.minimum(a, b) - 1e-5).all())
                self.assertTrue((actual <= torch.minimum(a, b) + np.log(2) + 1e-5).all())

    def test_nonlinear_combination_can_separate_modes(self):
        # Candidate 2 is the midpoint of candidates 0 and 1. No linear
        # candidate scorer can rank both endpoints above their midpoint,
        # or the midpoint above both endpoints. OR and AND can respectively.
        a, b = torch.tensor([1., -1., 0.]), torch.tensor([-1., 1., 0.])
        mean = combine_scores(a, b, "mean")
        union = combine_scores(a, b, "or")
        intersection = combine_scores(a, b, "and")
        torch.testing.assert_close(mean, torch.zeros_like(mean))
        self.assertTrue((union[:2] > union[2]).all())
        self.assertTrue((intersection[:2] < intersection[2]).all())

    def test_private_initialization_and_relative_perturbation(self):
        rng_before = torch.get_rng_state().clone()
        a = from_checkpoint(self.checkpoint, "or")
        b = from_checkpoint(self.checkpoint, "and")
        self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))
        torch.testing.assert_close(a.H_b, b.H_b, atol=0, rtol=0)
        norm = lambda x: x.abs().square().sum((-2, -1)).sqrt()
        torch.testing.assert_close(norm(a.H_b - a.H_a) / norm(a.H_a),
                                   torch.full_like(norm(a.H_a), .05), atol=1e-7, rtol=1e-5)

    def test_only_b_updates_all_combiners(self):
        for mode in MODES:
            model = from_checkpoint(self.checkpoint, mode)
            frozen = frozen_digest(model)
            before = model.H_b.detach().clone()
            opt = torch.optim.Adam([model.H_b], lr=.001)
            for _ in range(3):
                opt.zero_grad(set_to_none=True)
                scores = model.training_scores(self.source, self.relation, self.positive, self.negatives)
                F.cross_entropy(scores, torch.zeros(3, dtype=torch.long)).backward()
                self.assertTrue(torch.isfinite(model.H_b.grad).all())
                self.assertGreater(float(model.H_b.grad.abs().sum()), 0)
                opt.step()
            self.assertEqual(frozen_digest(model), frozen)
            self.assertFalse(torch.equal(model.H_b, before))
            self.assertTrue(all(p.grad is None for p in model.parameters() if not p.requires_grad))

    def test_checkpoint_round_trip_and_candidate_permutation(self):
        candidates = torch.tensor([[1, 2, 4, 2], [3, 9, 2, 0], [0, 4, 5, 6]])
        order = [3, 2, 0, 1]
        for mode in MODES:
            model = from_checkpoint(self.checkpoint, mode)
            expected = model.candidate_scores(self.source, self.relation, candidates)[0]
            buffer = io.BytesIO()
            torch.save(dict(model_type="H35DualOperator", model=model.state_dict(),
                            mode=mode, temperature=1.), buffer)
            buffer.seek(0)
            loaded = restore_adapted(torch.load(buffer, weights_only=False))
            actual = loaded.candidate_scores(self.source, self.relation, candidates)[0]
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            shuffled = loaded.candidate_scores(self.source, self.relation, candidates[:, order])[0]
            torch.testing.assert_close(shuffled, actual[:, order], atol=1e-6, rtol=1e-6)
            self.assertEqual(actual[0, 1], actual[0, 3])

    def test_stream_identical_and_typed(self):
        train = dict(head=np.array([0, 1, 2]), tail=np.array([0, 1, 3]),
                     relation=np.array([0, 0, 0]), head_type=np.array(["a"] * 3),
                     tail_type=np.array(["b"] * 3))
        streams = [TrainStream(train, {"a": 0, "b": 3}, {"a": 3, "b": 4}) for _ in MODES]
        for _ in range(20):
            samples = [s.sample(8, 16) for s in streams]
            for other in samples[1:]:
                for x, y in zip(samples[0], other):
                    np.testing.assert_array_equal(x, y)
            src, dst, relation, neg = samples[0]
            if relation == 0:
                self.assertTrue(((neg >= 3) & (neg < 7)).all())
                self.assertTrue((src < 3).all())
            else:
                self.assertTrue((neg < 3).all())
                self.assertTrue((src >= 3).all())
        self.assertEqual(len({s.digest.hexdigest() for s in streams}), 1)

    def test_probe_is_readonly_and_rng_neutral(self):
        model = from_checkpoint(self.checkpoint, "or")
        model.train()
        part = dict(head=np.array([0, 1]), tail=np.array([4, 5]),
                    relation=np.array([0, 0]), head_type=np.array(["x", "x"]),
                    tail_type=np.array(["x", "x"]),
                    head_neg=np.tile(np.arange(500) % 12, (2, 1)),
                    tail_neg=np.tile(np.arange(500) % 12, (2, 1)))
        before = tensor_digest(model.state_dict())
        rng_before = torch.get_rng_state().clone()
        stats, records = evaluate(model, part, {"x": 0}, 2, chunk=1)
        self.assertEqual(stats["queries"], 4)
        self.assertEqual(len(records["rank"]), 4)
        self.assertEqual(before, tensor_digest(model.state_dict()))
        self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
        self.assertTrue(model.training)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_official_ties(self):
        rank, rr = official_ranks(torch.tensor([[1., 1., 0.], [1., 2., 1.]]), Evaluator("ogbl-biokg"))
        torch.testing.assert_close(rank, torch.tensor([1.5, 2.5]))
        torch.testing.assert_close(rr, rank.reciprocal())

    def test_validation_only_loader(self):
        with patch("biokg.train_dual_operator.load", return_value="safe") as mocked:
            self.assertEqual(validation_only("root"), "safe")
            mocked.assert_called_once_with("root", include_test=False)


if __name__ == "__main__":
    unittest.main()
