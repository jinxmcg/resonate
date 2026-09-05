"""H31 eligibility, objective, RNG and training/evaluation separation checks."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from biokg.hard_negative import mining_weight, select_negatives, mixed_negative_ce
from biokg.train_positive_filter import TrainPositiveFilter
from resonate_wiki import SparseTableResonatE, score_batch


class HardNegativeTests(unittest.TestCase):
    def test_topk_excludes_known_positives_and_duplicate_ids(self):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for dev in devices:
            ids = torch.tensor([2, 2, 3, 4, 5], device=dev)
            scores = torch.tensor([[10., 10., 9., 8., 7.], [10., 10., 9., 8., 7.]], device=dev)
            known = torch.tensor([[1, 1, 0, 0, 0], [0, 0, 1, 0, 0]], dtype=torch.bool, device=dev)
            chosen, valid = select_negatives(scores, ids, known, 2, "topk")
            torch.testing.assert_close(ids[chosen], torch.tensor([[3, 4], [2, 4]], device=dev))
            self.assertTrue(valid.all())
            self.assertFalse(chosen.requires_grad)

    def test_random_control_is_eligible_reproducible_and_rng_isolated(self):
        ids = torch.tensor([0, 0, 1, 2, 3, 4])
        scores = torch.arange(6.).expand(20, -1)
        known = torch.zeros_like(scores, dtype=torch.bool)
        known[:, -1] = True
        state = torch.get_rng_state().clone()
        first = select_negatives(scores, ids, known, 3, "random", torch.Generator().manual_seed(31))
        second = select_negatives(scores, ids, known, 3, "random", torch.Generator().manual_seed(31))
        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertTrue(first[1].all())
        self.assertFalse(known.gather(1, first[0]).any())
        for row in ids[first[0]]:
            self.assertEqual(row.unique().numel(), 3)

    def test_mixture_and_gradients_match_manual_candidate_losses(self):
        logits = torch.tensor([[1., 2., 3., 4.], [4., 3., 2., 1.]], requires_grad=True)
        selected = torch.tensor([[2], [1]])
        valid = torch.ones_like(selected, dtype=torch.bool)
        actual = mixed_negative_ce(logits, selected, valid, .1)
        full = F.cross_entropy(logits, torch.zeros(2, dtype=torch.long))
        aux = .5 * (F.cross_entropy(logits[:1, [0, 3]], torch.tensor([0])) +
                    F.cross_entropy(logits[1:, [0, 2]], torch.tensor([0])))
        expected = .9 * full + .1 * aux
        torch.testing.assert_close(actual, expected)
        ga = torch.autograd.grad(actual, logits, retain_graph=True)[0]
        ge = torch.autograd.grad(expected, logits)[0]
        torch.testing.assert_close(ga, ge)
        # An unselected negative retains its contribution from the original CE.
        self.assertGreater(ga[0, 1].item(), 0.)

    def test_empty_pool_falls_back_and_short_pool_is_finite(self):
        for known in (torch.ones(2, 3, dtype=torch.bool), torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)):
            logits = torch.randn(2, 4, requires_grad=True)
            selected, valid = select_negatives(logits[:, 1:].detach(), torch.arange(3), known, 3, "topk")
            loss = mixed_negative_ce(logits, selected, valid, .1)
            self.assertTrue(torch.isfinite(loss))
            if known.all():
                torch.testing.assert_close(loss, F.cross_entropy(logits, torch.zeros(2, dtype=torch.long)))
            loss.backward()
            self.assertTrue(torch.isfinite(logits.grad).all())

    def test_warmup_schedule_and_zero_weight_are_exact(self):
        self.assertEqual(mining_weight(2500, .1, 2500, 1250), 0.)
        self.assertEqual(mining_weight(3125, .1, 2500, 1250), .05)
        self.assertEqual(mining_weight(3750, .1, 2500, 1250), .1)
        self.assertEqual(mining_weight(12500, .1, 2500, 1250), .1)
        logits = torch.randn(2, 4)
        got = mixed_negative_ce(logits, None, None, 0.)
        self.assertTrue(torch.equal(got, F.cross_entropy(logits, torch.zeros(2, dtype=torch.long))))

    def test_sparse_gradient_and_positive_exclusion(self):
        model = SparseTableResonatE(20, 2, k=4, block_size=4)
        index = TrainPositiveFilter(np.array([0, 0, 1]), np.array([0, 0, 0]), np.array([2, 3, 4]), 20, 1)
        src, dst, neg = torch.tensor([0, 1]), torch.tensor([2, 4]), torch.tensor([2, 3, 4, 5])
        logits, _, _ = score_batch(model, src, torch.tensor([0, 0]), dst, neg)
        known = index.mask(src, 0, neg)
        selected, valid = select_negatives(logits[:, 1:].detach(), neg, known, 2, "topk")
        self.assertFalse((known.gather(1, selected) & valid).any())
        mixed_negative_ce(logits, selected, valid, .1).backward()
        self.assertTrue(model.E_real.grad.is_sparse)
        self.assertTrue(torch.isfinite(model.E_real.grad.coalesce().values()).all())

    def test_trainer_train_only_index_and_unmodified_evaluation(self):
        from biokg import train_biokg_comp as trainer
        train = dict(head=np.array([0, 0, 1]), relation=np.zeros(3, dtype=np.int64), tail=np.array([2, 3, 4]),
                     head_type=["x"] * 3, tail_type=["x"] * 3)
        valid = dict(head=np.array([0]), relation=np.array([0]), tail=np.array([6]), head_type=["x"], tail_type=["x"])
        with tempfile.TemporaryDirectory(prefix="biokg-mining-test-") as directory:
            path = Path(directory) / "model.pt"
            argv = ["trainer", "--mining-mode", "topk", "--mining-count", "2", "--mining-warmup", "0",
                    "--mining-ramp", "1", "--steps", "2", "--eval", "valid", "--shell", "sparse",
                    "--k", "4", "--block-size", "4", "--batch", "2", "--neg", "4", "--save", str(path)]
            with patch("sys.argv", argv), \
                    patch.object(trainer, "load", return_value=({"train": train, "valid": valid}, {"x": 0}, 20,
                                 ["x"], {"x": 20})) as loader, \
                    patch.object(trainer, "TrainPositiveFilter", wraps=TrainPositiveFilter) as builder, \
                    patch.object(trainer, "eval_split") as evaluator:
                trainer.main()
            self.assertFalse(loader.call_args.kwargs["include_test"])
            for supplied, key in zip(builder.call_args.args[:3], ("head", "relation", "tail")):
                np.testing.assert_array_equal(supplied, train[key])
            self.assertIs(evaluator.call_args.args[1], valid)
            ck = torch.load(path, map_location="cpu", weights_only=False)
            self.assertFalse(ck["train_positive_filter"]["enabled"])
            self.assertEqual(ck["negative_mining"]["mode"], "topk")
            self.assertEqual(ck["negative_mining"]["candidate_slots"], 8)


if __name__ == "__main__":
    unittest.main()
