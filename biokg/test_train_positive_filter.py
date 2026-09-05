"""Exact train-only masking, directionality, and autograd checks for H30."""
import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from biokg.train_positive_filter import TrainPositiveFilter, mask_negative_logits
from resonate_wiki import SparseTableResonatE, score_batch


class TrainPositiveFilterTests(unittest.TestCase):
    def test_per_query_duplicates_reverse_and_unseen_entities(self):
        # The 0->4 edge appears twice; 0->5 is another known positive.
        h, r, t = np.array([0, 0, 0, 1, 2]), np.array([0, 0, 0, 0, 1]), np.array([4, 4, 5, 6, 7])
        index = TrainPositiveFilter(h, r, t, 10, 3)
        negatives = torch.tensor([4, 4, 5, 6, 7, 9])
        actual = index.mask(torch.tensor([0, 1, 3]), 0, negatives)
        torch.testing.assert_close(actual, torch.tensor([[1, 1, 1, 0, 0, 0],
                                                        [0, 0, 0, 1, 0, 0],
                                                        [0, 0, 0, 0, 0, 0]], dtype=torch.bool))
        reverse = index.mask(torch.tensor([4, 5, 6]), 3, torch.tensor([0, 1, 2]))
        torch.testing.assert_close(reverse, torch.tensor([[1, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=torch.bool))
        self.assertFalse(index.mask(torch.tensor([2]), 0, torch.tensor([7])).item())
        self.assertFalse(index.mask(torch.tensor([0]), 2, negatives).any())

    def test_random_membership_matches_exact_set_across_byte_boundaries(self):
        rng = np.random.default_rng(5)
        h, r, t = rng.integers(0, 37, 500), rng.integers(0, 3, 500), rng.integers(0, 37, 500)
        known = set(zip(h.tolist(), r.tolist(), t.tolist()))
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            index = TrainPositiveFilter(h, r, t, 40, 4, device)
            src = torch.arange(40, device=device)
            neg = torch.arange(40, device=device)
            for rel in range(8):
                actual = index.mask(src, rel, neg).cpu().numpy()
                expected = np.array([[(s, rel, d) in known if rel < 4 else
                                      (d, rel - 4, s) in known for d in range(40)] for s in range(40)])
                np.testing.assert_array_equal(actual, expected)

    def test_loss_and_gradients_match_explicit_candidate_removal(self):
        logits = torch.tensor([[1., 2., 3., 4.], [4., 2., 3., 1.]], requires_grad=True)
        mask = torch.tensor([[True, False, True], [False, True, False]])
        got = F.cross_entropy(mask_negative_logits(logits, mask), torch.zeros(2, dtype=torch.long))
        expected = (F.cross_entropy(logits[:1, [0, 2]], torch.tensor([0])) +
                    F.cross_entropy(logits[1:, [0, 1, 3]], torch.tensor([0]))) / 2
        torch.testing.assert_close(got, expected)
        actual_grad = torch.autograd.grad(got, logits, retain_graph=True)[0]
        expected_grad = torch.autograd.grad(expected, logits)[0]
        torch.testing.assert_close(actual_grad, expected_grad)
        self.assertTrue((actual_grad[:, 1:][mask] == 0).all())

    def test_all_negatives_masked_is_finite_and_zero_gradient(self):
        logits = torch.randn(3, 5, requires_grad=True)
        loss = F.cross_entropy(mask_negative_logits(logits, torch.ones(3, 4, dtype=torch.bool)),
                               torch.zeros(3, dtype=torch.long))
        self.assertEqual(loss.item(), 0.)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue((logits.grad == 0).all())

    def test_empty_mask_and_rng_are_unchanged(self):
        torch.manual_seed(4)
        state = torch.get_rng_state().clone()
        index = TrainPositiveFilter(np.array([0]), np.array([0]), np.array([1]), 4, 1)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        logits = torch.randn(2, 3)
        mask = index.mask(torch.tensor([2, 3]), 0, torch.tensor([0, 1]))
        self.assertTrue(torch.equal(logits, mask_negative_logits(logits, mask)))

    def test_sparse_model_gradients_remain_sparse_and_finite(self):
        model = SparseTableResonatE(20, 2, k=4, block_size=4)
        index = TrainPositiveFilter(np.array([0, 0, 1]), np.array([0, 0, 0]), np.array([2, 3, 4]), 20, 1)
        src, dst, neg = torch.tensor([0, 1]), torch.tensor([2, 4]), torch.tensor([2, 3, 4, 5])
        logits, _, _ = score_batch(model, src, torch.tensor([0, 0]), dst, neg)
        loss = F.cross_entropy(mask_negative_logits(logits, index.mask(src, 0, neg)), torch.zeros(2, dtype=torch.long))
        loss.backward()
        self.assertTrue(model.E_real.grad.is_sparse)
        self.assertTrue(torch.isfinite(model.E_real.grad.coalesce().values()).all())
        self.assertTrue(torch.isfinite(model.H.grad).all())

    def test_held_out_edges_do_not_enter_mask(self):
        # 0->3 is a held-out example and is deliberately NOT supplied.
        index = TrainPositiveFilter(np.array([0]), np.array([0]), np.array([2]), 5, 1)
        mask = index.mask(torch.tensor([0]), 0, torch.tensor([2, 3]))
        torch.testing.assert_close(mask, torch.tensor([[True, False]]))

    def test_trainer_uses_only_train_edges_and_leaves_evaluation_unchanged(self):
        from biokg import train_biokg_comp as trainer
        train = dict(head=np.array([0, 0, 1]), relation=np.zeros(3, dtype=np.int64),
                     tail=np.array([2, 3, 4]), head_type=["x"] * 3, tail_type=["x"] * 3)
        valid = dict(head=np.array([0]), relation=np.array([0]), tail=np.array([6]),
                     head_type=["x"], tail_type=["x"])
        with tempfile.TemporaryDirectory(prefix="biokg-filter-test-") as directory:
            path = Path(directory) / "model.pt"
            argv = ["trainer", "--filter-train-positives", "--eval", "valid", "--steps", "2",
                    "--k", "4", "--block-size", "4", "--shell", "sparse", "--batch", "2",
                    "--neg", "4", "--device", "cpu", "--save", str(path)]
            with patch("sys.argv", argv), \
                    patch.object(trainer, "load", return_value=({"train": train, "valid": valid},
                                 {"x": 0}, 20, ["x"], {"x": 20})) as loader, \
                    patch.object(trainer, "TrainPositiveFilter", wraps=TrainPositiveFilter) as builder, \
                    patch.object(trainer, "eval_split") as evaluator:
                trainer.main()
            self.assertFalse(loader.call_args.kwargs["include_test"])
            for supplied, key in zip(builder.call_args.args[:3], ("head", "relation", "tail")):
                np.testing.assert_array_equal(supplied, train[key])
            evaluator.assert_called_once()
            self.assertIs(evaluator.call_args.args[1], valid)
            ck = torch.load(path, map_location="cpu", weights_only=False)
            self.assertTrue(ck["train_positive_filter"]["enabled"])
            self.assertEqual(ck["train_positive_filter"]["sampled_count"], 16)
            self.assertFalse(any("mask" in key or "filter" in key for key in ck["model"]))


if __name__ == "__main__":
    unittest.main()
