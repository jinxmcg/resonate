"""Synthetic correctness checks for the H35B cross-block composition."""

import unittest

import torch
import torch.nn.functional as F

from biokg.sequential_operator import from_student, restore_sequential
from biokg.train_dual_operator import frozen_digest
from biokg.train_biokg_comp import score_batch
from resonate import ResonatE


class SequentialOperatorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(355)
        torch.set_num_threads(2)
        self.base = ResonatE(20, 2, k=4, block=True, block_size=4)
        self.ck = dict(model=self.base.state_dict())
        self.src, self.rel = torch.tensor([0, 1]), torch.tensor([0, 1])
        self.dst, self.neg = torch.tensor([2, 3]), torch.tensor([4, 5, 6])

    def test_identity_reproduces_reference(self):
        expected = score_batch(self.base, self.src, self.rel, self.dst, self.neg)[0]
        for mixing in ("local", "shuffle"):
            model = from_student(self.ck, mixing)
            actual = model.training_scores(self.src, self.rel, self.dst, self.neg)
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-6)
            self.assertEqual(model.n_params(True), self.base.H.numel() * 2)

    def test_local_equals_block_product(self):
        model = from_student(self.ck, "local")
        with torch.no_grad():
            model.H_b.add_(.1 * torch.randn_like(model.H_b))
        x = torch.randn(2, 16, dtype=torch.complex64)
        _, actual = model.transform(x, self.rel)
        product = model.H_b[self.rel] @ model.H_a[self.rel]
        expected = torch.einsum("bkij,bkj->bki", product, x.reshape(2, 4, 4)).reshape(2, 16)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    def test_shuffle_creates_cross_block_dependency(self):
        operators = {}
        for mixing in ("local", "shuffle"):
            m = from_student(self.ck, mixing)
            with torch.no_grad():
                m.H_a.copy_(torch.eye(4, dtype=torch.complex64).expand_as(m.H_a))
                m.H_b[0, 0, 0, 1] = 1
            operators[mixing] = m
        x = torch.zeros(1, 16, dtype=torch.complex64)
        x[0, 4] = 1  # input from original block 1
        r = torch.tensor([0])
        local = operators["local"].transform(x, r)[1]
        shuffled = operators["shuffle"].transform(x, r)[1]
        self.assertEqual(local[0, 0], 0)
        self.assertEqual(shuffled[0, 0], 1)  # now reaches original block 0

    def test_training_freezes_source_and_roundtrip(self):
        for mixing in ("local", "shuffle"):
            m = from_student(self.ck, mixing)
            digest = frozen_digest(m)
            before = m.H_b.detach().clone()
            opt = torch.optim.Adam([m.H_b], lr=.001)
            scores = m.training_scores(self.src, self.rel, self.dst, self.neg)
            F.cross_entropy(scores, torch.zeros(2, dtype=torch.long)).backward()
            self.assertTrue(torch.isfinite(m.H_b.grad).all())
            opt.step()
            self.assertFalse(torch.equal(before, m.H_b))
            self.assertEqual(digest, frozen_digest(m))
            restored = restore_sequential(dict(model_type="H35BSequentialOperator",
                                                model=m.state_dict(), mixing=mixing))
            cand = torch.tensor([[1, 4, 7], [2, 5, 6]])
            expected = m.candidate_scores(self.src, self.rel, cand)[0]
            actual = restored.candidate_scores(self.src, self.rel, cand[:, [2, 0, 1]])[0]
            torch.testing.assert_close(actual, expected[:, [2, 0, 1]], atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
