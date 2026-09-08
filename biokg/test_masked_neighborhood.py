"""MN1 synthetic tests: no real data, checkpoints or held-out labels."""

import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from biokg.masked_neighborhood import (
    NeighborIndex, Stream, check_access, logits_from_query, make_model, make_optimizers,
    objective, pair_bucket, ranks, score_candidates, state_digest, update,
)
from biokg.masked_neighborhood_run import candidates, evaluate_models, pair_interval, restore, snapshot


def fixture():
    h = np.repeat(np.arange(12), 6)
    t = np.tile(np.arange(16, 22), 12)
    fit = dict(head=h, tail=t, relation=np.zeros(len(h), np.int64), head_type_id=np.zeros(len(h), np.uint8),
               tail_type_id=np.zeros(len(h), np.uint8), row_id=np.arange(len(h)))
    manifest = dict(entity_types=['drug'], offsets={'drug': 0}, counts={'drug': 64})
    return fit, manifest


class MaskedNeighborhoodTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4)
        torch.manual_seed(36856)
        self.model = make_model(64, 2, 'cpu')
        self.fit, self.meta = fixture()
        self.index = NeighborIndex(self.fit, 64, 1)
        self.batch = (torch.arange(4), torch.zeros(4, dtype=torch.long), torch.arange(16, 20), torch.arange(22, 38))

    def test_sampling_matches_explicit_target_removed_list(self):
        source, _, target, _ = self.batch
        generator = torch.Generator().manual_seed(10)
        actual, valid = self.index.sample_without_target(source, target, 0, 128, generator)
        u = torch.rand((4, 128), generator=torch.Generator().manual_seed(10))
        expected = []
        for row, value in enumerate(target):
            pool = torch.tensor([t for t in range(16, 22) if t != int(value)])
            expected.append(pool[(u[row]*len(pool)).long()])
        torch.testing.assert_close(actual, torch.stack(expected), atol=0, rtol=0)
        self.assertTrue(valid.all())
        self.assertFalse((actual == target[:, None]).any())

    def test_duplicate_and_reverse_exclusion(self):
        fit = {k: np.concatenate([v, v, v]) for k, v in self.fit.items()}
        index = NeighborIndex(fit, 64, 1)
        for s, t, relation in ((torch.tensor([0]), torch.tensor([16]), 0),
                               (torch.tensor([16]), torch.tensor([0]), 1)):
            chosen, valid = index.sample_without_target(s, t, relation, 32, torch.Generator().manual_seed(4))
            self.assertTrue(valid.all())
            self.assertFalse((chosen == t[:, None]).any())
            self.assertTrue(index.known(s.numpy()[:, None], chosen.numpy(), relation).all())

    def test_missing_target_rejected(self):
        with self.assertRaises(ValueError):
            self.index.sample_without_target(torch.tensor([0]), torch.tensor([30]), 0, 8, torch.Generator())

    def test_single_neighbor_fallback_is_exact_original_loss_and_gradients(self):
        index = NeighborIndex(dict(head=np.array([0]), tail=np.array([16]), relation=np.array([0])), 64, 1)
        batch = tuple(v[:1] if j < 3 else v for j, v in enumerate(self.batch))
        ctx, valid = index.sample_without_target(batch[0], batch[2], 0, 8, torch.Generator())
        self.assertFalse(valid.any())
        plain, _ = objective(self.model, batch)
        masked, _ = objective(self.model, batch, ctx, valid)
        self.assertTrue(torch.equal(plain, masked))
        a = torch.autograd.grad(plain, tuple(self.model.parameters()))
        b = torch.autograd.grad(masked, tuple(self.model.parameters()))
        for x, y in zip(a, b):
            torch.testing.assert_close(x.to_dense() if x.is_sparse else x, y.to_dense() if y.is_sparse else y, atol=0, rtol=0)

    def test_plain_loss_matches_original_recipe(self):
        src, rel, pos, neg = self.batch
        q = self.model.hop(self.model.embed(src), rel)
        ep, en = self.model.rows(pos), self.model.rows(neg)
        expected = F.cross_entropy(logits_from_query(self.model, q, ep, en), torch.zeros(len(src), dtype=torch.long))
        expected += .1*(q-ep).abs().square().sum(-1).mean()
        got, _ = objective(self.model, self.batch)
        torch.testing.assert_close(got, expected, atol=1e-6, rtol=1e-6)

    def test_context_zero_weight_or_fraction_is_exact_identity(self):
        context, valid = self.index.sample_without_target(self.batch[0], self.batch[2], 0, 8, torch.Generator())
        original, _ = objective(self.model, self.batch)
        for kwargs in ({'weight': 0}, {'fraction': 0}):
            got, _ = objective(self.model, self.batch, context, valid, **kwargs)
            self.assertTrue(torch.equal(got, original))

    def test_training_changes_single_model_with_finite_sparse_gradients(self):
        before = state_digest(self.model.state_dict())
        context, valid = self.index.sample_without_target(self.batch[0], self.batch[2], 0, 8, torch.Generator())
        opts = make_optimizers(self.model)
        loss, _ = update(self.model, opts, self.batch, context, valid)
        self.assertTrue(np.isfinite(loss))
        self.assertNotEqual(before, state_digest(self.model.state_dict()))
        self.assertEqual(set(self.model.state_dict()), {'E_real', 'H', 'log_tau'})
        self.assertTrue(self.model.E_real.grad.is_sparse)
        self.assertTrue(all(torch.isfinite(p).all() for p in self.model.parameters()))

    def test_batch_stream_deterministic_and_context_rng_independent(self):
        first, other = [Stream(self.fit, self.meta, 3) for _ in range(2)]
        for _ in range(10):
            a, b = first.sample(4, 12, 'cpu'), other.sample(4, 12, 'cpu')
            for x, y in zip(a, b):
                torch.testing.assert_close(x, y)
            self.index.sample_without_target(a[0], a[2], int(a[1][0]), 8, torch.Generator())
        self.assertEqual(first.digest.hexdigest(), other.digest.hexdigest())
        self.assertEqual(first.drug_relations, {0, 1})

    def test_candidate_types_filter_and_repeatability(self):
        src, pos = np.array([0, 1]), np.array([16, 17])
        one = candidates(src, pos, 0, self.index, (0, 64), np.random.default_rng(1))
        other = candidates(src, pos, 0, self.index, (0, 64), np.random.default_rng(1))
        np.testing.assert_array_equal(one, other)
        np.testing.assert_array_equal(one[:, 0], pos)
        self.assertFalse((one[:, 1:] == src[:, None]).any())
        self.assertFalse(self.index.known(src[:, None], one[:, 1:], 0).any())
        self.assertTrue(((one >= 0) & (one < 64)).all())

    def test_candidate_permutation_duplicates_and_average_ties(self):
        src = torch.tensor([0, 1])
        rel = torch.zeros(2, dtype=torch.long)
        cand = torch.tensor([[16, 22, 23, 22], [17, 23, 22, 23]])
        a = score_candidates(self.model, src, rel, cand)
        b = score_candidates(self.model, src, rel, cand[:, [3, 0, 2, 1]])
        torch.testing.assert_close(b, a[:, [3, 0, 2, 1]], atol=0, rtol=0)
        torch.testing.assert_close(a[:, 1], a[:, 3], atol=0, rtol=0)
        torch.testing.assert_close(ranks(torch.tensor([[4., 2., 4., 5.], [1., 1., 1., 1.]])), torch.tensor([2.5, 2.5]))

    def test_probe_no_updates_or_rng_perturbation(self):
        stream = Stream(self.fit, self.meta, 4)
        before, rng = state_digest(self.model.state_dict()), torch.get_rng_state().clone()
        numpy_state = copy.deepcopy(stream.rng.bit_generator.state)
        self.model.train()
        part = {k: v[:3] for k, v in self.fit.items()}
        scores, digest = evaluate_models([self.model], part, stream, self.index, 8)
        again, other = evaluate_models([self.model], part, stream, self.index, 8)
        np.testing.assert_array_equal(scores, again)
        self.assertEqual(digest, other)
        self.assertEqual(before, state_digest(self.model.state_dict()))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(numpy_state, stream.rng.bit_generator.state)
        self.assertTrue(self.model.training)
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_snapshot_fork_model_optimizer_parity_and_independent_storage(self):
        opts = make_optimizers(self.model)
        update(self.model, opts, self.batch)
        with tempfile.TemporaryDirectory(prefix='mn1-fixture-') as directory:
            out = Path(directory)
            receipt = snapshot(self.model, opts, out, 'base', 1)
            a, ao = restore(out/'base.pt', 'cpu', True)
            b, bo = restore(out/'base.pt', 'cpu', True)
            self.assertEqual(state_digest(a.state_dict()), receipt['model'])
            self.assertEqual(state_digest([o.state_dict() for o in ao]), receipt['optimizer'])
            self.assertEqual(state_digest([o.state_dict() for o in bo]), receipt['optimizer'])
            self.assertTrue(all(x.data_ptr() != y.data_ptr() for x, y in zip(a.parameters(), b.parameters())))
            update(a, ao, self.batch)
            self.assertNotEqual(state_digest(a.state_dict()), state_digest(b.state_dict()))
            self.assertEqual(state_digest(b.state_dict()), receipt['model'])

    def test_hash_buckets_match_independent_python_arithmetic(self):
        n = 93773
        h, t = np.array([0, 2, 1000, 90000]), np.array([1, 100, 20, 90000])
        mask = (1 << 64)-1
        expected = []
        for a, b in zip(h, t):
            x = (int(min(a, b))*n+int(max(a, b))+36840+0x9e3779b97f4a7c15) & mask
            x = ((x ^ (x >> 30))*0xbf58476d1ce4e5b9) & mask
            x = ((x ^ (x >> 27))*0x94d049bb133111eb) & mask
            expected.append((x ^ (x >> 31)) % 100)
        np.testing.assert_array_equal(pair_bucket(h, t, n), expected)
        np.testing.assert_array_equal(pair_bucket(t, h, n), expected)

    def test_guard_training_cannot_open_holdout_or_foreign_checkpoint(self):
        data, out = Path('/tmp/mn1-data'), Path('/tmp/mn1-out')
        self.assertEqual(check_access(data/'fit_train.npz', data, out, 'train'), 'fit_train.npz')
        self.assertEqual(check_access(data/'holdout_train.npz', data, out, 'evaluate'), 'holdout_train.npz')
        for path in (data/'holdout_train.npz', data/'test.pt', data/'valid.pt', Path('/tmp/old-model.pt')):
            with self.assertRaises(PermissionError):
                check_access(path, data, out, 'train')

    def test_pair_bootstrap_groups_duplicate_rows(self):
        from biokg.masked_neighborhood import SPEC
        delta, pairs = np.array([1., 1., -.5]), np.array([10, 10, 20])
        rng = np.random.default_rng(SPEC['bootstrap_seed'])
        values = []
        for _ in range(100):
            ids = rng.integers(2, size=2)
            values.append(np.array([2., -.5])[ids].sum()/np.array([2, 1])[ids].sum())
        np.testing.assert_array_equal(pair_interval(delta, pairs, 100), np.quantile(values, [.025, .975]))


if __name__ == '__main__':
    unittest.main()
