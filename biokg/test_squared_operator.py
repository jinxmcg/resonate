"""H37 synthetic tests, no real graph or trained artifacts."""

import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from biokg.masked_neighborhood import (
    NeighborIndex, Stream, check_access, make_model, make_optimizers,
    objective as original_objective, score_candidates, state_digest,
)
from biokg.masked_neighborhood_run import evaluate_models as original_evaluate
from biokg.squared_operator import (
    SPEC, SquaredOperator, gram_normalizer, make_pair, objective, real_view,
    restore, snapshot, squared_weight, update,
)
from biokg.squared_operator_run import evaluate_models, summary
from biokg.test_masked_neighborhood import fixture


class SquaredTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4)
        torch.manual_seed(37056)
        self.models = make_pair(64, 2, 'cpu')
        self.batch = (torch.arange(4), torch.zeros(4, dtype=torch.long), torch.arange(16, 20), torch.arange(22, 38))

    def test_real_view_matches_complex_real_inner_product(self):
        q = torch.randn(5, 7, dtype=torch.complex128)
        v = torch.randn(9, 7, dtype=torch.complex128)
        torch.testing.assert_close(real_view(q) @ real_view(v).T, (q @ v.conj().T).real)

    def test_exact_probability_sums_and_all_gradients(self):
        q = torch.randn(3, 4, dtype=torch.complex128, requires_grad=True)
        v = torch.randn(7, 4, dtype=torch.complex128, requires_grad=True)
        lt = torch.tensor(.4, dtype=torch.float64, requires_grad=True)
        direct_weights = squared_weight((q @ v.conj().T).real, lt)
        exact = gram_normalizer(q, v, lt)
        torch.testing.assert_close(exact, direct_weights.sum(-1), atol=1e-12, rtol=1e-12)
        torch.testing.assert_close((direct_weights/exact[:, None]).sum(-1), torch.ones(3, dtype=torch.float64))
        positive = direct_weights[torch.arange(3), torch.tensor([0, 3, 6])]
        direct_loss = (direct_weights.sum(-1).log()-positive.log()).mean()
        gram_loss = (exact.log()-positive.log()).mean()
        a = torch.autograd.grad(direct_loss, (q, v, lt), retain_graph=True)
        b = torch.autograd.grad(gram_loss, (q, v, lt))
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y, atol=1e-11, rtol=1e-10)

    def test_gradcheck_exact_normalizer(self):
        q = torch.randn(2, 3, dtype=torch.complex128, requires_grad=True)
        v = torch.randn(4, 3, dtype=torch.complex128, requires_grad=True)
        lt = torch.tensor(.2, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda a, b, c: gram_normalizer(a, b, c).log(),
            (q, v, lt), eps=1e-6, atol=1e-5, rtol=1e-4))

    def test_zero_queries_and_zero_dots_are_finite_uniform(self):
        q = torch.zeros(2, 3, dtype=torch.complex128, requires_grad=True)
        v = torch.randn(4, 3, dtype=torch.complex128, requires_grad=True)
        lt = torch.tensor(0., dtype=torch.float64, requires_grad=True)
        weights = squared_weight((q @ v.conj().T).real, lt)
        normalizer = gram_normalizer(q, v, lt)
        torch.testing.assert_close(weights/normalizer[:, None], torch.full((2, 4), .25, dtype=torch.float64))
        (normalizer.log()-weights[:, 0].log()).mean().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in (q, v, lt)))

    def test_sign_symmetry_and_epsilon_scale(self):
        dot = torch.tensor([-2., -.2, 0., .2, 2.])
        lt = torch.tensor(.7)
        torch.testing.assert_close(squared_weight(dot, lt), squared_weight(-dot, lt), atol=0, rtol=0)
        self.assertEqual(float(squared_weight(torch.tensor(0.), lt)), float(torch.tensor(SPEC['epsilon'])))
        self.assertGreater(float(squared_weight(dot[-1], lt)), float(squared_weight(dot[3], lt)))

    def test_linear_control_matches_original_loss_and_gradients(self):
        m = make_model(64, 2, 'cpu')
        m.load_state_dict(self.models[0].state_dict())
        a, _ = original_objective(m, self.batch)
        b, _ = objective(self.models[0], self.batch, (0, 64))
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        ga = torch.autograd.grad(a, tuple(m.parameters()))
        gb = torch.autograd.grad(b, tuple(self.models[0].parameters()))
        for x, y in zip(ga, gb):
            torch.testing.assert_close(x.to_dense() if x.is_sparse else x, y.to_dense() if y.is_sparse else y, atol=0, rtol=0)

    def test_full_model_gram_loss_and_gradients_match_enumeration(self):
        m = self.models[1]
        lo, hi = 16, 32
        src, rel, pos, _ = self.batch
        q, ep = m.hop(m.embed(src), rel), m.rows(pos)
        weights = squared_weight((q @ m.table()[lo:hi].conj().T).real, m.log_tau)
        direct = (weights.sum(1).log()-weights[torch.arange(4), pos-lo].log()).mean()
        direct += .1*(q-ep).abs().square().sum(-1).mean()
        compact, _ = objective(m, self.batch, (lo, hi))
        # GEMM computes the reference positive; production uses a paired sum.
        # Near-zero dot products amplify their fp32 reduction-order difference
        # through log(epsilon + dot^2). Float64 algebra/gradients are checked
        # separately above at 1e-11; allow fp32 rounding here, not a detachment.
        torch.testing.assert_close(compact, direct, atol=2e-5, rtol=2e-6)
        ga = torch.autograd.grad(direct, tuple(m.parameters()))
        gb = torch.autograd.grad(compact, tuple(m.parameters()))
        for x, y in zip(ga, gb):
            torch.testing.assert_close(x, y, atol=2e-5, rtol=2e-4)

    def test_normalizer_reaches_eligible_nonpositive_not_other_types(self):
        m = self.models[1]
        loss, _ = objective(m, self.batch, (16, 24))
        loss.backward()
        g = m.E_real.grad
        self.assertFalse(g.is_sparse)
        self.assertGreater(float(g[20:24].abs().sum()), 0.)
        self.assertEqual(float(g[24:].abs().sum()), 0.)
        self.assertGreater(float(g[:4].abs().sum()), 0.)
        self.assertEqual(float(g[4:16].abs().sum()), 0.)

    def test_normalizer_is_recomputed_not_stale_or_detached(self):
        m = self.models[1]
        loss, _ = objective(m, self.batch, (16, 24))
        with torch.no_grad():
            m.E_real[23].mul_(2.)
        second, _ = objective(m, self.batch, (16, 24))
        self.assertFalse(torch.allclose(loss, second))
        self.assertGreater(float(torch.autograd.grad(second, m.E_real)[0][23].abs().sum()), 0.)

    def test_squared_loss_ignores_sampled_negatives(self):
        m = self.models[1]
        a, _ = objective(m, self.batch, (16, 24))
        b, _ = objective(m, (*self.batch[:3], torch.tensor([63, 62])), (16, 24))
        torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_invalid_support_relation_and_model_rejected(self):
        m = self.models[1]
        for support in ((0, 10), (16, 16), (-1, 64), (0, 65)):
            with self.assertRaises(ValueError):
                objective(m, self.batch, support)
        with self.assertRaises(ValueError):
            objective(m, (self.batch[0], torch.tensor([0, 1, 0, 0]), *self.batch[2:]), (16, 24))
        with self.assertRaises(ValueError):
            SquaredOperator(64, 2, 'unknown')

    def test_same_parameter_budget_and_independent_storage(self):
        a, b = self.models
        self.assertEqual(a.n_params(), b.n_params())
        self.assertEqual(a.H.shape, (2, 36, 4, 4))
        self.assertEqual(set(a.state_dict()), {'H', 'E_real', 'log_tau'})
        self.assertEqual(state_digest(a.state_dict()), state_digest(b.state_dict()))
        self.assertTrue(all(x.data_ptr() != y.data_ptr() for x, y in zip(a.parameters(), b.parameters())))
        for m in self.models:
            old = state_digest(m.state_dict())
            value, _ = update(m, make_optimizers(m), self.batch, (16, 32))
            self.assertTrue(np.isfinite(value))
            self.assertNotEqual(old, state_digest(m.state_dict()))
            self.assertTrue(all(torch.isfinite(p).all() for p in m.parameters()))

    def test_dense_and_sparse_same_optimizer_rule(self):
        a = self.models[0]
        b = copy.deepcopy(a)
        b.sparse_grad = False
        opts = [make_optimizers(m) for m in (a, b)]
        for _ in range(3):
            for m, oo in zip((a, b), opts):
                update(m, oo, self.batch, (16, 32))
        for x, y in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(x, y, atol=2e-6, rtol=2e-5)

    def test_checkpoint_standalone_scorer_and_metadata(self):
        cand = torch.tensor([[16, 22], [17, 23], [18, 24], [19, 25]])
        with tempfile.TemporaryDirectory(prefix='h37-test-') as directory:
            out = Path(directory)
            for m in self.models:
                oo = make_optimizers(m)
                update(m, oo, self.batch, (16, 32))
                snapshot(m, oo, out, 1)
                got = restore(out/(m.arm+'.pt'))
                self.assertEqual(got.arm, m.arm)
                torch.testing.assert_close(got.candidate_scores(*self.batch[:2], cand),
                                           m.candidate_scores(*self.batch[:2], cand), atol=0, rtol=0)
            ck = torch.load(out/'squared.pt', weights_only=False)
            ck['spec'] = dict(ck['spec'], epsilon=.01)
            torch.save(ck, out/'bad.pt')
            with self.assertRaises(ValueError):
                restore(out/'bad.pt')

    def test_candidate_columns_duplicates_and_linear_parity(self):
        cand = torch.tensor([[16, 22, 23, 22], [17, 23, 22, 23], [18, 24, 22, 24], [19, 25, 22, 25]])
        perm = [3, 0, 2, 1]
        for m in self.models:
            s = m.candidate_scores(*self.batch[:2], cand)
            torch.testing.assert_close(m.candidate_scores(*self.batch[:2], cand[:, perm]), s[:, perm], atol=0, rtol=0)
            torch.testing.assert_close(s[:, 1], s[:, 3], atol=0, rtol=0)
        torch.testing.assert_close(self.models[0].candidate_scores(*self.batch[:2], cand),
                                   score_candidates(self.models[0], *self.batch[:2], cand), atol=0, rtol=0)

    def test_evaluation_rng_detachment_and_original_candidate_stream(self):
        fit, meta = fixture()
        part = {k: v[:4] for k, v in fit.items()}
        index, stream = NeighborIndex(fit, 64, 1), Stream(fit, meta, 0)
        hashes = [state_digest(m.state_dict()) for m in self.models]
        rng, np_rng = torch.get_rng_state().clone(), copy.deepcopy(stream.rng.bit_generator.state)
        ranked, digest = evaluate_models(self.models, part, stream, index, SPEC['evaluation_seed'])
        original, old_digest = original_evaluate([self.models[0]], part, stream, index, SPEC['evaluation_seed'])
        np.testing.assert_array_equal(ranked[0], original[0])
        self.assertEqual(digest, old_digest)
        self.assertEqual(hashes, [state_digest(m.state_dict()) for m in self.models])
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(np_rng, stream.rng.bit_generator.state)
        self.assertTrue(all(p.grad is None for m in self.models for p in m.parameters()))

    def test_guard_and_primary_gate(self):
        for path in ('/tmp/h37-data/holdout_train.npz', '/tmp/foreign.pt'):
            with self.assertRaises(PermissionError):
                check_access(path, '/tmp/h37-data', '/tmp/h37-out', 'train')
        fit, meta = fixture()
        held = {k: v[:4] for k, v in fit.items()}
        ranked = np.array([[[2]*4]*2, [[1]*4]*2], np.float32)
        result = summary(ranked, held, fit, meta)
        self.assertEqual(result['primary']['delta_mrr'], .5)
        self.assertEqual(result['primary']['pair_cluster_95'], [.5, .5])
        self.assertTrue(result['advance_gate'])
        self.assertFalse(summary(ranked[::-1], held, fit, meta)['advance_gate'])


if __name__ == '__main__':
    unittest.main()
