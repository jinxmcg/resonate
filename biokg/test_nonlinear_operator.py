"""NL1 synthetic correctness checks; no real graph or trained weights."""

import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from resonate import cnorm
from biokg.masked_neighborhood import (
    NeighborIndex, Stream, make_model, make_optimizers, objective, score_candidates,
    state_digest, update,
)
from biokg.masked_neighborhood_run import evaluate_models
from biokg.nonlinear_operator import (
    NonlinearOperator, SPEC, activate, activation_diagnostic, make_pair, restore,
    snapshot, summary,
)
from biokg.test_masked_neighborhood import fixture


class NonlinearTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4)
        torch.manual_seed(36956)
        self.models = make_pair(64, 2, 'cpu')
        self.batch = (torch.arange(4), torch.zeros(4, dtype=torch.long), torch.arange(16, 20), torch.arange(22, 38))

    def test_alpha_zero_original_output_loss_gradient_parity(self):
        original = make_model(64, 2, 'cpu')
        original.load_state_dict(self.models[0].state_dict())
        a, _ = objective(original, self.batch)
        b, _ = objective(self.models[0], self.batch)
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        ga = torch.autograd.grad(a, tuple(original.parameters()))
        gb = torch.autograd.grad(b, tuple(self.models[0].parameters()))
        for x, y in zip(ga, gb):
            torch.testing.assert_close(x.to_dense() if x.is_sparse else x, y.to_dense() if y.is_sparse else y, atol=0, rtol=0)

    def test_explicit_real_arithmetic_before_global_normalization(self):
        m = self.models[1]
        src, rel = self.batch[:2]
        x = m.embed(src).reshape(4, 36, 4)
        h = m.H[rel]
        real = (h.real*x.real[:, :, None, :]-h.imag*x.imag[:, :, None, :]).sum(-1)
        imag = (h.real*x.imag[:, :, None, :]+h.imag*x.real[:, :, None, :]).sum(-1)
        z = torch.complex(real, imag).reshape(4, 144)
        expected = cnorm(z/(1+torch.sqrt(z.real.square()+z.imag.square())))
        actual = m.hop(m.embed(src), rel)
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
        wrong_order = cnorm(activate(cnorm(z), 1.))
        with torch.no_grad():
            m.H.mul_(5)
        actual_scaled = m.hop(m.embed(src), rel)
        self.assertFalse(torch.allclose(actual_scaled, wrong_order, atol=1e-5, rtol=1e-5))

    def test_phase_magnitude_and_zero_behavior(self):
        z = torch.tensor([0, 1+2j, -4+3j, 1e10-1e10j], dtype=torch.complex64, requires_grad=True)
        y = activate(z, 1.)
        self.assertTrue(torch.isfinite(y).all())
        self.assertEqual(y[0], 0)
        self.assertTrue((y.abs() <= 1.000001).all())
        torch.testing.assert_close(y[1:]/y[1:].abs(), z[1:]/z[1:].abs())
        y.real.sum().backward()
        self.assertTrue(torch.isfinite(z.grad).all())

    def test_complex_double_gradcheck(self):
        z = torch.tensor([0, .3+.7j, -.6+.2j], dtype=torch.complex128, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda v: activate(v, 1.), (z,), eps=1e-6, atol=1e-5, rtol=1e-4))

    def test_invalid_strengths_rejected(self):
        for alpha in (-1., float('inf'), float('nan')):
            with self.assertRaises(ValueError):
                activate(torch.ones(1), alpha)
        with self.assertRaises(ValueError):
            NonlinearOperator(64, 2, .5)

    def test_function_is_nonlinear_and_changes_h_gradients(self):
        z = torch.tensor([.2+.1j, .8-1.2j])
        self.assertFalse(torch.allclose(activate(2*z, 1.), 2*activate(z, 1.)))
        gradients = []
        for m in self.models:
            loss, _ = objective(m, self.batch)
            gradients.append(torch.autograd.grad(loss, m.H)[0])
        self.assertFalse(torch.allclose(*gradients, atol=1e-7, rtol=1e-5))

    def test_same_initial_storage_budget_independent_optimization(self):
        a, b = self.models
        self.assertEqual(a.n_params(), b.n_params())
        self.assertEqual(set(a.state_dict()), {'E_real', 'H', 'log_tau'})
        self.assertEqual(a.H.shape, (2, 36, 4, 4))
        self.assertEqual(state_digest(a.state_dict()), state_digest(b.state_dict()))
        self.assertTrue(all(x.data_ptr() != y.data_ptr() for x, y in zip(a.parameters(), b.parameters())))
        opts = [make_optimizers(m) for m in self.models]
        self.assertEqual(state_digest([o.state_dict() for o in opts[0]]), state_digest([o.state_dict() for o in opts[1]]))
        for m, oo in zip(self.models, opts):
            old_h, old_e = m.H.detach().clone(), m.E_real.detach().clone()
            value, _ = update(m, oo, self.batch)
            self.assertTrue(np.isfinite(value))
            self.assertTrue(m.E_real.grad.is_sparse)
            self.assertFalse(torch.equal(old_h, m.H))
            self.assertFalse(torch.equal(old_e, m.E_real))
            self.assertTrue(all(torch.isfinite(p).all() for p in m.parameters()))
        self.assertNotEqual(state_digest(a.state_dict()), state_digest(b.state_dict()))

    def test_standalone_checkpoint_preserves_activation(self):
        with tempfile.TemporaryDirectory(prefix='nl1-test-') as directory:
            out = Path(directory)
            for name, m in zip(('linear', 'nonlinear'), self.models):
                oo = make_optimizers(m)
                update(m, oo, self.batch)
                snapshot(m, oo, out, name, 1)
                got = restore(out/(name+'.pt'))
                self.assertEqual(got.alpha, m.alpha)
                self.assertEqual(state_digest(got.state_dict()), state_digest(m.state_dict()))
                cand = torch.tensor([[16, 22], [17, 23], [18, 24], [19, 25]])
                torch.testing.assert_close(score_candidates(got, *self.batch[:2], cand),
                                           score_candidates(m, *self.batch[:2], cand), atol=0, rtol=0)
                with self.assertRaises(FileExistsError):
                    snapshot(m, oo, out, name, 1)
            ck = torch.load(out/'nonlinear.pt', weights_only=False)
            ck['alpha'] = 0.
            torch.save(ck, out/'bad.pt')
            with self.assertRaises(ValueError):
                restore(out/'bad.pt')

    def test_candidate_symmetry_duplicates_and_fixed_loss(self):
        m = self.models[1]
        cand = torch.tensor([[16, 22, 23, 22], [17, 23, 22, 23], [18, 24, 22, 24], [19, 25, 22, 25]])
        scores = score_candidates(m, *self.batch[:2], cand)
        perm = [3, 0, 2, 1]
        torch.testing.assert_close(score_candidates(m, *self.batch[:2], cand[:, perm]), scores[:, perm], atol=0, rtol=0)
        torch.testing.assert_close(scores[:, 1], scores[:, 3], atol=0, rtol=0)
        _, parts = objective(m, self.batch)
        self.assertEqual(parts['context_rows'], 0)
        self.assertNotIn('context_ce', parts)

    def test_probe_detached_rng_isolated_and_repeatable(self):
        fit, meta = fixture()
        index = NeighborIndex(fit, 64, 1)
        stream = Stream(fit, meta, 0)
        part = {k: v[:3] for k, v in fit.items()}
        digest = [state_digest(m.state_dict()) for m in self.models]
        cpu_rng = torch.get_rng_state().clone()
        np_rng = copy.deepcopy(stream.rng.bit_generator.state)
        first, one = evaluate_models(self.models, part, stream, index, SPEC['probe_seed'])
        second, two = evaluate_models(self.models, part, stream, index, SPEC['probe_seed'])
        np.testing.assert_array_equal(first, second)
        self.assertEqual(one, two)
        self.assertTrue(torch.equal(cpu_rng, torch.get_rng_state()))
        self.assertEqual(np_rng, stream.rng.bit_generator.state)
        self.assertEqual(digest, [state_digest(m.state_dict()) for m in self.models])
        self.assertTrue(all(p.grad is None for m in self.models for p in m.parameters()))

    def test_diagnostics_do_not_change_model(self):
        for m in self.models:
            before = state_digest(m.state_dict())
            d = activation_diagnostic(m, *self.batch[:2])
            self.assertEqual(before, state_digest(m.state_dict()))
            self.assertLessEqual(d['attenuation_q10_q50_q90'][-1], 1.)
            if m.alpha == 0:
                self.assertEqual(d['attenuation_q10_q50_q90'], [1., 1., 1.])

    def test_summary_direction_and_gate_arithmetic(self):
        fit, meta = fixture()
        held = {k: v[:4] for k, v in fit.items()}
        ranked = np.array([[[2, 2, 2, 2], [2, 2, 2, 2]], [[1, 1, 1, 1], [1, 1, 1, 1]]], np.float32)
        result = summary(ranked, held, fit, meta)
        self.assertEqual(result['primary']['delta_mrr'], .5)
        self.assertEqual(result['primary']['pair_cluster_95'], [.5, .5])
        self.assertTrue(result['advance_gate'])
        reverse = summary(ranked[::-1], held, fit, meta)
        self.assertFalse(reverse['advance_gate'])
        self.assertFalse(result['holdout_gradients'])
        self.assertEqual(result['extra_parameters'], 0)


if __name__ == '__main__':
    unittest.main()
