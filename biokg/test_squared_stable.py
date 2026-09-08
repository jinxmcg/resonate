"""H37N zero/underflow normalization and runner-adapter regression tests."""

import tempfile
from pathlib import Path
import unittest

import torch

from resonate import cnorm
from biokg.masked_neighborhood import make_optimizers, state_digest
from biokg.squared_operator import objective, make_pair as old_pair, update
from biokg.squared_operator_stable import (
    SPEC, make_pair, stable_cnorm, snapshot, restore,
)


class StableTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4)
        torch.manual_seed(37057)

    def test_fp32_forward_exact_at_zero_underflow_and_normal(self):
        z = torch.randn(4, 144, dtype=torch.complex64)
        z[0].zero_()
        z[1].mul_(1e-25)
        z[2].mul_(1e-19)
        self.assertTrue(torch.equal(cnorm(z), stable_cnorm(z)))

    def test_finite_backward_at_underflow_where_old_fails(self):
        z = (torch.randn(3, 144, dtype=torch.complex64)*1e-25).requires_grad_()
        old = torch.autograd.grad(cnorm(z).real.sum(), z)[0]
        new = torch.autograd.grad(stable_cnorm(z).real.sum(), z)[0]
        self.assertFalse(torch.isfinite(old).all())
        self.assertTrue(torch.isfinite(new).all())

    def test_zero_derivative_is_identity_over_epsilon(self):
        z = torch.zeros(2, 4, dtype=torch.complex64, requires_grad=True)
        grad = torch.autograd.grad(stable_cnorm(z).real.sum(), z)[0]
        torch.testing.assert_close(grad.real, torch.full_like(grad.real, 1e8), atol=0, rtol=0)
        self.assertTrue((grad.imag == 0).all())

    def test_ordinary_value_and_gradient_exact(self):
        z = torch.randn(7, 144, dtype=torch.complex64, requires_grad=True)
        a, b = cnorm(z), stable_cnorm(z)
        self.assertTrue(torch.equal(a, b))
        ga = torch.autograd.grad(a.real.square().sum(), z)[0]
        gb = torch.autograd.grad(b.real.square().sum(), z)[0]
        self.assertTrue(torch.equal(ga, gb))

    def test_models_storage_loss_and_regular_gradients_match_original(self):
        old = old_pair(64, 2, 'cpu')
        new = make_pair(64, 2, 'cpu')
        batch = (torch.arange(4), torch.zeros(4, dtype=torch.long), torch.arange(16, 20), torch.arange(22, 38))
        for a, b in zip(old, new):
            b.load_state_dict(a.state_dict())
            la, _ = objective(a, batch, (16, 32))
            lb, _ = objective(b, batch, (16, 32))
            self.assertTrue(torch.equal(la, lb))
            ga = torch.autograd.grad(la, tuple(a.parameters()))
            gb = torch.autograd.grad(lb, tuple(b.parameters()))
            for x, y in zip(ga, gb):
                self.assertTrue(torch.equal(x.to_dense() if x.is_sparse else x, y.to_dense() if y.is_sparse else y))

    def test_checkpoint_preserves_repair_and_config(self):
        models = make_pair(64, 2, 'cpu')
        batch = (torch.arange(4), torch.zeros(4, dtype=torch.long), torch.arange(16, 20), torch.arange(22, 38))
        with tempfile.TemporaryDirectory(prefix='h37n-test-') as directory:
            for m in models:
                opts = make_optimizers(m)
                update(m, opts, batch, (16, 32))
                snapshot(m, opts, Path(directory), 1)
                got = restore(Path(directory)/(m.arm+'.pt'))
                self.assertEqual(state_digest(got.state_dict()), state_digest(m.state_dict()))
                self.assertEqual(type(got), type(m))
        self.assertEqual(SPEC['epsilon'], 1e-6)

    def test_adapter_wiring_in_fresh_subprocess(self):
        import subprocess
        import sys
        code = ('from biokg import squared_operator_run as r; '
                'from biokg.squared_operator_stable import configured_runner, SPEC, make_pair; '
                'old_update=r.update; r=configured_runner(); '
                'assert r.SPEC == SPEC and r.make_pair is make_pair and r.update is old_update; '
                'assert r.SPEC["protocol"] == "H37N"')
        subprocess.run([sys.executable, '-c', code], check=True)


if __name__ == '__main__':
    unittest.main()
