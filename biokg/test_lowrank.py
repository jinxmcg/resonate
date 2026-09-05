"""Numerical checks for H29. Run from the repository: python -m unittest discover -s biokg -p test_lowrank.py."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from resonate import ResonatE, cnorm
from resonate_wiki import SparseTableResonatE, score_batch

torch.set_num_threads(2)


class LowRankTests(unittest.TestCase):
    def test_zero_correction_preserves_baseline_and_rng(self):
        for cls in (ResonatE, SparseTableResonatE):
            for local in (False, True):
                kwargs = dict(k=4, block_size=4)
                if cls is ResonatE:
                    kwargs['block'] = True
                torch.manual_seed(17)
                base = cls(80, 6, **kwargs)
                next_base = torch.randn(5)
                torch.manual_seed(17)
                model = cls(80, 6, low_rank=2, low_rank_local=local, **kwargs)
                self.assertTrue(torch.equal(next_base, torch.randn(5)))
                for key, value in base.state_dict().items():
                    self.assertTrue(torch.equal(value, model.state_dict()[key]), key)
                src, rel = torch.arange(8), torch.arange(8) % 6
                z = model.hop(model.embed(src), rel)
                self.assertTrue(torch.equal(z, base.hop(base.embed(src), rel)))
                logits, _, _ = score_batch(model, src, rel, src + 1, torch.arange(20))
                torch.nn.functional.cross_entropy(logits, torch.zeros(8, dtype=torch.long)).backward()
                self.assertTrue(torch.isfinite(model.lr_v.grad).all())
                self.assertGreater(model.lr_v.grad.abs().sum().item(), 0)
                if cls is SparseTableResonatE:
                    self.assertTrue(model.E_real.grad.is_sparse)

    def test_nonzero_operator_and_gradients_match_explicit_matrix(self):
        for local in (False, True):
            torch.manual_seed(3)
            model = ResonatE(20, 3, k=4, block=True, block_size=4,
                             low_rank=2, low_rank_local=local)
            with torch.no_grad():
                model.lr_v.normal_(std=0.1)
            z = torch.randn(5, 16, dtype=torch.cfloat, requires_grad=True)
            rel = torch.tensor([0, 2, 1, 2, 0])
            matrices = []
            for r in rel:
                base = torch.block_diag(*model.H[r].unbind())
                u, v = model.lr_u[r], model.lr_v[r]
                if local:
                    u, v = u.reshape(4, 4, 2), v.reshape(4, 4, 2)
                    residual = torch.block_diag(*(u @ v.conj().transpose(-1, -2)).unbind())
                else:
                    residual = u @ v.conj().T
                matrices.append(base + residual)
            expected = cnorm(torch.einsum('bij,bj->bi', torch.stack(matrices), z))
            actual = model.hop(z, rel)
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
            weight = torch.randn_like(actual)
            params = (z, model.H, model.lr_u, model.lr_v)
            ga = torch.autograd.grad((actual * weight.conj()).real.sum(), params, retain_graph=True)
            ge = torch.autograd.grad((expected * weight.conj()).real.sum(), params)
            for a, e in zip(ga, ge):
                torch.testing.assert_close(a, e, atol=2e-6, rtol=2e-5)

    def test_global_connects_blocks_but_control_does_not(self):
        for local in (False, True):
            model = ResonatE(20, 2, k=4, block=True, block_size=4,
                             low_rank=2, low_rank_local=local)
            with torch.no_grad():
                model.H.copy_(torch.eye(4).expand_as(model.H))
                model.lr_u.zero_(); model.lr_v.zero_()
                model.lr_u[0, 8, 0] = 1
                model.lr_v[0, 0, 0] = 1
            z = torch.zeros(1, 16, dtype=torch.cfloat); z[0, 0] = 1
            result = model.hop(z, torch.tensor([0]))
            self.assertEqual(bool(result[0, 8].abs() > 0), not local)

    def test_parameter_count_and_legacy_checkpoints(self):
        base = ResonatE(80, 6, k=4, block=True, block_size=4)
        base.load_state_dict(base.state_dict(), strict=True)
        self.assertFalse(any(key.startswith('lr_') for key in base.state_dict()))
        for local in (False, True):
            model = ResonatE(80, 6, k=4, block=True, block_size=4,
                             low_rank=2, low_rank_local=local)
            self.assertEqual(model.n_params() - base.n_params(), 4 * 6 * 16 * 2)

    def test_conversion_and_teacher_loading_preserve_scores(self):
        from train_biokg_comp import load_teacher
        script = Path(__file__).with_name('sparse_to_dense.py')
        for local in (False, True):
            model = SparseTableResonatE(80, 6, k=4, block_size=4,
                                       low_rank=2, low_rank_local=local)
            with torch.no_grad():
                model.lr_v.normal_(std=0.1)
            args = dict(k=4, block_size=4, low_rank=2, low_rank_local=local, shell='sparse')
            with tempfile.TemporaryDirectory(prefix='biokg-lowrank-') as directory:
                src, dst = [os.path.join(directory, name) for name in ('source.pt', 'dense.pt')]
                torch.save(dict(model=model.state_dict(), args=args, n_rel=6, offset={'entity': 0}), src)
                subprocess.run([sys.executable, str(script), src, dst], check=True, capture_output=True)
                restored = load_teacher(dst, 80, 6, torch.device('cpu'))
                h, r = torch.arange(8), torch.arange(8) % 6
                a, _, _ = score_batch(model, h, r, h + 1, torch.arange(20))
                b, _, _ = score_batch(restored, h, r, h + 1, torch.arange(20))
                torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)

    def test_validation_loader_never_opens_test_split(self):
        from train_biokg_comp import load
        fake = unittest.mock.MagicMock()
        fake.__getitem__.return_value = {'num_nodes_dict': {'entity': 80}}
        fake.root = '/unused/ogbl_biokg'; fake.meta_info = {'split': 'random'}
        with patch('ogb.linkproppred.LinkPropPredDataset', return_value=fake), \
                patch('train_biokg_comp.torch.load', return_value={}) as reader:
            splits, _, _, _, _ = load('/unused', include_test=False)
        self.assertEqual(set(splits), {'train', 'valid'})
        self.assertEqual([Path(call.args[0]).name for call in reader.call_args_list], ['train.pt', 'valid.pt'])
        fake.get_edge_split.assert_not_called()

    def test_eval_only_restores_architecture_and_table_format(self):
        import train_biokg_comp as trainer
        triples = dict(head=np.arange(12), tail=np.arange(12) + 1,
                       relation=np.arange(12) % 3,
                       head_type=np.full(12, 'entity'), tail_type=np.full(12, 'entity'))
        for cls in (ResonatE, SparseTableResonatE):
            kwargs = dict(k=4, block_size=4, low_rank=2, low_rank_local=True)
            model = cls(80, 6, **kwargs, **({'block': True} if cls is ResonatE else {}))
            with torch.no_grad():
                model.lr_v.normal_(std=0.1)
            with tempfile.TemporaryDirectory(prefix='biokg-eval-') as directory:
                filename = os.path.join(directory, 'model.pt')
                torch.save(dict(model=model.state_dict(), args=kwargs, n_rel=6,
                                offset={'entity': 0}), filename)
                argv = ['trainer', '--eval-only', '--eval', 'valid', '--save', filename]
                with patch.object(sys, 'argv', argv), \
                        patch.object(trainer, 'load', return_value=(
                            {'train': triples, 'valid': {}}, {'entity': 0}, 80,
                            ['entity'], {'entity': 80})) as loader, \
                        patch.object(trainer, 'eval_split', return_value=0.0) as evaluator:
                    trainer.main()
                self.assertFalse(loader.call_args.kwargs['include_test'])
                restored = evaluator.call_args.args[0]
                self.assertIsInstance(restored, cls)
                self.assertEqual(restored.low_rank, 2)
                self.assertTrue(restored.low_rank_local)
                for key, value in model.state_dict().items():
                    self.assertTrue(torch.equal(value, restored.state_dict()[key]), key)


if __name__ == '__main__':
    unittest.main()
