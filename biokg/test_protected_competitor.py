"""TF3 synthetic tests; no dataset/checkpoint access or parameter edits."""

import copy
import unittest

import numpy as np
import torch

from resonate import ResonatE
from biokg.joint_operator import retained_loss
from biokg.mixed_operator import from_student
from biokg.protected_competitor import (
    ARMS, candidate_rows, complete_probes, group_stats, inspect_arm, matched_random,
    probe_readouts, probe_specs, probe_tensors, protected_loss, protection_mask, summarize,
)
from biokg.train_dual_operator import tensor_digest
from biokg.train_forensics import symmetric_graph


class ProtectedCompetitorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3683)
        torch.set_num_threads(4)
        base = ResonatE(12, 4, k=2, block=True, block_size=2)
        self.model = from_student(dict(model=base.state_dict(), args=dict(k=2, block_size=2)), 'single').a.requires_grad_(True)
        self.batch = (torch.tensor([0, 1, 2]), torch.zeros(3, dtype=torch.long),
                      torch.tensor([5, 6, 7]), torch.tensor([4, 8, 8, 10]))
        self.probes = [dict(kind='focal_model' if i == 0 else 'background', source=i,
                            relation=0, positive=5+i, negative=8) for i in range(3)]

    def test_symmetric_known_rows_duplicates_and_missing(self):
        graph = symmetric_graph([0, 0, 3], [1, 1, 2], 6)
        np.testing.assert_array_equal(candidate_rows(graph, [1, 1, 2, 4, 0], 0), [1, 1, 0, 0, 0])
        np.testing.assert_array_equal(candidate_rows(graph, [0, 3, 5], 2), [0, 1, 0])

    def test_mask_protects_only_chosen_id_and_not_positive_column(self):
        mask = protection_mask([True, False, True], [8, 4, 8, 9], 8, 'cpu')
        np.testing.assert_array_equal(mask.numpy(), [[0, 1, 0, 1, 0], [0, 0, 0, 0, 0], [0, 1, 0, 1, 0]])
        self.assertEqual(mask.dtype, torch.bool)

    def test_random_pool_exact_slots_and_determinism(self):
        graph = symmetric_graph([0, 0], [1, 2], 10)
        pool = np.array([105, 103, 105, 104, 104])
        chosen, got = matched_random(pool, 105, graph, 0, 100, 36830)
        other, repeated = matched_random(pool, 105, graph, 0, 100, 36830)
        self.assertEqual(chosen, other)
        np.testing.assert_array_equal(got, repeated)
        np.testing.assert_array_equal(got[pool != 105], pool[pool != 105])
        np.testing.assert_array_equal(got == chosen, pool == 105)
        self.assertNotIn(chosen, [101, 102, *pool])
        self.assertGreaterEqual(chosen, 100)
        self.assertLess(chosen, 110)

    def test_random_pool_no_eligible_candidate_fails(self):
        graph = symmetric_graph([0], [1], 3)
        with self.assertRaises(ValueError):
            matched_random([100, 102], 102, graph, 0, 100, 1)

    def test_empty_protection_is_exact_original_loss(self):
        outputs = self.model.training_outputs(*self.batch)
        targets = torch.randn(3, 5)
        total, parts = protected_loss(outputs, targets, torch.zeros(3, 5, dtype=torch.bool))
        expected, original = retained_loss(outputs, targets)
        self.assertTrue(torch.equal(total, expected))
        self.assertTrue(torch.equal(parts['ce'], original['ce']))
        self.assertTrue(torch.equal(parts['kd'], original['kd_t_squared']))
        self.assertTrue(torch.equal(parts['trajectory'], .1*original['trajectory']))

    def test_ce_zero_masked_derivative_kd_and_trajectory_unmodified(self):
        outputs = self.model.training_outputs(*self.batch)
        targets = torch.randn(3, 5)
        mask = protection_mask([True, False, True], self.batch[3].numpy(), 8, 'cpu')
        total, parts = protected_loss(outputs, targets, mask)
        _, original = retained_loss(outputs, targets)
        g = torch.autograd.grad(parts['ce'], outputs[0], retain_graph=True)[0]
        self.assertEqual(int(torch.count_nonzero(g[mask])), 0)
        self.assertTrue((g[:, 0] < 0).all())
        self.assertTrue((g[:, 1] > 0).all())  # Other candidates not masked.
        self.assertTrue(torch.equal(parts['kd'], original['kd_t_squared']))
        self.assertTrue(torch.equal(parts['trajectory'], .1*original['trajectory']))
        self.assertTrue(torch.equal(total, parts['ce']+parts['kd']+parts['trajectory']))
        self.assertLess(float(parts['ce'].detach()), float(original['ce'].detach()))

    def test_bad_mask_and_trainable_teacher_rejected(self):
        outputs = self.model.training_outputs(*self.batch)
        for mask in (torch.zeros(3, 4, dtype=torch.bool), torch.zeros(3, 5), torch.ones(3, 5, dtype=torch.bool)):
            with self.assertRaises(ValueError):
                protected_loss(outputs, torch.randn(3, 5), mask)
        with self.assertRaises(ValueError):
            protected_loss(outputs, torch.randn(3, 5, requires_grad=True), torch.zeros(3, 5, dtype=torch.bool))

    def test_jvp_reverse_mode_finite_difference_and_unchanged_state(self):
        before = tensor_digest(self.model.state_dict())
        target = torch.randn(3, 5)
        mask = protection_mask([False, True, False], self.batch[3].numpy(), 8, 'cpu')
        result = inspect_arm(self.model, self.batch, target, mask, self.probes)
        loss, _ = protected_loss(self.model.training_outputs(*self.batch), target, mask)
        grads = torch.autograd.grad(loss, (self.model.E, self.model.H_b))
        ids = probe_tensors(self.probes, 'cpu')
        e, h = self.model.E.detach().to(torch.complex128), self.model.H_b.detach().to(torch.complex128)
        ge, gh = [g.to(torch.complex128) for g in grads]
        eps = 1e-5
        finite = (probe_readouts(e-eps*ge, h-eps*gh, *ids)-probe_readouts(e+eps*ge, h+eps*gh, *ids))/(2*eps)
        np.testing.assert_allclose(result['total_change'], finite.numpy(), atol=2e-5, rtol=3e-4)
        np.testing.assert_allclose(result['total_change'], np.asarray(result['entity_change'])+result['operator_change'], atol=2e-6)
        self.assertEqual(before, tensor_digest(self.model.state_dict()))
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))
        self.assertTrue(result['jvp_reverse_mode_parity'])

    def test_probe_order_uniqueness_and_empty_group(self):
        case = dict(source_local_id=0, positive_local_id=1, winner_local_id=5, pipeline_winner_local_id=6)
        sources, positives = [100, 101, 101, 102, 103, 104], [101, 102, 103, 103, 104, 105]
        probes = probe_specs(case, sources, positives, [0, 0, 1, 1, 1, 0], [0]*6, 109, 100, 2, 2)
        self.assertEqual([p['source'] for p in probes if p['kind'] == 'background'], [101, 102])
        self.assertEqual([p['source'] for p in probes if p['kind'] == 'hard_known'], [101, 102])
        self.assertTrue(all(p['positive'] == 105 for p in probes if p['kind'] == 'hard_known'))
        self.assertFalse(any(p['kind'] == 'random_known' for p in probes))

    def test_catalog_probe_filtering_and_score_parity(self):
        graph = symmetric_graph([0, 0, 1], [1, 2, 3], 8)
        specs = [dict(kind='background', source=0, positive=1, negative=None)]
        got = complete_probes(self.model, graph, copy.deepcopy(specs), 0, 0)
        self.assertEqual(got, complete_probes(self.model, graph, copy.deepcopy(specs), 0, 0))
        self.assertNotIn(got[0]['negative'], [1, 2])
        with torch.no_grad():
            ids = probe_tensors(got, 'cpu')
            readouts = probe_readouts(self.model.E, self.model.H_b, *ids)
            outputs = self.model.training_outputs(ids[0], ids[1], ids[2], ids[3])
            np.testing.assert_allclose(readouts[0, :2].numpy(), (outputs[0]/self.model.log_tau.exp())[0].numpy(), atol=1e-6)

    def test_equal_case_summary_not_pooled_probe_average(self):
        cases, records = [{}, {}], []
        for i in range(2):
            probes = [dict(kind='background')]*(i+1)
            values = [[float(i), 0., float(i), 0., 0.]]*(i+1)
            for arm in ARMS:
                records.append(dict(case_index=i, arm=arm, protected_rows=0, groups=group_stats(probes, values)))
        result = summarize(cases, records)
        group = result['arms']['baseline']['groups']['background']
        self.assertEqual(group['equal_case_mean_margin_change'], .5)
        self.assertEqual(group['probe_occurrences'], 3)
        self.assertEqual(result['arms']['baseline']['groups']['hard_known']['cases'], 0)
        self.assertIsNone(result['arms']['baseline']['groups']['hard_known']['equal_case_mean_margin_change'])
        self.assertFalse(result['training_started'])


if __name__ == '__main__':
    unittest.main()
