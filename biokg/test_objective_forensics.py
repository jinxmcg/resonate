"""TF2 synthetic checks: no real dataset/checkpoint access."""

import unittest

import numpy as np
import torch

from resonate import ResonatE, cnorm
from biokg.candidate_retrieval import cosine_cache
from biokg.joint_operator import retained_loss
from biokg.mixed_operator import from_student
from biokg.objective_forensics import (
    blend, gradient_measure, inspect_objective, pair_removed_graph, retrieval_features,
    sample_batch, select_cases, summarize,
)
from biokg.relation_analogy import normalize
from biokg.train_dual_operator import tensor_digest
from biokg.train_forensics import catalog_rank


class ObjectiveForensicsTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3682)
        torch.set_num_threads(4)
        base = ResonatE(12, 4, k=2, block=True, block_size=2)
        self.model = from_student(dict(model=base.state_dict(), args=dict(k=2, block_size=2)), 'single')
        self.model.eval().requires_grad_(False)

    def test_balanced_selection_and_pair_deduplication(self):
        rows = [dict(direction=d, source_local_id=s, positive_local_id=t, filtered_rank=r, sample_index=i)
                for i, (d, s, t, r) in enumerate([(0, 0, 1, 3), (0, 2, 3, 1), (0, 4, 5, 10),
                                                   (1, 1, 0, 4), (1, 6, 7, 2), (1, 8, 9, 11)])]
        self.assertEqual([r['sample_index'] for r in select_cases(rows, 2)], [0, 2, 4])
        self.assertEqual([r['sample_index'] for r in select_cases(rows, 1)], [0, 4])

    def test_remove_all_pair_copies_preserve_direction(self):
        h, t = np.array([0, 0, 1, 2, 3]), np.array([1, 1, 0, 0, 2])
        forward, removed = pair_removed_graph(h, t, 4, 0, 1, 0)
        reverse, other = pair_removed_graph(h, t, 4, 0, 1, 1)
        self.assertEqual(removed, 3)
        self.assertEqual(other, 3)
        np.testing.assert_array_equal(forward.toarray(), reverse.T.toarray())
        self.assertEqual(forward[2, 0], 1)
        self.assertEqual(forward[0, 2], 0)
        self.assertEqual(forward.nnz, 2)

    def test_blend_exact_nested_arithmetic(self):
        raw = np.random.default_rng(5).normal(size=(7, 17)).astype(np.float32)
        weights = np.array([.1, .2, .3, .15, .25], np.float32)
        bc = dict(groups={'0/1': dict(weights=weights.tolist())})
        cs = dict(groups={'0/1': dict(beta=.1)})
        score, z = blend(raw, bc, cs, 0, 1)
        np.testing.assert_array_equal(z, normalize(raw))
        features = [z[0]] + [.5*z[i]+.5*z[0] for i in range(1, 5)]
        base = np.zeros(17, np.float32)
        for weight, feature in zip(weights, features):
            base += weight*feature
        expected = (np.float32(1)-np.float32(.1))*base+np.float32(.1)*((z[5]+z[6])*np.float32(.5))
        np.testing.assert_array_equal(score, expected)
        cs['groups']['0/1']['beta'] = 0.
        np.testing.assert_array_equal(blend(raw, bc, cs, 0, 1)[0], base)

    def test_retrieval_matches_independent_holder_calculation(self):
        graph, _ = pair_removed_graph([0, 0, 1, 2, 3, 4, 4], [1, 2, 3, 3, 2, 2, 5], 6, 0, 1, 0)
        with torch.inference_mode():
            table = cnorm(self.model.E[4:10])
            cosine = cosine_cache(table)
            actual = retrieval_features(self.model, graph, 0, 0, 4, cosine)
        dense = graph.toarray().astype(bool)
        for candidate in range(6):
            holders = np.flatnonzero(dense[:, candidate])
            holders = holders[holders != 0]
            sims = cosine[0, holders]
            js = []
            for holder in holders:
                union = np.logical_or(dense[0], dense[holder]).sum()
                js.append(np.logical_and(dense[0], dense[holder]).sum()/union if union else 0.)
            if len(holders):
                top, jac = np.sort(sims)[-3:], np.sort(js)[-3:]
                expected = [top[-1]**3, top.mean()**3, jac[-1], jac.mean()]
            else:
                expected = [-1.]*4
            np.testing.assert_allclose(actual[1:5, candidate], expected, atol=1e-6)

    def test_conditional_sampling_forced_pool_and_source_roles(self):
        train = dict(head=np.array([0, 1, 2, 3]), tail=np.array([1, 2, 3, 4]))
        for direction in (0, 1):
            case = dict(train_row=1, direction=direction, directed_relation=direction*2, winner_local_id=5)
            got = sample_batch(train, np.arange(4), case, 3, 4, 6, batch_size=8, negative_count=5)
            other = sample_batch(train, np.arange(4), case, 3, 4, 6, batch_size=8, negative_count=5)
            for key in got:
                np.testing.assert_array_equal(got[key], other[key])
            self.assertEqual(got['source'][0], (1 if direction == 0 else 2)+4)
            self.assertEqual(got['positive'][0], (2 if direction == 0 else 1)+4)
            self.assertTrue(((got['negatives_random'] >= 4) & (got['negatives_random'] < 10)).all())
            self.assertIn(9, got['negatives_forced'])
            np.testing.assert_array_equal(got['negatives_random'][1:], got['negatives_forced'][1:])

    def batch(self):
        return torch.tensor([0, 1, 2]), torch.zeros(3, dtype=torch.long), torch.tensor([5, 6, 7]), torch.tensor([4, 8, 9, 10])

    def test_gradient_projections_parity_and_unchanged_values(self):
        student = self.model.a.requires_grad_(True)
        before = tensor_digest(student.state_dict())
        result = inspect_objective(student, self.batch(), torch.randn(3, 5), 8)
        self.assertTrue(result['summed_gradient_parity_passed'])
        for scope in ('focal', 'batch'):
            for group in ('entities', 'operators', 'temperature', 'representation'):
                gains = [result['scopes'][scope][c]['groups'][group]['descent_margin_change'] for c in ('ce', 'kd', 'trajectory', 'total')]
                self.assertAlmostEqual(sum(gains[:3]), gains[3], places=5)
            self.assertEqual(result['scopes'][scope]['total']['groups']['temperature']['descent_margin_change'], 0.)
        self.assertEqual(before, tensor_digest(student.state_dict()))
        self.assertTrue(all(p.grad is None for p in student.parameters()))

    def test_projection_matches_functional_finite_difference(self):
        student = self.model.a.requires_grad_(True)
        batch = self.batch()
        outputs = student.training_outputs(*batch)
        margin = (outputs[1][0]*(student.E[5]-student.E[8]).conj()).sum().real
        params = student.E, student.H_b, student.log_tau
        gm = torch.autograd.grad(margin, params, retain_graph=True, allow_unused=True)
        loss, _ = retained_loss(outputs, torch.randn(3, 5))
        gl = torch.autograd.grad(loss, params)
        projected = gradient_measure(gm, gl)['representation']['descent_margin_change']
        def value(step):
            # Functional temporary tensors only; never mutate even the fixture.
            e = student.E.detach().to(torch.complex128)-step*gl[0].to(torch.complex128)
            h = student.H_b.detach().to(torch.complex128)-step*gl[1].to(torch.complex128)
            x = cnorm(e[[0]]).reshape(1, -1, 2)
            q = cnorm(torch.einsum('bkij,bkj->bki', h[[0]], x).reshape(1, -1))[0]
            return float((q*(e[5]-e[8]).conj()).sum().real)
        finite = (value(1e-5)-value(-1e-5))/(2e-5)
        np.testing.assert_allclose(projected, finite, atol=2e-5, rtol=2e-4)

    def test_teacher_targets_must_be_frozen(self):
        self.model.a.requires_grad_(True)
        with self.assertRaises(ValueError):
            inspect_objective(self.model.a, self.batch(), torch.randn(3, 5, requires_grad=True), 8)

    def test_catalog_ranking_filters_other_train_answers(self):
        self.assertEqual(catalog_rank([.5, .7, .5, .9], 0, [0, 3]), 2.5)


if __name__ == '__main__':
    unittest.main()
