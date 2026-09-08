"""Synthetic tests of TF1 indexing, positive filtering and explanations."""

import unittest

import numpy as np
import torch

from resonate import ResonatE
from biokg.mixed_operator import from_student
from biokg.train_forensics import (
    catalog_rank, choose_case, geometry, inspect, margin_parts, neighbors,
    structural_evidence, symmetric_graph,
)
from biokg.train_joint_operator import parameter_hashes


class TrainForensicsTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3681)
        torch.set_num_threads(4)
        student = ResonatE(20, 102, k=2, block=True, block_size=2)
        self.model = from_student(dict(model=student.state_dict(), args=dict(k=2, block_size=2)), 'single')
        self.model.eval().requires_grad_(False)

    def test_filter_other_positives_but_keep_focal_and_average_ties(self):
        scores = np.array([3., 4., 3., 2., 6.])
        self.assertEqual(catalog_rank(scores, 0, []), 3.5)
        self.assertEqual(catalog_rank(scores, 0, [0, 1, 4]), 1.5)
        self.assertEqual(catalog_rank(scores, 0, [0, 1, 2, 4]), 1.)

    def test_case_selection_first_near_miss_then_first_failure(self):
        rows = [dict(filtered_rank=r, index=i) for i, r in enumerate([1, 11, 3, 2])]
        self.assertEqual(choose_case(rows)['index'], 2)
        self.assertEqual(choose_case(rows[:2])['index'], 1)
        self.assertIsNone(choose_case(rows[:1]))

    def test_symmetric_pair_exclusion_and_duplicate_edges(self):
        graph = symmetric_graph([0, 0, 1, 2], [1, 1, 0, 1], 4)
        np.testing.assert_array_equal(neighbors(graph, 0), [1])
        np.testing.assert_array_equal(neighbors(graph, 1), [0, 2])
        np.testing.assert_array_equal(neighbors(graph, 0, (0, 1)), [])
        np.testing.assert_array_equal(neighbors(graph, 1, (0, 1)), [2])
        np.testing.assert_array_equal(neighbors(graph, 2, (0, 1)), [1])
        self.assertEqual(graph.nnz, 4)

    def test_margin_decomposition_exact_with_opposing_factors(self):
        parts = margin_parts([1, 3], [.8, .4], 2)
        self.assertGreater(parts['norm'], 0)
        self.assertLess(parts['alignment'], 0)
        self.assertAlmostEqual(sum(parts.values()), 2*(3*.4-1*.8))

    def test_geometry_reconstructs_logits_and_blocks(self):
        before = parameter_hashes(self.model)
        result = geometry(self.model, 0, 2, [5, 6])
        self.assertTrue(result['direct_score_reconstruction_passed'])
        contributions = np.asarray(result['block_contributions'])
        np.testing.assert_allclose(contributions.sum(1), result['scores'], atol=2e-5)
        self.assertAlmostEqual(sum(result['symmetric_margin_decomposition'].values()), result['winner_minus_positive'], places=5)
        self.assertEqual(before, parameter_hashes(self.model))

    def test_structure_removes_direct_training_pair(self):
        graph = symmetric_graph([0], [1], 8)
        result = structural_evidence(self.model, graph, 0, 1, 1, 2, 5)
        self.assertEqual(result['source_neighbors'], 0)
        self.assertEqual(result['candidate_holders'], 0)
        self.assertEqual(result['holder_cosine_top3'], [])
        self.assertEqual(result['candidate_analogy_top3'], [])

    def test_complete_synthetic_train_sample_is_deterministic_and_frozen(self):
        train = dict(head=np.array([0, 1, 2, 3]), tail=np.array([1, 2, 3, 4]),
                     relation=np.array([2, 2, 3, 3]), head_type=['drug']*4, tail_type=['drug']*4)
        before = parameter_hashes(self.model)
        result, scores = inspect(self.model, train, dict(drug=5), dict(drug=8))
        second, other = inspect(self.model, train, dict(drug=5), dict(drug=8))
        self.assertEqual(result, second)
        np.testing.assert_array_equal(scores, other)
        self.assertEqual(scores.shape, (8, 8))
        self.assertEqual(sorted(r['train_row'] for r in result['records']), [0, 0, 1, 1, 2, 2, 3, 3])
        self.assertTrue(all(r['filtered_rank'] <= r['raw_rank'] for r in result['records']))
        self.assertEqual(before, parameter_hashes(self.model))
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in self.model.parameters()))


if __name__ == '__main__':
    unittest.main()
