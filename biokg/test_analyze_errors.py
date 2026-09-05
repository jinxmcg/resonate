"""Checks for the validation-only error diagnostic."""
import unittest

import numpy as np
import torch

from biokg.analyze_errors import groups, membership, rank_statistics, score_validation, train_features
from resonate import ResonatE


class ErrorAnalysisTests(unittest.TestCase):
    def test_ties_and_top_negative(self):
        sp = torch.tensor([3., 2., 1.])
        sn = torch.tensor([[1., 2., 0.], [2., 3., 2.], [3., 2., 1.]])
        cand = torch.tensor([[5, 6, 7], [8, 9, 10], [11, 12, 13]])
        got = rank_statistics(sp, sn, cand)
        torch.testing.assert_close(got["rank"], torch.tensor([1., 3., 3.5]))
        torch.testing.assert_close(got["top_negative"], torch.tensor([6, 9, 11]))
        torch.testing.assert_close(got["ties"], torch.tensor([0, 2, 1]))

    def test_membership_boundaries(self):
        np.testing.assert_array_equal(membership(np.array([2, 4]), np.array([1, 2, 3, 4, 5])),
                                      [False, True, False, True, False])
        self.assertFalse(membership(np.array([], dtype=int), np.array([1])).any())

    def test_train_features_unique_edges_and_reverse_direction(self):
        # Duplicate 0->1 must not inflate degree; relation 0 has edges
        # 0->1, 0->2, 3->0. The evaluation queries are 0->3 and 3->0.
        train = dict(head=np.array([0, 0, 0, 3]), tail=np.array([1, 1, 2, 0]),
                     relation=np.zeros(4, dtype=int), head_type=["x"] * 4, tail_type=["x"] * 4)
        q = dict(rank=np.array([2., 1.]), rr=np.array([.5, 1.]), relation=np.array([0, 0]),
                 direction=np.array([0, 1]), source=np.array([0, 3]), target=np.array([3, 0]),
                 top_negative=np.array([2, 2]))
        collision = train_features(q, train, {"x": 0}, 4, {"x": 4})
        np.testing.assert_array_equal(q["source_relation_degree"], [2, 0])
        np.testing.assert_array_equal(q["target_relation_degree"], [0, 2])
        np.testing.assert_array_equal(q["source_degree"], [3, 1])
        np.testing.assert_array_equal(q["reverse_in_train"], [True, True])
        np.testing.assert_array_equal(q["positive_in_train"], [False, False])
        np.testing.assert_array_equal(q["top_negative_in_train"], [True, False])
        self.assertEqual(collision[0]["train_triples"], 3)
        self.assertAlmostEqual(collision[0]["expected_known_positives_per_4096"], 4096 * 5 / 3 / 4)
        rows = groups(q, np.array(["a", "b"]))
        self.assertEqual(sum(row["queries"] for row in rows), 2)
        self.assertAlmostEqual(sum(row["mrr_deficit_contribution"] for row in rows), .25)

    def test_scoring_direction_offsets_and_no_updates(self):
        torch.manual_seed(7)
        model = ResonatE(12, 2, k=2, block=True, block_size=2).eval()
        part = dict(head=np.array([0, 1]), tail=np.array([1, 2]), relation=np.array([0, 0]),
                    head_type=["a", "a"], tail_type=["b", "b"],
                    head_neg=np.array([[2, 3], [3, 4]]), tail_neg=np.array([[0, 2], [0, 3]]))
        before = {key: value.clone() for key, value in model.state_dict().items()}
        got = score_validation(model, part, {"a": 0, "b": 6}, 2, torch.device("cpu"), chunk=1)
        np.testing.assert_array_equal(got["source"], [0, 1, 7, 8])
        np.testing.assert_array_equal(got["target"], [7, 8, 0, 1])
        for i, candidates in enumerate(([6, 8], [6, 9], [2, 3], [3, 4])):
            with torch.no_grad():
                source = torch.tensor([got["source"][i]])
                rel = torch.tensor([0 if i < 2 else 1])
                z = model.out(model.hop(model.embed(source), rel), rel)
                scores = (z @ model.rows(torch.tensor([got["target"][i], *candidates])).conj().T).real
                expected = 1 + .5 * ((scores[0, 1:] > scores[0, 0]).sum() +
                                     (scores[0, 1:] >= scores[0, 0]).sum()).item()
                self.assertEqual(got["rank"][i], expected)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]))
        self.assertTrue(all(param.grad is None for param in model.parameters()))


if __name__ == "__main__":
    unittest.main()
