"""RS1 synthetic checks; never open real datasets or checkpoints."""

import unittest

import numpy as np
import torch

from resonate import ResonatE
from biokg.candidate_retrieval import apply_beta, fit_beta
from biokg.confirm_feature_blend import insert_report, split_masks
from biokg.mixed_operator import from_student
from biokg.relation_analogy import rank_rows
from biokg.reverse_scoring import (
    apply_targeted, build_reverse, cached_scores, check_path, cs_scores,
    direct_reverse, fold_stats, inverse_relation, query_data, reverse_table,
    screen_passed, summarize,
)
from biokg.train_joint_operator import parameter_hashes


class ReverseScoringTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3660)
        torch.set_num_threads(4)
        base = ResonatE(20, 4, k=2, block=True, block_size=2)
        self.model = from_student(dict(model=base.state_dict(), args=dict(k=2, block_size=2)), 'single')
        self.model.eval().requires_grad_(False)
        self.rng = np.random.default_rng(3660)

    def test_inverse_involution_and_rejection(self):
        relations = np.arange(4)
        np.testing.assert_array_equal(inverse_relation(relations, 4), [2, 3, 0, 1])
        np.testing.assert_array_equal(inverse_relation(inverse_relation(relations, 4), 4), relations)
        for rel, count in (([-1], 4), ([4], 4), ([0], 3), ([0], 0)):
            with self.assertRaises(ValueError):
                inverse_relation(rel, count)

    def test_cache_matches_direct_inverse_not_forward(self):
        source = np.array([0, 1, 2])
        candidates = np.array([[5, 7, 8, 9], [6, 7, 10, 11], [5, 8, 12, 7]])
        for rel in range(4):
            reverse = int(inverse_relation(rel, 4))
            table = reverse_table(self.model, reverse, 5, 8, chunk=3)
            got = cached_scores(self.model, table, source, candidates-5, chunk=2)
            expected = direct_reverse(self.model, source, np.full(3, reverse), candidates)
            np.testing.assert_allclose(got, expected, atol=1e-5, rtol=1e-5)
            wrong = direct_reverse(self.model, source, np.full(3, rel), candidates)
            self.assertFalse(np.allclose(got, wrong))

    def test_candidate_permutation_duplicates_extension_and_chunks(self):
        table = reverse_table(self.model, 2, 5, 8, chunk=3)
        np.testing.assert_allclose(table.numpy(), reverse_table(self.model, 2, 5, 8, chunk=8).numpy(), atol=1e-7)
        source = np.array([0, 1, 2])
        candidates = np.tile([0, 3, 2, 3, 6], (3, 1))
        actual = cached_scores(self.model, table, source, candidates, chunk=3)
        np.testing.assert_array_equal(actual, cached_scores(self.model, table, source, candidates, chunk=1))
        order = np.array([4, 1, 0, 3, 2])
        np.testing.assert_array_equal(actual[:, order], cached_scores(self.model, table, source, candidates[:, order]))
        extended = np.concatenate([candidates, np.tile([1, 7], (3, 1))], axis=1)
        np.testing.assert_array_equal(actual, cached_scores(self.model, table, source, extended)[:, :5])
        np.testing.assert_array_equal(actual[:, 1], actual[:, 3])
        for bad in (-1, 8):
            with self.assertRaises(ValueError):
                cached_scores(self.model, table, source[:1], np.array([[bad]]))

    def test_raw_target_norm_is_retained(self):
        table = reverse_table(self.model, 2, 5, 8)
        source, candidates = np.array([0]), np.array([[0, 3, 4, 7]])
        before = cached_scores(self.model, table, source, candidates)
        with torch.no_grad():
            self.model.a.E[0] *= 3
        after = cached_scores(self.model, table, source, candidates)
        np.testing.assert_allclose(after, 3*before, atol=1e-6, rtol=1e-5)
        np.testing.assert_allclose(after, direct_reverse(self.model, source, np.array([2]), candidates+5), atol=1e-5)

    def test_indexing_list_types_both_directions_and_frozen_model(self):
        valid = dict(head=np.array([0, 2, 1]), tail=np.array([3, 1, 4]), relation=np.array([0, 1, 1]),
            head_type=['drug', 'other', 'drug'], tail_type=['drug', 'drug', 'drug'],
            head_neg=np.tile(np.arange(500)%8, (3, 1)), tail_neg=np.tile(np.arange(500)%8, (3, 1)))
        data = query_data(valid, dict(drug=5, other=0), 4)
        np.testing.assert_array_equal(data['rows'], [0, 2, 3, 5])
        np.testing.assert_array_equal(data['source'], [5, 6, 8, 9])
        np.testing.assert_array_equal(data['relation'], [0, 1, 2, 3])
        np.testing.assert_array_equal(data['reverse_relation'], [2, 3, 0, 1])
        np.testing.assert_array_equal(data['candidates'][:, 0], [8, 9, 5, 6])
        self.assertIsInstance(valid['head_type'], list)
        before = parameter_hashes(self.model)
        output = np.empty((4, 501), np.float32)
        audit = build_reverse(self.model, data, dict(drug=5), dict(drug=8), output)
        self.assertEqual(len(audit), 4)
        self.assertTrue(all(a['permutation_exact'] for a in audit))
        np.testing.assert_allclose(output, direct_reverse(self.model, data['source'], data['reverse_relation'], data['candidates']), atol=1e-5, rtol=1e-5)
        self.assertEqual(parameter_hashes(self.model), before)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in self.model.parameters()))

    def metadata(self):
        return dict(relation=np.tile(np.repeat([0, 1], 20), 2), direction=np.repeat([0, 1], 40),
                    family=np.tile(np.repeat(['drug-drug', 'other'], 20), 2))

    def test_nested_cs1_arithmetic_is_exact(self):
        base, pair = [self.rng.normal(size=(80, 9)).astype(np.float32) for _ in range(2)]
        meta = self.metadata()
        recipe = dict(groups={f'{r}/{d}': dict(beta=b) for r, d, b in ((0, 0, 0), (0, 1, .025), (1, 0, .1), (1, 1, .2))})
        scores = cs_scores(base, pair, recipe, meta['relation'], meta['direction'])
        expected = apply_beta(base, pair, recipe, meta['relation'], meta['direction'], np.ones(80, bool))
        np.testing.assert_array_equal(rank_rows(scores), expected)
        np.testing.assert_array_equal(scores[:20], base[:20])

    def test_target_only_fit_and_zero_alpha_control(self):
        base = self.rng.normal(size=(80, 9)).astype(np.float32)
        meta, fit = self.metadata(), split_masks(40, 0)[0]
        target = np.flatnonzero(meta['family'] == 'drug-drug')
        reverse = self.rng.normal(size=(len(target), 9)).astype(np.float32)
        for alpha in (0., .2):
            recipe = dict(groups={f'0/{d}': dict(beta=alpha) for d in (0, 1)})
            control, treatment = apply_targeted(base, reverse, target, recipe, meta, ~fit)
            np.testing.assert_array_equal(control, rank_rows(base[~fit]))
            other = meta['family'][~fit] != 'drug-drug'
            np.testing.assert_array_equal(control[other], treatment[other])
            if alpha == 0:
                np.testing.assert_array_equal(control, treatment)

    def test_report_poisoning_sparse_fallback_and_ties(self):
        base, extra = [self.rng.normal(size=(80, 9)).astype(np.float32) for _ in range(2)]
        meta = self.metadata()
        for fit in split_masks(40, 0):
            recipe = fit_beta(base, extra, fit, **meta, min_rows=8)
            poisoned_base, poisoned_extra = base.copy(), extra.copy()
            poisoned_base[~fit, 0] += 1000
            poisoned_extra[~fit, 0] -= 1000
            self.assertEqual(recipe, fit_beta(poisoned_base, poisoned_extra, fit, **meta, min_rows=8))
            tied = fit_beta(np.ones_like(base), np.ones_like(extra), fit, **meta, min_rows=1000)
            self.assertTrue(all(g['beta'] == 0 and g['level'] == 'global' for g in tied['groups'].values()))

    def test_average_ties_and_paired_fold_statistics(self):
        np.testing.assert_array_equal(rank_rows(np.array([[1, 1, 0], [0, 1, 0]], np.float32)), [1.5, 2.5])
        fit = np.array([False, True, False, True]*2)
        control, treatment = np.array([2., 4., 4., 2.]), np.ones(4)
        result = fold_stats(control, treatment, fit, np.array(['drug-drug']*8), 0, 0, replicates=100)
        self.assertEqual(result['primary']['delta_mrr'], .625)
        self.assertEqual(result['primary']['bootstrap_95'], [.625, .625])
        self.assertEqual(result['recovered_from_rank2_to10'], 4)
        self.assertEqual(result['lost_top1'], 0)

    def test_screen_gate_boundary(self):
        def record(delta, lower):
            return dict(primary=dict(delta_mrr=delta, bootstrap_95=[lower, .001]))
        self.assertFalse(screen_passed(record(.000499, .0001)))
        self.assertFalse(screen_passed(record(.0005, 0)))
        self.assertTrue(screen_passed(record(.0005, .00001)))

    def test_failed_screen_has_no_fabricated_aggregate(self):
        ranks = {a: np.full((3, 8), np.nan) for a in ('baseline', 'reverse_pipeline')}
        coverage = {a: np.zeros((3, 8), np.uint8) for a in ranks}
        fit = np.array([False, True, False, True]*2)
        for a in ranks:
            insert_report(ranks[a][0], coverage[a][0], fit, np.ones(4))
        folds = [dict(primary=dict(delta_mrr=0., bootstrap_95=[0., 0.]))]
        result = summarize(ranks, coverage, folds, np.array(['drug-drug']*8), replicates=100)
        self.assertEqual(result['stage'], 'screen_rejected')
        self.assertIsNone(result['confirmation'])
        self.assertEqual(result['evaluated_folds'], 1)
        with self.assertRaises(AssertionError):
            summarize(ranks, coverage, folds*6, np.array(['drug-drug']*8), replicates=100)

    def test_confirmation_clusters_partition_and_direction_repeats(self):
        ranks = dict(baseline=np.full((3, 8), 2.), reverse_pipeline=np.ones((3, 8)))
        coverage = {a: np.ones((3, 8), np.uint8) for a in ranks}
        folds = [dict(primary=dict(delta_mrr=.5, bootstrap_95=[.5, .5]))]*6
        result = summarize(ranks, coverage, folds, np.array(['drug-drug']*8), replicates=100)
        confirmation = result['confirmation']
        self.assertEqual(confirmation['primary']['bootstrap_95'], [.5, .5])
        self.assertEqual(confirmation['primary']['delta_mrr'], .5)
        self.assertEqual(confirmation['unique_triples'], 4)
        self.assertTrue(confirmation['useful_gain_flag'])
        self.assertEqual(confirmation['mean_metrics']['reverse_pipeline']['mrr'], 1.)

    def test_file_guard_rejects_test_and_external_models(self):
        allowed = {'/tmp/rs1-fixture/dataset/split/random/valid.pt', '/tmp/rs1-fixture/own.pt'}
        out, dataset = '/tmp/rs1-fixture/output', '/tmp/rs1-fixture/dataset'
        self.assertEqual(check_path(dataset+'/split/random/valid.pt', allowed, out, dataset), 'split/random/valid.pt')
        self.assertIsNone(check_path('/tmp/rs1-fixture/own.pt', allowed, out, dataset))
        self.assertIsNone(check_path(out+'/raw.npy', allowed, out, dataset))
        for path in (dataset+'/split/random/test.pt', dataset+'/processed/geometric_data_processed.pt', '/tmp/external.pt', '/tmp/external.npy'):
            with self.assertRaises(PermissionError):
                check_path(path, allowed, out, dataset)


if __name__ == '__main__':
    unittest.main()
