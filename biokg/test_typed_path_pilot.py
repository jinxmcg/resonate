"""TP1 synthetic split-isolation, exact path and selection regression tests."""

import unittest
from pathlib import Path

import numpy as np

from biokg.typed_path_pilot import (
    DATASET, WEIGHTS, binary_graph, buckets, candidates_for, check_path, direct_path,
    mixture, normalize, pair_bootstrap, pair_keys, prepare, ranks, raw_paths,
    report_arrays, select_recipe, selected_ranks, summarize,
)


def fixture():
    counts = dict(disease=5, drug=60, protein=7, sideeffect=6)
    offset, total = {}, 0
    for name in sorted(counts):
        offset[name], total = total, total+counts[name]
    records = []
    for i in range(40):
        for rel, mediator in enumerate(('protein', 'sideeffect', 'disease')):
            for z in range(counts[mediator]):
                if (i+z) % 3 == 0:
                    records.append((z, rel, i, mediator, 'drug') if mediator == 'sideeffect'
                                   else (i, rel, z, 'drug', mediator))
        for j in range(i+1, 40):
            records.extend([(i, 3, j, 'drug', 'drug'), (j, 3, i, 'drug', 'drug'),
                            (i, 3, j, 'drug', 'drug'), (i, 4, j, 'drug', 'drug')])
    records.append((0, 3, 0, 'drug', 'drug'))
    train = {name: np.asarray([row[i] for row in records], dtype=np.int64 if i < 3 else str)
             for i, name in enumerate(('head', 'relation', 'tail', 'head_type', 'tail_type'))}
    return train, offset, counts


class TypedPathTests(unittest.TestCase):
    def test_guard_allows_only_train_counts_and_own_cache(self):
        out = Path('/tmp/tp1-guard-test')
        self.assertEqual(check_path(DATASET/'split/random/train.pt', out), 'split/random/train.pt')
        self.assertEqual(check_path(DATASET/'raw/num-node-dict.csv.gz', out), 'raw/num-node-dict.csv.gz')
        self.assertIsNone(check_path(out/'raw_features.npy', out))
        for path in (DATASET/'split/random/valid.pt', DATASET/'split/random/test.pt',
                     DATASET/'raw/relations/edges.csv.gz', DATASET/'processed/data_processed',
                     Path('/tmp/foreign.pt'), Path('/tmp/predictions.npz')):
            with self.assertRaises(PermissionError):
                check_path(path, out)

    def test_pair_grouping_reverse_and_cross_relation_stable(self):
        h, t = np.array([2, 8, 2, 9]), np.array([8, 2, 8, 3])
        keys = pair_keys(h, t, 20)
        np.testing.assert_array_equal(keys[:3], [48]*3)
        bucket = buckets(keys)
        np.testing.assert_array_equal(bucket[:3], np.repeat(bucket[0], 3))
        np.testing.assert_array_equal(buckets(keys[::-1]), bucket[::-1])
        self.assertTrue(((bucket >= 0) & (bucket < 100)).all())
        all_buckets = buckets(np.arange(10000))
        self.assertTrue((all_buckets < 90).any())
        self.assertTrue(((all_buckets >= 90) & (all_buckets < 95)).any())
        self.assertTrue((all_buckets >= 95).any())

    def test_prepare_removes_all_relations_reverse_and_duplicates(self):
        train, offset, counts = fixture()
        queries, paths, direct, audit = prepare(train, offset, counts, cap=8)
        self.assertGreater(audit['duplicate_triples_removed'], 0)
        self.assertTrue(audit['grouped_pair_disjoint'])
        self.assertEqual([p['signature'] for p in audit['paths']],
                         [['drug', 'protein'], ['sideeffect', 'drug'], ['drug', 'disease']])
        for role in (0, 1):
            ids = queries[queries[:, 4] == role]
            for rel in (3, 4):
                selected = ids[ids[:, 1] == rel]
                self.assertLessEqual(len(selected), 8)
                self.assertEqual(len(selected), len(np.unique(selected[:, 3])))
            expected = (buckets(ids[:, 3]) >= 90) & (buckets(ids[:, 3]) < 95) if role == 0 else buckets(ids[:, 3]) >= 95
            self.assertTrue(expected.all())
        heldout = set(queries[:, 3].tolist())
        for graph in direct.values():
            a, b = graph.nonzero()
            keys = pair_keys(a+offset['drug'], b+offset['drug'], sum(counts.values()))
            self.assertTrue((buckets(keys) < 90).all())
            self.assertFalse(heldout.intersection(keys.tolist()))
            self.assertTrue((graph.data == 1).all())
        for mediator, graph in zip(('protein', 'sideeffect', 'disease'), paths):
            a, b = graph.nonzero()
            keys = pair_keys(a+offset['drug'], b+offset[mediator], sum(counts.values()))
            self.assertTrue((buckets(keys) < 90).all())
        self.assertTrue((queries[:, 0] != queries[:, 2]).all())

    def test_prepare_is_order_independent_and_rejects_ambiguous_schema(self):
        train, offset, counts = fixture()
        a = prepare(train, offset, counts, cap=8)
        perm = np.random.default_rng(4).permutation(len(train['head']))
        b = prepare({k: v[perm] for k, v in train.items()}, offset, counts, cap=8)
        np.testing.assert_array_equal(a[0], b[0])
        self.assertEqual(a[3], b[3])
        for x, y in zip(a[1], b[1]):
            np.testing.assert_array_equal(x.toarray(), y.toarray())
        bad = {k: v.copy() for k, v in train.items()}
        ids = np.flatnonzero(bad['relation'] == 0)
        bad['relation'][ids[:2]] = 9
        with self.assertRaises(ValueError):
            prepare(bad, offset, counts)

    def test_candidate_types_direction_filtering_and_row_rng_independence(self):
        graph = binary_graph([0, 0, 1, 4], [1, 2, 2, 3], (7, 7))
        queries = np.array([[0, 8, 3, 3, 0], [1, 8, 4, 11, 1]])
        sources, relation, candidates, prior = candidates_for(queries, {8: graph}, 7, negatives=2)
        np.testing.assert_array_equal(sources, [0, 1, 3, 4])
        np.testing.assert_array_equal(relation, [8]*4)
        np.testing.assert_array_equal(candidates[:, 0], [3, 4, 0, 1])
        for row in range(4):
            local = graph if row < 2 else graph.T.tocsr()
            known = set(local.indices[local.indptr[sources[row]]:local.indptr[sources[row]+1]])
            self.assertFalse(known.intersection(candidates[row, 1:].tolist()))
            self.assertNotIn(sources[row], candidates[row])
            self.assertEqual(len(np.unique(candidates[row])), 3)
            np.testing.assert_array_equal(prior[row], np.asarray(local.sum(0)).ravel()[candidates[row]])
        other = queries.copy()
        other[1, [0, 2]] = [2, 6]
        changed = candidates_for(other, {8: graph}, 7, negatives=2)[2]
        np.testing.assert_array_equal(changed[[0, 2]], candidates[[0, 2]])
        with self.assertRaises(ValueError):
            candidates_for(queries, {8: graph}, 7, negatives=6)

    def test_exact_paths_weight_hubs_and_exclude_self(self):
        graph = binary_graph([0, 0, 1, 1, 2, 2, 3], [0, 1, 0, 1, 0, 2, 0], (5, 4))
        candidates = np.arange(5)[None]
        got = raw_paths([graph], np.array([0]), candidates)[0, 0]
        np.testing.assert_array_equal(got, [0, .75, .25, .25, 0])
        np.testing.assert_array_equal(got, direct_path(graph, 0, candidates[0]))
        empty = binary_graph([], [], (5, 4))
        np.testing.assert_array_equal(raw_paths([empty], np.array([0]), candidates), np.zeros((1, 1, 5)))

    def test_raw_duplicate_permutation_extension_and_chunk_parity(self):
        rng = np.random.default_rng(3)
        a, b = np.nonzero(rng.random((18, 9)) < .3)
        graph = binary_graph(np.r_[a, a], np.r_[b, b], (18, 9))
        sources = np.arange(7)
        candidates = np.tile([2, 1, 5, 5, 0, 17], (7, 1))
        first = raw_paths([graph], sources, candidates, chunk=3)
        np.testing.assert_array_equal(first, raw_paths([graph], sources, candidates, chunk=1))
        np.testing.assert_array_equal(first[:, :, ::-1], raw_paths([graph], sources, candidates[:, ::-1]))
        extended = np.c_[candidates, np.full(7, 10)]
        np.testing.assert_array_equal(first, raw_paths([graph], sources, extended)[:, :, :-1])
        np.testing.assert_array_equal(first[:, :, 2], first[:, :, 3])
        for row, source in enumerate(sources):
            np.testing.assert_allclose(first[0, row], direct_path(graph, source, candidates[row]), atol=1e-6, rtol=1e-6)

    def test_normalization_zero_rows_and_official_ties(self):
        np.testing.assert_array_equal(normalize(np.ones((4, 7), np.float32)), np.zeros((4, 7)))
        scores = np.array([[4, 2, 4, 5], [1, 1, 1, 1], [4, 1, 2, 3]], np.float32)
        np.testing.assert_array_equal(ranks(scores), [2.5, 2.5, 1.])
        np.testing.assert_array_equal(ranks(normalize(scores)), ranks(scores))

    def test_selection_fit_report_poisoning_and_fallback(self):
        rng = np.random.default_rng(7)
        features = rng.normal(size=(3, 40, 13)).astype(np.float32)
        fit = np.tile(np.array([True]*10+[False]*10), 2)
        rel = np.tile(np.repeat([0, 1], 10), 2)
        direction = np.repeat([0, 1], 20)
        first = select_recipe(features, fit, rel, direction, min_fit=8)
        other = features.copy()
        other[:, ~fit, 0] += 10000
        self.assertEqual(first, select_recipe(other, fit, rel, direction, min_fit=8))
        self.assertEqual(first['groups']['1/0']['level'], 'global')
        self.assertEqual(first['groups']['0/0']['level'], 'relation')
        got = selected_ranks(features, first, rel, direction)
        for row in range(40):
            weights = first['groups'][f'{rel[row]}/{direction[row]}']['weights']
            self.assertEqual(got[row], ranks(mixture(features[:, row:row+1], weights))[0])
        tied = select_recipe(np.ones_like(features), fit, rel, direction, min_fit=100)
        self.assertEqual(tied['global_choice']['index'], 0)
        self.assertTrue(all(g['index'] == 0 and g['level'] == 'global' for g in tied['groups'].values()))

    def test_pair_bootstrap_weights_relation_clusters(self):
        delta, pairs = np.array([1., 1., -.5]), np.array([10, 10, 20])
        rng = np.random.default_rng(3655)
        expected = []
        for _ in range(50):
            ids = rng.integers(2, size=2)
            expected.append(np.array([2., -.5])[ids].sum()/np.array([2, 1])[ids].sum())
        np.testing.assert_array_equal(pair_bootstrap(delta, pairs, replicates=50), np.quantile(expected, [.025, .975]))

    def test_small_end_to_end_and_summary_boundaries(self):
        train, offset, counts = fixture()
        queries, paths, direct, _ = prepare(train, offset, counts, cap=8)
        source, relation, candidates, prior = candidates_for(queries, direct, counts['drug'], negatives=4)
        raw = raw_paths(paths, source, candidates, chunk=5)
        normalized = np.stack([normalize(np.log1p(f)) for f in raw])
        fit, direction = np.tile(queries[:, 4] == 0, 2), np.repeat([0, 1], len(queries))
        recipe = select_recipe(normalized, fit, relation, direction, min_fit=5)
        result = report_arrays(normalized, raw, prior, recipe, relation, direction, ~fit)
        summary = summarize(queries, raw, result, replicates=30)
        expected = float((1/result['selected']-1/result['degree']).mean())
        self.assertAlmostEqual(summary['primary']['delta_mrr'], expected)
        self.assertFalse(summary['valid_loaded'] or summary['test_loaded'] or summary['checkpoint_loaded'])
        self.assertFalse(summary['pipeline_complementarity_tested'] or summary['submission_changed'])
        self.assertEqual(summary['model_updates'], 0)
        self.assertEqual(summary['lr'], 0)
        self.assertEqual(summary['report_triples'], int((queries[:, 4] == 1).sum()))


if __name__ == '__main__':
    unittest.main()
