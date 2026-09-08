"""TH1 synthetic tests, no real dataset or checkpoint access."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from biokg.train_holdout import CUTOFF, DATASET, FIELDS, SEED, check_path, load_fit, pack, partition
from biokg.typed_path_pilot import buckets, digest, pair_keys, write_json


def fixture():
    counts, offsets = dict(drug=80, protein=80), dict(drug=0, protein=80)
    rows = []
    for a in range(30):
        for b in range(a+1, 40):
            rows.extend([(a, 0, b, 'drug', 'drug'), (b, 0, a, 'drug', 'drug'),
                         (a, 0, b, 'drug', 'drug'), (a, 1, b, 'drug', 'drug'),
                         (a, 2, b, 'drug', 'protein')])
    rows.extend([(0, 0, 0, 'drug', 'drug'), (0, 1, 0, 'drug', 'drug')])
    train = {name: np.asarray([r[j] for r in rows], dtype=np.int64 if j < 3 else str)
             for j, name in enumerate(('head', 'relation', 'tail', 'head_type', 'tail_type'))}
    return train, offsets, counts


class TrainHoldoutTests(unittest.TestCase):
    def test_hash_and_typed_keys_match_fixed_reference(self):
        # Scalar Python arithmetic, independent of the vectorized helper.
        def reference(key):
            mask = (1 << 64)-1
            x = (key+36840+0x9e3779b97f4a7c15) & mask
            x = ((x ^ (x >> 30))*0xbf58476d1ce4e5b9) & mask
            x = ((x ^ (x >> 27))*0x94d049bb133111eb) & mask
            return (x ^ (x >> 31)) % 100
        keys = np.array([0, 1, 123, 999999, 93773**2-1], np.int64)
        np.testing.assert_array_equal(buckets(keys, seed=SEED), [reference(int(k)) for k in keys])
        self.assertEqual(SEED, 36840)
        self.assertEqual(CUTOFF, 95)
        self.assertNotEqual(pair_keys(np.array([0]), np.array([1]), 160)[0],
                            pair_keys(np.array([0]), np.array([81]), 160)[0])

    def test_reverse_duplicate_cross_relation_and_self_grouping(self):
        train, offsets, counts = fixture()
        parts, assignment, summary = partition(train, offsets, counts)
        role = assignment['role']
        for start in range(0, len(role)-2, 5):
            np.testing.assert_array_equal(role[start:start+4], [role[start]]*4)
            np.testing.assert_array_equal(assignment['pair_key'][start:start+4], [assignment['pair_key'][start]]*4)
            self.assertNotEqual(assignment['pair_key'][start], assignment['pair_key'][start+4])
        self.assertEqual(role[-1], role[-2])
        self.assertTrue(summary['pair_disjoint'])
        self.assertEqual(summary['total_rows'], len(train['head']))
        a, b = [assignment['pair_key'][parts[n]['row_id']] for n in ('fit', 'holdout')]
        self.assertFalse(np.intersect1d(a, b).size)

    def test_every_original_row_preserved_once_with_correct_values(self):
        train, offsets, counts = fixture()
        parts, assignment, summary = partition(train, offsets, counts)
        ids = np.concatenate([parts[n]['row_id'] for n in ('fit', 'holdout')])
        np.testing.assert_array_equal(np.sort(ids), np.arange(len(train['head'])))
        for j, name in enumerate(('fit', 'holdout')):
            self.assertEqual(set(parts[name]), set(FIELDS))
            for key in ('head', 'relation', 'tail'):
                np.testing.assert_array_equal(parts[name][key], train[key][parts[name]['row_id']])
            self.assertTrue((assignment['role'][parts[name]['row_id']] == j).all())
        self.assertEqual(sum(v['rows'] for v in summary['roles'].values()), len(ids))

    def test_order_invariance_and_no_input_mutation(self):
        train, offsets, counts = fixture()
        before = {k: v.copy() for k, v in train.items()}
        first = partition(train, offsets, counts)
        perm = np.random.default_rng(9).permutation(len(train['head']))
        other = partition({k: v[perm] for k, v in train.items()}, offsets, counts)
        for key in first[1]:
            np.testing.assert_array_equal(other[1][key], first[1][key][perm])
        self.assertEqual(first[2], other[2])
        for key in train:
            np.testing.assert_array_equal(train[key], before[key])

    def test_group_assignment_does_not_depend_on_relation_label(self):
        train, offsets, counts = fixture()
        _, assignment, _ = partition(train, offsets, counts)
        modified = {k: v.copy() for k, v in train.items()}
        modified['relation'][assignment['role'] == 1] = 98
        _, other, _ = partition(modified, offsets, counts)
        for key in assignment:
            np.testing.assert_array_equal(assignment[key], other[key])

    def test_invalid_fields_types_bounds_and_offsets_rejected(self):
        train, offsets, counts = fixture()
        for field, value in (('head', -1), ('tail', 80), ('relation', -1), ('head_type', 'x')):
            bad = {k: v.copy() for k, v in train.items()}
            bad[field][0] = value
            with self.assertRaises(ValueError):
                pack(bad, offsets, counts)
        with self.assertRaises(ValueError):
            pack({**train, 'relation': train['relation'].astype(float)}, offsets, counts)
        with self.assertRaises(ValueError):
            pack({**train, 'head_neg': np.zeros(1)}, offsets, counts)
        with self.assertRaises(ValueError):
            pack({**train, 'tail': train['tail'][:-1]}, offsets, counts)
        with self.assertRaises(ValueError):
            pack(train, dict(drug=0, protein=79), counts)
        with self.assertRaises(ValueError):
            pack({k: v[:0] for k, v in train.items()}, offsets, counts)

    def test_fit_loader_reads_no_holdout_or_full_train(self):
        train, offsets, counts = fixture()
        parts, _, summary = partition(train, offsets, counts)
        with tempfile.TemporaryDirectory(prefix='th1-fixture-') as tmp:
            out = Path(tmp)
            np.savez_compressed(out/'fit_train.npz', **parts['fit'])
            write_json(out/'manifest.json', dict(protocol='TH1', seed=SEED, cutoff=CUTOFF,
                entity_types=sorted(counts), offsets=offsets, counts=counts, summary=summary,
                artifact_sha256={'fit_train.npz': digest(out/'fit_train.npz')}))
            # No holdout file exists. Spy verifies the only NumPy artifact read.
            with patch('biokg.train_holdout.np.load', wraps=np.load) as loader:
                fit, actual_offsets, actual_counts = load_fit(out)
            self.assertEqual(loader.call_count, 1)
            self.assertEqual(Path(loader.call_args.args[0]), out/'fit_train.npz')
            self.assertFalse(loader.call_args.kwargs['allow_pickle'])
            self.assertEqual(actual_offsets, offsets)
            self.assertEqual(actual_counts, counts)
            for key in train:
                np.testing.assert_array_equal(fit[key], train[key][parts['fit']['row_id']])
            # A mismatching receipt must fail before deserialization.
            manifest = json.loads((out/'manifest.json').read_text())
            manifest['artifact_sha256']['fit_train.npz'] = '0'*64
            write_json(out/'manifest.json', manifest)
            with self.assertRaises(ValueError), patch('biokg.train_holdout.np.load') as loader:
                load_fit(out)
            loader.assert_not_called()

    def test_guard_rejects_validation_test_checkpoints_and_foreign_caches(self):
        out = Path('/tmp/th1-guard')
        self.assertEqual(check_path(DATASET/'split/random/train.pt', out), 'split/random/train.pt')
        self.assertEqual(check_path(DATASET/'raw/num-node-dict.csv.gz', out), 'raw/num-node-dict.csv.gz')
        self.assertIsNone(check_path(out/'fit_train.npz', out))
        for path in (DATASET/'split/random/valid.pt', DATASET/'split/random/test.pt',
                     DATASET/'processed/data_processed', DATASET/'raw/full_graph.csv.gz',
                     Path('/tmp/full-train-model.pt'), Path('/tmp/old-features.npz'),
                     Path('/tmp/weights.safetensors'), out/'..'/'foreign.npy'):
            with self.assertRaises(PermissionError):
                check_path(path, out)


if __name__ == '__main__':
    unittest.main()
