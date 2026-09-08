"""TH1: immutable 95/5 pair-grouped TRAIN split, no model or evaluation."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from biokg.typed_path_pilot import DATASET, EXPECTED, buckets, digest, pair_keys, read_train, verify, write_json


SEED, CUTOFF = 36840, 95
FIELDS = ('head', 'relation', 'tail', 'head_type_id', 'tail_type_id', 'row_id')
ROLES = ('fit', 'holdout')
BINARY = ('.pt', '.pth', '.pkl', '.npy', '.npz', '.safetensors', '.bin')


def check_path(path, output, dataset=DATASET):
    path, output, dataset = Path(path).resolve(), Path(output).resolve(), Path(dataset).resolve()
    if path.is_relative_to(dataset):
        relative = str(path.relative_to(dataset))
        if relative not in EXPECTED:
            raise PermissionError(f'TH1 forbids dataset file: {relative}')
        return relative
    if path.suffix in BINARY and not path.is_relative_to(output):
        raise PermissionError(f'TH1 forbids foreign artifact: {path}')
    return None


def install_guard(output, opened):
    def guard(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes)):
            path = args[0].decode() if isinstance(args[0], bytes) else args[0]
            relative = check_path(path, output)
            if relative:
                opened.add(relative)
    sys.addaudithook(guard)


def pack(train, offset, counts):
    if set(train) != {'head', 'relation', 'tail', 'head_type', 'tail_type'}:
        raise ValueError('Expected only original TRAIN fields')
    names = sorted(counts)
    if not names or len(names) > 255 or set(offset) != set(names):
        raise ValueError('Invalid type metadata')
    cursor = 0
    for name in names:
        if counts[name] <= 0 or offset[name] != cursor:
            raise ValueError('Invalid node counts or sorted type offsets')
        cursor += counts[name]
    if cursor*cursor > np.iinfo(np.int64).max:
        raise ValueError('Pair keys would overflow')
    data = {k: np.asarray(v) for k, v in train.items()}
    n = len(data['head'])
    if not n or any(v.ndim != 1 or len(v) != n for v in data.values()):
        raise ValueError('Empty or unaligned TRAIN fields')
    out = {}
    for field in ('head', 'relation', 'tail'):
        if not np.issubdtype(data[field].dtype, np.integer) or (data[field] < 0).any():
            raise ValueError('IDs must be nonnegative integers')
        out[field] = data[field].astype(np.int64)
    for side in ('head', 'tail'):
        types = data[side+'_type']
        if not set(np.unique(types)).issubset(names):
            raise ValueError('Unknown entity type')
        type_ids = np.empty(n, np.uint8)
        for j, name in enumerate(names):
            mask = types == name
            if (out[side][mask] >= counts[name]).any():
                raise ValueError('Entity outside its typed range')
            type_ids[mask] = j
        out[side+'_type_id'] = type_ids
    out['row_id'] = np.arange(n, dtype=np.int64)
    return out


def partition(train, offset, counts):
    packed = pack(train, offset, counts)
    names = sorted(counts)
    offsets = np.array([offset[x] for x in names], np.int64)
    h = packed['head']+offsets[packed['head_type_id']]
    t = packed['tail']+offsets[packed['tail_type_id']]
    keys = pair_keys(h, t, sum(counts.values()))
    bucket = buckets(keys, seed=SEED)
    role = (bucket >= CUTOFF).astype(np.uint8)
    indices = [np.flatnonzero(role == j) for j in range(2)]
    if any(not len(ids) for ids in indices):
        raise ValueError('Both parts must be nonempty; do not retry with another seed')
    parts = {name: {k: v[ids] for k, v in packed.items()} for name, ids in zip(ROLES, indices)}
    pair_sets = [np.unique(keys[ids]) for ids in indices]
    assert not np.intersect1d(*pair_sets).size
    np.testing.assert_array_equal(np.sort(np.concatenate(indices)), np.arange(len(keys)))
    nodes = [np.unique(np.r_[h[ids], t[ids]]) for ids in indices]
    fit_node_present = np.zeros(sum(counts.values()), bool)
    fit_node_present[nodes[0]] = True
    held = indices[1]
    cold = ~fit_node_present[h[held]] | ~fit_node_present[t[held]]
    summary = dict(total_rows=len(keys), total_pairs=sum(len(s) for s in pair_sets),
        pair_disjoint=True, original_rows_preserved=True,
        roles={name: dict(rows=len(ids), pairs=len(pair_sets[j]), entities=len(nodes[j]),
            row_fraction=len(ids)/len(keys), pair_fraction=len(pair_sets[j])/sum(len(s) for s in pair_sets),
            self_link_rows=int((h[ids] == t[ids]).sum())) for j, (name, ids) in enumerate(zip(ROLES, indices))},
        holdout_entities_absent_from_fit=int(np.setdiff1d(nodes[1], nodes[0]).size),
        holdout_rows_with_unseen_endpoint=int(cold.sum()),
        relations={str(int(r)): {name: int((parts[name]['relation'] == r).sum()) for name in ROLES}
                   for r in np.unique(packed['relation'])},
        type_endpoints={name: {role_name: {side: int((parts[role_name][side+'_type_id'] == j).sum())
                        for side in ('head', 'tail')} for role_name in ROLES} for j, name in enumerate(names)})
    return parts, dict(pair_key=keys, bucket=bucket, role=role), summary


def load_fit(directory):
    """Future training entry point: only fit bytes and metadata, never holdout."""
    out = Path(directory)
    manifest = json.loads((out/'manifest.json').read_text())
    if manifest['protocol'] != 'TH1' or manifest['seed'] != SEED or manifest['cutoff'] != CUTOFF:
        raise ValueError('Unexpected TRAIN reservation protocol')
    verify({str(out/'fit_train.npz'): manifest['artifact_sha256']['fit_train.npz']})
    with np.load(out/'fit_train.npz', allow_pickle=False) as data:
        if set(data.files) != set(FIELDS):
            raise ValueError('Unexpected fit artifact fields')
        fit = {k: data[k] for k in FIELDS}
    if len(fit['head']) != manifest['summary']['roles']['fit']['rows']:
        raise ValueError('Fit row count mismatch')
    names = np.asarray(manifest['entity_types'])
    train = {k: fit[k] for k in ('head', 'relation', 'tail')}
    train.update({side+'_type': names[fit[side+'_type_id']] for side in ('head', 'tail')})
    # Validate only the returned fit records, without accessing full TRAIN.
    pack(train, manifest['offsets'], manifest['counts'])
    return train, manifest['offsets'], manifest['counts']


def receipt(repo):
    hashes = {str(DATASET/name): value for name, value in EXPECTED.items()}
    for name in ('TRAIN_HOLDOUT.md', 'train_holdout.py', 'test_train_holdout.py', 'typed_path_pilot.py'):
        path = repo/'biokg'/name
        hashes[str(path)] = digest(path)
    return dict(protocol='TH1', seed=SEED, cutoff=CUTOFF, hashes=hashes,
        roles=list(ROLES), grouping='unordered_global_entity_pair_all_relations',
        official_train_modified=False, checkpoint_loaded=False, validation_loaded=False,
        test_loaded=False, model_updates=0, lr=0., numpy=np.__version__)


def build(directory):
    out = Path(directory).resolve()
    out.mkdir(parents=True, exist_ok=False)
    before = receipt(Path(__file__).resolve().parents[1])
    opened = set()
    install_guard(out, opened)
    started = time.monotonic()
    verify(before['hashes'])
    write_json(out/'prerun.json', before)
    print(json.dumps(dict(stage='partitioning_train', seed=SEED, reserved_pair_percent=5, lr=0.)), flush=True)
    train, offsets, counts = read_train()
    parts, assignment, summary = partition(train, offsets, counts)
    for name in ROLES:
        np.savez_compressed(out/(name+'_train.npz'), **parts[name])
        print(json.dumps(dict(stage='part_saved', part=name, **summary['roles'][name], lr=0.)), flush=True)
    np.savez_compressed(out/'assignment.npz', **assignment)
    artifacts = ('fit_train.npz', 'holdout_train.npz', 'assignment.npz')
    manifest = dict(protocol='TH1', seed=SEED, cutoff=CUTOFF, entity_types=sorted(counts), offsets=offsets, counts=counts,
        fields=list(FIELDS), artifact_sha256={n: digest(out/n) for n in artifacts}, summary=summary,
        old_full_train_checkpoints_ineligible=True, model_updates=0, holdout_evaluated=False)
    write_json(out/'manifest.json', manifest)
    verify(before['hashes'])
    assert sorted(opened) == sorted(EXPECTED)
    write_json(out/'audit.json', dict(passed=True, dataset_files_opened=sorted(opened),
        artifact_sha256={n: digest(out/n) for n in (*artifacts, 'manifest.json', 'prerun.json')},
        source_input_hashes_unchanged=True, seconds=time.monotonic()-started, model_updates=0, lr=0.))
    print(json.dumps(dict(stage='reservation_complete', summary=summary, seconds=time.monotonic()-started, lr=0.)), flush=True)


def audit(directory):
    out = Path(directory).resolve()
    before = json.loads((out/'prerun.json').read_text())
    opened = set()
    install_guard(out, opened)
    saved = json.loads((out/'audit.json').read_text())
    manifest = json.loads((out/'manifest.json').read_text())
    assert before['protocol'] == manifest['protocol'] == 'TH1'
    assert before['seed'] == manifest['seed'] == SEED and before['cutoff'] == manifest['cutoff'] == CUTOFF
    assert saved['passed'] and saved['source_input_hashes_unchanged']
    verify(before['hashes'])
    verify({str(out/n): h for n, h in saved['artifact_sha256'].items()})
    verify({str(out/n): h for n, h in manifest['artifact_sha256'].items()})
    train, offsets, counts = read_train()
    parts, assignment, summary = partition(train, offsets, counts)
    assert summary == manifest['summary'] and offsets == manifest['offsets'] and counts == manifest['counts']
    assert manifest['fields'] == list(FIELDS) and manifest['entity_types'] == sorted(counts)
    for name, expected in [(n+'_train.npz', parts[n]) for n in ROLES]+[('assignment.npz', assignment)]:
        with np.load(out/name, allow_pickle=False) as data:
            assert set(data.files) == set(expected)
            for key, value in expected.items():
                assert data[key].dtype == value.dtype
                np.testing.assert_array_equal(data[key], value)
    fit, fit_offsets, fit_counts = load_fit(out)
    assert fit_offsets == offsets and fit_counts == counts
    ids = parts['fit']['row_id']
    for key in train:
        np.testing.assert_array_equal(fit[key], np.asarray(train[key])[ids])
    verify(before['hashes'])
    assert sorted(opened) == saved['dataset_files_opened'] == sorted(EXPECTED)
    result = dict(audit_passed=True, original_rows_and_pair_assignment_reproduced=True,
        pair_disjoint=True, compact_artifacts_and_fit_loader_reproduced=True,
        all_hashes_verified=True, validation_loaded=False, test_loaded=False,
        checkpoint_loaded=False, holdout_evaluated=False, model_updates=0, lr=0.)
    write_json(out/'endpoint_audit.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='biokg/results/train_holdout/s0')
    parser.add_argument('--audit', action='store_true')
    args = parser.parse_args()
    (audit if args.audit else build)(args.out)
