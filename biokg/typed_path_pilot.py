"""TP1: three typed paths, construction/fit/report TRAIN only, no model."""

import argparse
import csv
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy
from scipy import sparse
import torch


DATASET = Path('/mnt/geocore/geocore/data_ogb/ogbl_biokg')
EXPECTED = {
    'split/random/train.pt': '958ad61478efffda0db28cf05e49c14cba6880e0f2d50d6023aca99b354d0385',
    'raw/num-node-dict.csv.gz': 'de2fb3e32b2f3328cd82adaab40a18983954c0c356e4099ea6ca9a3fc3017375',
}
PATH_TYPES = ('protein', 'sideeffect', 'disease')
WEIGHTS = np.asarray([[1/3, 1/3, 1/3], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                      [.5, .5, 0], [.5, 0, .5], [0, .5, .5]], np.float32)
CAP, NEGATIVES, MIN_FIT, BOOTSTRAPS = 256, 500, 128, 2000


def digest(path):
    result = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def verify(hashes):
    for path, expected in hashes.items():
        if digest(path) != expected:
            raise ValueError(f'Hash changed: {path}')


def check_path(path, output, dataset=DATASET):
    path, output, dataset = Path(path).resolve(), Path(output).resolve(), Path(dataset).resolve()
    if path.is_relative_to(dataset):
        relative = str(path.relative_to(dataset))
        if relative not in EXPECTED:
            raise PermissionError(f'TP1 forbids dataset file: {relative}')
        return relative
    if path.suffix in ('.pt', '.pth', '.pkl', '.npy', '.npz') and not path.is_relative_to(output):
        raise PermissionError(f'TP1 forbids foreign binary artifact: {path}')
    return None


def install_guard(output, opened):
    def guard(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes)):
            value = args[0].decode() if isinstance(args[0], bytes) else args[0]
            relative = check_path(value, output)
            if relative:
                opened.add(relative)
    sys.addaudithook(guard)


def read_train(dataset=DATASET):
    with gzip.open(Path(dataset)/'raw/num-node-dict.csv.gz', 'rt') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    counts = {name: int(value) for name, value in rows[0].items()}
    offset, total = {}, 0
    for name in sorted(counts):
        offset[name], total = total, total+counts[name]
    train = torch.load(Path(dataset)/'split/random/train.pt', map_location='cpu', weights_only=False)
    assert set(train) == {'head', 'relation', 'tail', 'head_type', 'tail_type'}
    return train, offset, counts


def pair_keys(h, t, n_entities):
    return np.minimum(h, t).astype(np.int64)*n_entities+np.maximum(h, t)


def buckets(keys, seed=3650):
    # SplitMix64 overflow is deliberate, fixed unsigned integer arithmetic.
    x = np.asarray(keys, np.uint64)+np.uint64(seed)+np.uint64(0x9e3779b97f4a7c15)
    x = (x ^ (x >> np.uint64(30)))*np.uint64(0xbf58476d1ce4e5b9)
    x = (x ^ (x >> np.uint64(27)))*np.uint64(0x94d049bb133111eb)
    return ((x ^ (x >> np.uint64(31))) % np.uint64(100)).astype(np.uint8)


def binary_graph(a, b, shape):
    graph = sparse.csr_matrix((np.ones(len(a), np.float64), (a, b)), shape=shape)
    graph.sum_duplicates()
    graph.data[:] = 1.
    graph.sort_indices()
    return graph


def prepare(train, offset, counts, cap=CAP):
    ht, tt = np.asarray(train['head_type']), np.asarray(train['tail_type'])
    for side, types in (('head', ht), ('tail', tt)):
        local = np.asarray(train[side])
        for name in np.unique(types):
            values = local[types == name]
            assert (values >= 0).all() and (values < counts[str(name)]).all()
    h = np.asarray(train['head'], np.int64)+np.asarray([offset[x] for x in ht])
    t = np.asarray(train['tail'], np.int64)+np.asarray([offset[x] for x in tt])
    r = np.asarray(train['relation'], np.int64)
    signatures = {}
    for rel in np.unique(r):
        mask = r == rel
        signatures[int(rel)] = (str(ht[mask][0]), str(tt[mask][0]))
        assert np.all(ht[mask] == signatures[int(rel)][0])
        assert np.all(tt[mask] == signatures[int(rel)][1])
    triples = np.unique(np.stack([h, r, t], 1), axis=0)
    h, r, t = triples.T
    n_entities, drug_off, n_drug = sum(counts.values()), offset['drug'], counts['drug']
    keys = pair_keys(h, t, n_entities)
    bucket = buckets(keys)
    construction = bucket < 90
    construction_keys = np.unique(keys[construction])
    fit_keys = np.unique(keys[(bucket >= 90) & (bucket < 95)])
    report_keys = np.unique(keys[bucket >= 95])
    assert not np.intersect1d(construction_keys, fit_keys).size
    assert not np.intersect1d(construction_keys, report_keys).size
    assert not np.intersect1d(fit_keys, report_keys).size
    paths, path_records = [], []
    for mediator in PATH_TYPES:
        relations = [rel for rel, sig in signatures.items()
                     if sig in (('drug', mediator), (mediator, 'drug'))]
        if len(relations) != 1:
            raise ValueError(f'Expected one drug/{mediator} relation, got {relations}')
        rel = relations[0]
        mask = construction & (r == rel)
        a, b = (h[mask], t[mask]) if signatures[rel][0] == 'drug' else (t[mask], h[mask])
        graph = binary_graph(a-drug_off, b-offset[mediator], (n_drug, counts[mediator]))
        paths.append(graph)
        path_records.append(dict(mediator=mediator, relation=rel, signature=list(signatures[rel]),
                                 construction_edges=int(graph.nnz)))
    query_parts, direct = [], {}
    for rel, sig in sorted(signatures.items()):
        if sig != ('drug', 'drug'):
            continue
        mask = construction & (r == rel)
        direct[rel] = binary_graph(h[mask]-drug_off, t[mask]-drug_off, (n_drug, n_drug))
        for role in (0, 1):
            mask = (r == rel) & (h != t) & ((bucket >= 90) & (bucket < 95) if role == 0 else bucket >= 95)
            ids = np.flatnonzero(mask)
            _, first = np.unique(keys[ids], return_index=True)
            ids = ids[first]
            rng = np.random.default_rng(np.random.SeedSequence([3651, rel, role]))
            if len(ids) > cap:
                ids = np.sort(rng.choice(ids, cap, replace=False))
            query_parts.extend((int(h[i]-drug_off), rel, int(t[i]-drug_off), int(keys[i]), role) for i in ids)
    queries = np.asarray(query_parts, np.int64).reshape(-1, 5)
    if not len(queries) or not all(np.any(queries[:, 4] == role) for role in (0, 1)):
        raise ValueError('Need both held-out TRAIN roles')
    graph_audit = dict(original_triples=len(train['head']), unique_triples=len(triples),
        duplicate_triples_removed=len(train['head'])-len(triples),
        construction_triples=int(construction.sum()), construction_pairs=len(construction_keys),
        fit_pairs=len(fit_keys), report_pairs=len(report_keys), grouped_pair_disjoint=True,
        paths=path_records, selected_fit_triples=int((queries[:, 4] == 0).sum()),
        selected_report_triples=int((queries[:, 4] == 1).sum()),
        query_columns=['head_local_drug', 'relation', 'tail_local_drug', 'global_pair_key', 'role'],
        roles={'fit': 0, 'report': 1}, n_drug=n_drug, offset=offset, counts=counts)
    return queries, paths, direct, graph_audit


def candidates_for(queries, direct, n_drug, negatives=NEGATIVES):
    n = len(queries)
    result = np.empty((2*n, negatives+1), np.int64)
    sources = np.concatenate([queries[:, 0], queries[:, 2]])
    relations = np.tile(queries[:, 1], 2)
    prior = np.empty_like(result, dtype=np.float32)
    for rel in np.unique(queries[:, 1]):
        for direction in (0, 1):
            graph = direct[int(rel)] if direction == 0 else direct[int(rel)].T.tocsr()
            degree = np.asarray(graph.sum(0)).ravel().astype(np.float32)
            for i in np.flatnonzero(queries[:, 1] == rel):
                h, r, t = queries[i, :3]
                source, positive = (h, t) if direction == 0 else (t, h)
                known = graph.indices[graph.indptr[source]:graph.indptr[source+1]]
                assert positive not in known and source != positive
                allowed = np.ones(n_drug, bool)
                allowed[known] = False
                allowed[[source, positive]] = False
                rng = np.random.default_rng(np.random.SeedSequence([3653, int(h), int(r), int(t), direction]))
                choices = np.flatnonzero(allowed)
                if len(choices) < negatives:
                    raise ValueError('Insufficient distinct unknown drug candidates')
                row = direction*n+i
                result[row, 0] = positive
                result[row, 1:] = rng.choice(choices, negatives, replace=False)
                prior[row] = degree[result[row]]
    return sources, relations, result, prior


def raw_paths(paths, sources, candidates, log=lambda **x: None, chunk=128):
    result = np.empty((len(paths), *candidates.shape), np.float32)
    last = time.monotonic()
    for j, graph in enumerate(paths):
        degree = np.asarray(graph.sum(0)).ravel()
        inverse = np.divide(1., degree, out=np.zeros_like(degree), where=degree > 0)
        right = graph.multiply(inverse).T.tocsr()
        for start in range(0, len(sources), chunk):
            sl = slice(start, start+chunk)
            dense = (graph[sources[sl]] @ right).toarray()
            values = dense[np.arange(len(dense))[:, None], candidates[sl]]
            values[candidates[sl] == sources[sl, None]] = 0
            result[j, sl] = values.astype(np.float32)
            if time.monotonic()-last >= 35:
                log(stage='paths', path=j, queries=min(start+chunk, len(sources)), total=len(sources))
                last = time.monotonic()
        log(stage='path_complete', path=j, queries=len(sources))
    assert np.isfinite(result).all() and (result >= 0).all()
    return result


def direct_path(graph, source, candidates):
    degree = np.asarray(graph.sum(0)).ravel()
    neighbors = set(graph.indices[graph.indptr[source]:graph.indptr[source+1]].tolist())
    result = []
    for candidate in candidates:
        other = graph.indices[graph.indptr[candidate]:graph.indptr[candidate+1]]
        result.append(0. if candidate == source else sum(1./degree[z] for z in other if z in neighbors))
    return np.asarray(result, np.float32)


def normalize(values):
    values = np.asarray(values, np.float32)
    return (values-values.mean(-1, keepdims=True))/(values.std(-1, keepdims=True)+np.float32(1e-6))


def ranks(scores):
    assert np.isfinite(scores).all()
    return 1.+((scores[:, 1:] > scores[:, :1]).sum(1)+(scores[:, 1:] >= scores[:, :1]).sum(1))/2.


def mixture(features, weights):
    result = np.zeros(features.shape[1:], np.float32)
    for feature, weight in zip(features, weights):
        if weight:
            result += feature*np.float32(weight)
    return result


def select_recipe(features, fit, relation, direction, min_fit=MIN_FIT):
    ids = np.flatnonzero(fit)
    if not len(ids):
        raise ValueError('Empty fit role')
    rr = np.stack([1/ranks(mixture(features[:, ids], weights)) for weights in WEIGHTS])
    def choose(mask):
        values = rr[:, mask].mean(1)
        index = int(np.argmax(values))
        return dict(index=index, weights=WEIGHTS[index].tolist(), fit_mrr=float(values[index]),
                    fit_queries=int(mask.sum()))
    global_choice = choose(np.ones(len(ids), bool))
    groups = {}
    for rel in np.unique(relation):
        for d in (0, 1):
            mask = (relation[ids] == rel) & (direction[ids] == d)
            local = mask.sum() >= min_fit
            choice = choose(mask) if local else global_choice
            groups[f'{rel}/{d}'] = dict(**choice, level='relation' if local else 'global',
                                      group_fit_queries=int(mask.sum()))
    return dict(weights=WEIGHTS.tolist(), global_choice=global_choice, min_fit=min_fit, groups=groups)


def selected_ranks(features, recipe, relation, direction):
    result = np.full(features.shape[1], np.nan)
    for rel in np.unique(relation):
        for d in (0, 1):
            ids = np.flatnonzero((relation == rel) & (direction == d))
            if len(ids):
                result[ids] = ranks(mixture(features[:, ids], recipe['groups'][f'{rel}/{d}']['weights']))
    assert np.isfinite(result).all()
    return result


def report_arrays(features, raw, prior, recipe, relation, direction, report):
    f, r = features[:, report], raw[:, report]
    result = {'selected': selected_ranks(f, recipe, relation[report], direction[report]),
              'uniform': ranks(mixture(f, WEIGHTS[0])), 'pooled': ranks(np.log1p(r.sum(0))),
              'degree': ranks(prior[report])}
    result.update({name: ranks(f[i]) for i, name in enumerate(PATH_TYPES)})
    return result


def metrics(rank):
    return dict(mrr=float((1/rank).mean()), hits1=float((rank == 1).mean()),
                hits10=float((rank <= 10).mean()), queries=len(rank))


def pair_bootstrap(delta, pairs, replicates=BOOTSTRAPS):
    unique, inverse = np.unique(pairs, return_inverse=True)
    sums, counts = np.bincount(inverse, weights=delta), np.bincount(inverse)
    rng = np.random.default_rng(3655)
    sampled = []
    for _ in range(replicates):
        ids = rng.integers(len(unique), size=len(unique))
        sampled.append(float(sums[ids].sum()/counts[ids].sum()))
    return np.quantile(sampled, [.025, .975]).tolist()


def summarize(queries, raw, all_ranks, replicates=BOOTSTRAPS):
    report_triples = queries[:, 4] == 1
    n = int(report_triples.sum())
    paired_delta = (1/all_ranks['selected']-1/all_ranks['degree']).reshape(2, n).mean(0)
    pairs = queries[report_triples, 3]
    interval = pair_bootstrap(paired_delta, pairs, replicates)
    report = np.tile(report_triples, 2)
    positive_coverage = float((raw[:, report, 0] > 0).any(0).mean())
    per_path = {name: float((raw[i, report, 0] > 0).mean()) for i, name in enumerate(PATH_TYPES)}
    rel = np.tile(queries[report_triples, 1], 2)
    direction = np.repeat([0, 1], n)
    groups = []
    for relation in np.unique(rel):
        for d in (0, 1):
            mask = (rel == relation) & (direction == d)
            groups.append(dict(relation=int(relation), direction=d,
                               metrics={k: metrics(v[mask]) for k, v in all_ranks.items()}))
    delta = float(paired_delta.mean())
    return dict(protocol='TP1', scope='stratified held-out TRAIN drug-drug feasibility only',
        metrics={k: metrics(v) for k, v in all_ranks.items()}, report_triples=n,
        report_unique_pairs=len(np.unique(pairs)), fit_triples=int((queries[:, 4] == 0).sum()),
        positive_path_coverage=positive_coverage, per_path_positive_coverage=per_path,
        primary=dict(delta_mrr=delta, bootstrap_95=interval, seed=3655, replicates=replicates,
                     unit='unordered entity pair, both directions and all sampled relations', descriptive=True),
        descriptive_deltas={key: float((1/all_ranks['selected']-1/all_ranks[key]).mean())
                            for key in ('uniform', 'pooled')}, groups=groups,
        feasibility_flag=bool(positive_coverage >= .1 and delta >= .01 and interval[0] > 0),
        pipeline_complementarity_tested=False, official_mrr_evaluated=False,
        valid_loaded=False, test_loaded=False, checkpoint_loaded=False, model_updates=0,
        lr=0., gpu_used=False, submission_changed=False, automatic_next_run=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='biokg/results/typed_path_pilot/s0')
    args = parser.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    opened = set()
    install_guard(out, opened)
    started = time.monotonic()
    def log(**event):
        event.update(seconds=time.monotonic()-started, lr=0., validation_mrr_evaluated=False)
        print(json.dumps(event, allow_nan=False), flush=True)
        with (out/'progress.jsonl').open('a') as handle:
            handle.write(json.dumps(event, allow_nan=False)+'\n')
    hashes = {str(DATASET/name): value for name, value in EXPECTED.items()}
    for name in ('TYPED_PATH_PILOT.md', 'typed_path_pilot.py', 'test_typed_path_pilot.py', 'audit_typed_path_pilot.py'):
        path = Path(__file__).parent/name
        hashes[str(path)] = digest(path)
    verify(hashes)
    receipt = dict(protocol='TP1', hashes=hashes, paths=list(PATH_TYPES), weights=WEIGHTS.tolist(),
        cap=CAP, negatives=NEGATIVES, min_fit=MIN_FIT, bootstrap_replicates=BOOTSTRAPS,
        split_seed=3650, sample_seed=3651, negative_seed=3653, bootstrap_seed=3655,
        torch=torch.__version__, numpy=np.__version__, scipy=scipy.__version__, device='cpu',
        started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    write_json(out/'prerun.json', receipt)
    log(stage='inputs_verified')
    train, offset, counts = read_train()
    queries, paths, direct, graph_audit = prepare(train, offset, counts)
    del train
    np.save(out/'queries.npy', queries, allow_pickle=False)
    write_json(out/'graph_audit.json', graph_audit)
    log(stage='graph_ready', fit_triples=graph_audit['selected_fit_triples'],
        report_triples=graph_audit['selected_report_triples'], paths=graph_audit['paths'])
    sources, relation, candidates, prior = candidates_for(queries, direct, counts['drug'])
    np.save(out/'candidates.npy', candidates, allow_pickle=False)
    np.save(out/'degree_scores.npy', prior, allow_pickle=False)
    log(stage='candidates_ready', directed_queries=len(sources))
    raw = raw_paths(paths, sources, candidates, log=log)
    np.save(out/'raw_features.npy', raw, allow_pickle=False)
    normalized = np.stack([normalize(np.log1p(feature)) for feature in raw])
    np.save(out/'normalized_features.npy', normalized, allow_pickle=False)
    fit, direction = np.tile(queries[:, 4] == 0, 2), np.repeat([0, 1], len(queries))
    recipe = select_recipe(normalized, fit, relation, direction)
    write_json(out/'recipe.json', recipe)
    log(stage='fit_recipe_frozen', global_choice=recipe['global_choice'])
    result_ranks = report_arrays(normalized, raw, prior, recipe, relation, direction, ~fit)
    np.savez_compressed(out/'report_ranks.npz', **result_ranks)
    result = summarize(queries, raw, result_ranks)
    result['seconds_through_summary'] = time.monotonic()-started
    result['dataset_files_opened'] = sorted(opened)
    assert sorted(opened) == sorted(EXPECTED)
    write_json(out/'summary.json', result)
    verify(hashes)
    artifacts = ('prerun.json', 'queries.npy', 'graph_audit.json', 'candidates.npy', 'degree_scores.npy',
                 'raw_features.npy', 'normalized_features.npy', 'recipe.json', 'report_ranks.npz', 'summary.json')
    write_json(out/'audit.json', dict(source_inputs_unchanged=True, dataset_files_opened=sorted(opened),
        artifact_sha256={name: digest(out/name) for name in artifacts}))
    log(stage='complete', metrics=result['metrics'], primary=result['primary'],
        positive_path_coverage=result['positive_path_coverage'], feasibility_flag=result['feasibility_flag'])


if __name__ == '__main__':
    main()
