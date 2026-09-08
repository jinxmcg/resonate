"""RS1: frozen existing reverse operators added to the CS1 drug-drug score."""

import argparse
import csv
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from biokg.candidate_retrieval import (
    BETAS, DATASET, apply_beta, baseline_scores, candidates_for, fit_beta,
)
from biokg.compare_feature_pipeline import metrics
from biokg.confirm_feature_blend import (
    N_TRIPLES, SEEDS, bootstrap, insert_report, load_features, split_masks,
    summarize as summarize_partitions, verify_hashes, write_json,
)
from biokg.gradient_diagnostic import sha256
from biokg.mixed_operator import restore_mixed
from biokg.relation_analogy import normalize, rank_rows
from biokg.train_biokg_comp import globalize
from biokg.train_joint_operator import parameter_hashes


BASELINE_MRR = .8582166512116752
SCREEN_MIN_DELTA = .0005


def inverse_relation(relation, n_operators):
    relation = np.asarray(relation, np.int64)
    if n_operators <= 0 or n_operators % 2 or (relation < 0).any() or (relation >= n_operators).any():
        raise ValueError('Invalid directed relation index')
    return (relation+n_operators//2) % n_operators


def check_path(path, allowed, output, dataset=DATASET):
    path, output, dataset = Path(path).resolve(), Path(output).resolve(), Path(dataset).resolve()
    if path.is_relative_to(dataset):
        relative = str(path.relative_to(dataset))
        if relative not in ('split/random/train.pt', 'split/random/valid.pt', 'raw/num-node-dict.csv.gz'):
            raise PermissionError(f'RS1 forbids dataset file: {relative}')
    else:
        relative = None
    if path.suffix in ('.pt', '.pth', '.pkl', '.bin', '.safetensors', '.npy', '.npz'):
        if str(path) not in allowed and not path.is_relative_to(output):
            raise PermissionError(f'RS1 forbids binary artifact: {path}')
    return relative


def install_guard(receipt, out, opened):
    allowed = {str(Path(path).resolve()) for path in receipt['hashes']}
    def guard(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes)):
            path = args[0].decode() if isinstance(args[0], bytes) else args[0]
            relative = check_path(path, allowed, out)
            if relative:
                opened.add(relative)
    sys.addaudithook(guard)


def load_valid():
    with gzip.open(DATASET/'raw/num-node-dict.csv.gz', 'rt') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    counts = {name: int(value) for name, value in rows[0].items()}
    offset, total = {}, 0
    for name in sorted(counts):
        offset[name], total = total, total+counts[name]
    valid = torch.load(DATASET/'split/random/valid.pt', map_location='cpu', weights_only=False)
    return valid, offset, counts


def query_data(valid, offset, n_operators):
    valid = dict(valid, head_type=np.asarray(valid['head_type']), tail_type=np.asarray(valid['tail_type']))
    h, r, t = globalize(valid, offset)
    n = len(r)
    ids = np.flatnonzero((valid['head_type'] == 'drug') & (valid['tail_type'] == 'drug'))
    rows = np.r_[ids, n+ids]
    candidates = np.concatenate([candidates_for(valid, ids, d, offset, t if d == 0 else h) for d in (0, 1)])
    source = np.r_[h[ids], t[ids]]
    directed = np.r_[r[ids], r[ids]+n_operators//2]
    return dict(rows=rows, source=source, relation=directed,
                reverse_relation=inverse_relation(directed, n_operators), candidates=candidates)


@torch.inference_mode()
def reverse_table(model, relation, start, count, chunk=1024):
    if model.mode != 'single' or model.E.device.type != 'cpu':
        raise ValueError('RS1 requires one frozen CPU model')
    out = torch.empty((count, model.E.shape[1]), dtype=torch.complex64)
    for begin in range(0, count, chunk):
        ids = torch.arange(start+begin, start+min(begin+chunk, count))
        rel = torch.full_like(ids, int(relation))
        out[begin:begin+len(ids)] = model.a.query(ids, rel, model.a.H_b)
    return out


@torch.inference_mode()
def cached_scores(model, table, source, local_candidates, chunk=32):
    # Advanced NumPy column indexing can return Fortran-ordered arrays. Keep
    # gather/reduction layout fixed so permutations cannot change fp32 ranks.
    source = np.ascontiguousarray(source, dtype=np.int64)
    local_candidates = np.ascontiguousarray(local_candidates, dtype=np.int64)
    if (local_candidates < 0).any() or (local_candidates >= len(table)).any():
        raise ValueError('Candidate type mismatch')
    result = np.empty(local_candidates.shape, np.float32)
    for begin in range(0, len(source), chunk):
        sl = slice(begin, begin+chunk)
        queries = table[torch.as_tensor(local_candidates[sl])]
        target = model.E[torch.as_tensor(source[sl])].conj()
        values = (queries*target[:, None, :]).sum(-1).real*model.a.log_tau.exp()
        result[sl] = values.numpy()
    assert np.isfinite(result).all()
    return result


@torch.inference_mode()
def direct_reverse(model, source, reverse, candidates):
    """Reference using the unchanged model API, no transformed-entity cache."""
    rows, width = candidates.shape
    s = torch.as_tensor(candidates.reshape(-1))
    rel = torch.repeat_interleave(torch.as_tensor(reverse), width)
    target = torch.repeat_interleave(torch.as_tensor(source), width)[:, None]
    result = model.candidate_outputs(s, rel, target)[0]
    return result.reshape(rows, width).numpy()


def build_reverse(model, data, offset, counts, output, log=lambda **x: None):
    audits, processed, last = [], 0, time.monotonic()
    for directed in np.unique(data['relation']):
        ids = np.flatnonzero(data['relation'] == directed)
        reverse = int(np.unique(data['reverse_relation'][ids]).item())
        table = reverse_table(model, reverse, offset['drug'], counts['drug'])
        for start in range(0, len(ids), 256):
            selected = ids[start:start+256]
            output[selected] = cached_scores(model, table, data['source'][selected],
                                              data['candidates'][selected]-offset['drug'])
            processed += len(selected)
            if time.monotonic()-last >= 35:
                log(stage='reverse_features', queries=processed, total=len(data['rows']))
                last = time.monotonic()
        sample = ids[:1]
        actual = output[sample]
        expected = direct_reverse(model, data['source'][sample], data['reverse_relation'][sample], data['candidates'][sample])
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
        permuted = cached_scores(model, table, data['source'][sample],
                                 (data['candidates'][sample]-offset['drug'])[:, ::-1].copy(), chunk=1)
        np.testing.assert_array_equal(actual[:, ::-1], permuted)
        audits.append(dict(directed_relation=int(directed), reverse_relation=reverse,
            queries=len(ids), local_row=int(sample[0]), global_row=int(data['rows'][sample[0]]),
            maximum_direct_error=float(np.max(np.abs(actual-expected))), permutation_exact=True))
        del table
        log(stage='relation_complete', directed_relation=int(directed), queries=processed, total=len(data['rows']))
    assert processed == len(data['rows']) and np.isfinite(output).all()
    if hasattr(output, 'flush'):
        output.flush()
    return audits


def cs_scores(base, pair, recipe, relation, direction):
    """Preserve the exact nested CS1 arithmetic, including its zero branch."""
    out = np.empty_like(base)
    for rel in np.unique(relation):
        for d in (0, 1):
            ids = np.flatnonzero((relation == rel) & (direction == d))
            if not len(ids):
                continue
            beta = np.float32(recipe['groups'][f'{rel}/{d}']['beta'])
            for start in range(0, len(ids), 2048):
                rows = ids[start:start+2048]
                out[rows] = (np.float32(1)-beta)*base[rows]+beta*pair[rows]
    return out


def current_scores(features, pair, receipt, meta, seed, fold):
    bc = json.loads((Path(receipt['prior_bc'])/f'seed{seed}_fold{fold}_half_strength_recipe.json').read_text())
    cs = json.loads((Path(receipt['prior_cs'])/f'seed{seed}_fold{fold}_beta_recipe.json').read_text())
    base = baseline_scores(features['half_strength'], bc, meta['relation'], meta['direction'])
    return cs_scores(base, pair, cs, meta['relation'], meta['direction'])


def apply_targeted(base, reverse, target_rows, recipe, meta, report):
    report_ids = np.flatnonzero(report)
    control = rank_rows(base[report_ids])
    treatment = control.copy()
    mask = report[target_rows]
    positions = np.searchsorted(report_ids, target_rows[mask])
    np.testing.assert_array_equal(report_ids[positions], target_rows[mask])
    treatment[positions] = apply_beta(base[target_rows], reverse, recipe, meta['relation'][target_rows],
                                      meta['direction'][target_rows], mask)
    other = ~np.isin(report_ids, target_rows)
    np.testing.assert_array_equal(treatment[other], control[other])
    return control, treatment


def fold_stats(control, treatment, fit, family, seed, fold, replicates=2000):
    n = len(fit)//2
    np.testing.assert_array_equal(fit[:n], fit[n:])
    count = int((~fit[:n]).sum())
    delta = 1/treatment-1/control
    paired = (delta[:count]+delta[count:])*.5
    interval = bootstrap(paired, seed=3661+2*seed+fold, replicates=replicates)
    target = family[~fit] == 'drug-drug'
    direction = np.repeat([0, 1], count)
    return dict(seed=seed, fold=fold, fit_triples=int(fit[:n].sum()), report_triples=count,
        metrics=dict(baseline=metrics(control), reverse_pipeline=metrics(treatment)),
        drug_drug=dict(baseline=metrics(control[target]), reverse_pipeline=metrics(treatment[target]),
                       delta_mrr=float(delta[target].mean())),
        primary=dict(delta_mrr=float(paired.mean()), bootstrap_95=interval,
                     seed=3661+2*seed+fold, replicates=replicates, resampling_unit='original triple', descriptive=True),
        directions={str(d): {key: metrics(value[direction == d]) for key, value in
                            (('baseline', control), ('reverse_pipeline', treatment))} for d in (0, 1)},
        recovered_top1=int(((treatment == 1) & (control > 1)).sum()),
        lost_top1=int(((treatment > 1) & (control == 1)).sum()),
        recovered_from_rank2_to10=int(((treatment == 1) & (control > 1) & (control <= 10)).sum()),
        non_drug_drug_ranks_unchanged=bool(np.array_equal(control[~target], treatment[~target])))


def screen_passed(result):
    return bool(result['primary']['delta_mrr'] >= SCREEN_MIN_DELTA and result['primary']['bootstrap_95'][0] > 0)


def summarize(ranks, coverage, folds, family, replicates=2000):
    passed = screen_passed(folds[0])
    assert len(folds) == (6 if passed else 1)
    assert all(np.array_equal(np.isnan(ranks[a]), coverage[a] == 0) for a in ranks)
    confirmation = None
    if passed:
        assert all((coverage[a] == 1).all() for a in ranks)
        confirmation = summarize_partitions(dict(baseline=ranks['baseline'], half_strength=ranks['reverse_pipeline']),
                                             bootstrap_seed=3671, replicates=replicates)
        def rename(value):
            if isinstance(value, dict):
                return {('reverse_pipeline' if k == 'half_strength' else k): rename(v) for k, v in value.items()}
            if isinstance(value, list):
                return [rename(v) for v in value]
            return value
        confirmation = rename(confirmation)
        confirmation.update(protocol='RS1', dataset_splits_loaded=True)
        target = family == 'drug-drug'
        confirmation['drug_drug_mrr'] = {a: float((1/ranks[a][:, target]).mean()) for a in ranks}
    return dict(protocol='RS1', stage='confirmation_complete' if passed else 'screen_rejected',
        screen_passed=passed, evaluated_folds=len(folds), screen=folds[0], confirmation=confirmation,
        original_pipeline_aggregate_mrr=BASELINE_MRR, target_family='drug-drug',
        validation_reused=True, model_seed_robustness_tested=False, single_checkpoint=True,
        new_model_parameters=0, model_updates=0, gradients_absent=True, lr=0., device='cpu',
        train_deserialized=False, valid_deserialized=True, test_loaded=False,
        full_valid_refit=False, submission_changed=False, automatic_promotion=False)


def input_receipt(repo):
    cs = repo/'biokg/results/candidate_retrieval/s0_retry1'
    old = json.loads((cs/'prerun.json').read_text())
    audit = json.loads((cs/'audit.json').read_text())
    assert json.loads((cs/'endpoint_audit.json').read_text())['audit_passed']
    hashes = dict(old['hashes'])
    for name in ('prerun.json', 'pair_scores.npy', 'ranks.npz', 'summary.json'):
        hashes[str(cs/name)] = audit['artifact_sha256'][name]
    for seed in SEEDS:
        for fold in (0, 1):
            name = f'seed{seed}_fold{fold}_beta_recipe.json'
            hashes[str(cs/name)] = audit['artifact_sha256'][name]
    for name in ('audit.json', 'endpoint_audit.json'):
        hashes[str(cs/name)] = sha256(cs/name)
    for name in ('REVERSE_SCORING.md', 'reverse_scoring.py', 'test_reverse_scoring.py', 'audit_reverse_scoring.py'):
        path = repo/'biokg'/name
        hashes[str(path)] = sha256(path)
    return dict(protocol='RS1', hashes=hashes, prior_cs=str(cs), prior_bc=old['prior_bc'],
        prior_fc=old['prior_fc'], checkpoint=old['checkpoint'], alphas=list(BETAS),
        min_fit_queries=2000, screen_min_delta=SCREEN_MIN_DELTA, partition_seeds=list(SEEDS),
        screen_bootstrap_seed=3661, confirmation_bootstrap_seed=3671, bootstrap_replicates=2000,
        target_family='drug-drug', device='cpu', torch=torch.__version__, numpy=np.__version__,
        model_updates=0, lr=0., test_loaded=False, started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='biokg/results/reverse_scoring/s0')
    args = parser.parse_args()
    torch.set_num_threads(4)
    repo, out = Path(__file__).resolve().parents[1], Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    receipt = input_receipt(repo)
    opened = set()
    install_guard(receipt, out, opened)
    started = time.monotonic()
    def log(**event):
        event.update(seconds=time.monotonic()-started, lr=0., baseline_aggregate_mrr=BASELINE_MRR)
        print(json.dumps(event, allow_nan=False), flush=True)
        with (out/'progress.jsonl').open('a') as handle:
            handle.write(json.dumps(event, allow_nan=False)+'\n')
    log(stage='verifying_inputs')
    verify_hashes(receipt['hashes'])
    write_json(out/'prerun.json', receipt)
    features, meta = load_features(receipt)
    valid, offset, counts = load_valid()
    assert len(valid['head']) == N_TRIPLES and sum(counts.values()) == 93773
    ck = torch.load(receipt['checkpoint'], map_location='cpu', weights_only=False)
    assert ck['mode'] == 'single' and ck['offset'] == offset and ck['n_rel'] == 102
    model = restore_mixed(ck, 'cpu')
    assert model.n_params() == 27124129
    before = parameter_hashes(model)
    assert before == json.loads((Path(receipt['prior_cs'])/'audit.json').read_text())['model_after']
    data = query_data(valid, offset, ck['n_rel'])
    np.testing.assert_array_equal(data['rows'], np.flatnonzero(meta['family'] == 'drug-drug'))
    np.testing.assert_array_equal(data['relation'], meta['relation'][data['rows']]+51*meta['direction'][data['rows']])
    np.savez_compressed(out/'query_data.npz', **data)
    raw = np.lib.format.open_memmap(out/'raw_reverse.npy', mode='w+', dtype=np.float32, shape=data['candidates'].shape)
    log(stage='reverse_features_start', queries=len(data['rows']))
    scoring_audit = build_reverse(model, data, offset, counts, raw, log)
    after = parameter_hashes(model)
    assert before == after and all(not p.requires_grad and p.grad is None for p in model.parameters())
    write_json(out/'scoring_audit.json', scoring_audit)
    del model, ck, valid
    reverse = np.lib.format.open_memmap(out/'normalized_reverse.npy', mode='w+', dtype=np.float32, shape=raw.shape)
    for start in range(0, len(raw), 2048):
        reverse[start:start+2048] = normalize(raw[start:start+2048])
    reverse.flush()
    pair = np.load(Path(receipt['prior_cs'])/'pair_scores.npy', mmap_mode='r', allow_pickle=False)
    with np.load(Path(receipt['prior_cs'])/'ranks.npz', allow_pickle=False) as saved:
        old_ranks = saved['candidate_side']
    ranks = {a: np.full((3, 2*N_TRIPLES), np.nan) for a in ('baseline', 'reverse_pipeline')}
    coverage = {a: np.zeros((3, 2*N_TRIPLES), np.uint8) for a in ranks}
    target_rows = data['rows']
    folds, artifacts = [], ['prerun.json', 'query_data.npz', 'raw_reverse.npy', 'normalized_reverse.npy', 'scoring_audit.json']
    for seed, fold in [(s, f) for s in SEEDS for f in (0, 1)]:
        fit = split_masks(N_TRIPLES, seed)[fold]
        log(stage='fold_start', seed=seed, fold=fold)
        base = current_scores(features, pair, receipt, meta, seed, fold)
        np.testing.assert_array_equal(rank_rows(base[~fit]), old_ranks[seed, ~fit])
        recipe = fit_beta(base[target_rows], reverse, fit[target_rows], meta['relation'][target_rows],
                          meta['family'][target_rows], meta['direction'][target_rows])
        name = f'seed{seed}_fold{fold}_reverse_alpha_recipe.json'
        write_json(out/name, recipe)
        artifacts.append(name)
        control, treatment = apply_targeted(base, reverse, target_rows, recipe, meta, ~fit)
        result = fold_stats(control, treatment, fit, meta['family'], seed, fold)
        result['global_alpha'] = recipe['global_choice']['beta']
        result['directed_group_alpha_counts'] = {str(a): sum(g['beta'] == a for g in recipe['groups'].values()) for a in BETAS}
        result['reverse_alone_drug_drug_report'] = metrics(rank_rows(reverse[(~fit)[target_rows]]))
        folds.append(result)
        write_json(out/'folds.json', folds)
        for arm, values in (('baseline', control), ('reverse_pipeline', treatment)):
            insert_report(ranks[arm][seed], coverage[arm][seed], fit, values)
        log(stage='fold_complete', **result)
        del base
        if seed == 0 and fold == 0 and not screen_passed(result):
            log(stage='fail_fast_screen_stop', primary=result['primary'])
            break
    np.savez_compressed(out/'ranks.npz', **ranks, **{a+'_coverage': v for a, v in coverage.items()})
    summary = summarize(ranks, coverage, folds, meta['family'])
    summary.update(seconds_through_summary=time.monotonic()-started, dataset_files_opened=sorted(opened))
    write_json(out/'summary.json', summary)
    artifacts += ['folds.json', 'ranks.npz', 'summary.json']
    verify_hashes(receipt['hashes'])
    write_json(out/'audit.json', dict(source_input_hashes_unchanged=True, model_before=before, model_after=after,
        model_unchanged=True, gradients_absent=True, baseline_report_ranks_exact=True,
        non_drug_drug_ranks_unchanged=True, dataset_files_opened=sorted(opened),
        artifact_sha256={name: sha256(out/name) for name in artifacts}))
    log(stage='complete', decision=summary['stage'], evaluated_folds=len(folds))


if __name__ == '__main__':
    main()
