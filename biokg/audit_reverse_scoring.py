"""RS1 endpoint replay: direct model checks and complete evaluated-fold replay."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from biokg.candidate_retrieval import BETAS, fit_beta
from biokg.compare_feature_pipeline import metrics
from biokg.confirm_feature_blend import N_TRIPLES, SEEDS, insert_report, load_features, split_masks, verify_hashes, write_json
from biokg.mixed_operator import restore_mixed
from biokg.relation_analogy import normalize, rank_rows
from biokg.reverse_scoring import (
    SCREEN_MIN_DELTA, apply_targeted, current_scores, direct_reverse, fold_stats,
    install_guard, inverse_relation, load_valid, query_data, screen_passed, summarize,
)
from biokg.train_joint_operator import parameter_hashes


def audit(directory):
    torch.set_num_threads(4)
    out = Path(directory).resolve()
    receipt = json.loads((out/'prerun.json').read_text())
    opened = set()
    install_guard(receipt, out, opened)
    recorded = json.loads((out/'audit.json').read_text())
    summary = json.loads((out/'summary.json').read_text())
    folds = json.loads((out/'folds.json').read_text())
    scoring = json.loads((out/'scoring_audit.json').read_text())
    assert receipt['protocol'] == summary['protocol'] == 'RS1'
    assert receipt['alphas'] == list(BETAS) and receipt['partition_seeds'] == list(SEEDS)
    assert receipt['screen_min_delta'] == SCREEN_MIN_DELTA and receipt['min_fit_queries'] == 2000
    assert receipt['screen_bootstrap_seed'] == 3661 and receipt['confirmation_bootstrap_seed'] == 3671
    assert receipt['bootstrap_replicates'] == 2000
    assert receipt['device'] == 'cpu' and receipt['model_updates'] == receipt['lr'] == 0
    assert not receipt['test_loaded'] and receipt['target_family'] == 'drug-drug'
    assert recorded['model_before'] == recorded['model_after']
    assert all(recorded[k] for k in ('model_unchanged', 'gradients_absent', 'baseline_report_ranks_exact',
                                    'non_drug_drug_ranks_unchanged', 'source_input_hashes_unchanged'))
    verify_hashes(receipt['hashes'])
    verify_hashes({str(out/name): digest for name, digest in recorded['artifact_sha256'].items()})
    print(json.dumps(dict(stage='audit_hashes', passed=True, lr=0.)), flush=True)
    valid, offset, counts = load_valid()
    assert len(valid['head']) == N_TRIPLES and sum(counts.values()) == 93773
    ck = torch.load(receipt['checkpoint'], map_location='cpu', weights_only=False)
    assert ck['offset'] == offset and ck['mode'] == 'single' and ck['n_rel'] == 102
    model = restore_mixed(ck, 'cpu')
    assert model.n_params() == 27124129
    assert parameter_hashes(model) == recorded['model_before']
    data = query_data(valid, offset, ck['n_rel'])
    with np.load(out/'query_data.npz', allow_pickle=False) as saved:
        assert set(saved.files) == set(data)
        for key in data:
            np.testing.assert_array_equal(saved[key], data[key])
    raw = np.load(out/'raw_reverse.npy', mmap_mode='r', allow_pickle=False)
    normalized = np.load(out/'normalized_reverse.npy', mmap_mode='r', allow_pickle=False)
    assert raw.shape == normalized.shape == data['candidates'].shape == (62270, 501)
    assert raw.dtype == normalized.dtype == np.float32
    relations = np.unique(data['relation'])
    assert len(scoring) == len(relations) == 76
    assert [a['directed_relation'] for a in scoring] == relations.tolist()
    error = 0.
    for record in scoring:
        rel = record['directed_relation']
        ids = np.flatnonzero(data['relation'] == rel)
        sample = ids[:1]
        assert record['local_row'] == int(sample[0]) and record['queries'] == len(ids)
        assert record['global_row'] == int(data['rows'][sample[0]])
        assert record['reverse_relation'] == int(inverse_relation(rel, 102))
        assert record['permutation_exact']
        direct = direct_reverse(model, data['source'][sample], data['reverse_relation'][sample], data['candidates'][sample])
        np.testing.assert_allclose(raw[sample], direct, atol=1e-5, rtol=1e-5)
        maximum = float(np.max(np.abs(raw[sample]-direct)))
        assert maximum == record['maximum_direct_error']
        error = max(error, maximum)
    assert parameter_hashes(model) == recorded['model_after']
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    del model, ck, valid
    for start in range(0, len(raw), 2048):
        sl = slice(start, start+2048)
        assert np.isfinite(raw[sl]).all()
        np.testing.assert_array_equal(normalized[sl], normalize(raw[sl]))
    print(json.dumps(dict(stage='audit_reverse_and_normalization', direct_queries=len(scoring), maximum_error=error, passed=True, lr=0.)), flush=True)
    features, meta = load_features(receipt)
    target = data['rows']
    np.testing.assert_array_equal(target, np.flatnonzero(meta['family'] == 'drug-drug'))
    np.testing.assert_array_equal(data['relation'], meta['relation'][target]+51*meta['direction'][target])
    pair = np.load(Path(receipt['prior_cs'])/'pair_scores.npy', mmap_mode='r', allow_pickle=False)
    with np.load(Path(receipt['prior_cs'])/'ranks.npz', allow_pickle=False) as saved:
        old_ranks = saved['candidate_side']
    ranks = {a: np.full((3, 2*N_TRIPLES), np.nan) for a in ('baseline', 'reverse_pipeline')}
    coverage = {a: np.zeros((3, 2*N_TRIPLES), np.uint8) for a in ranks}
    expected_folds = []
    expected_order = [(s, f) for s in SEEDS for f in (0, 1)]
    assert len(folds) == (6 if screen_passed(folds[0]) else 1)
    assert [(f['seed'], f['fold']) for f in folds] == expected_order[:len(folds)]
    assert sorted(p.name for p in out.glob('*_reverse_alpha_recipe.json')) == sorted(
        f'seed{s}_fold{f}_reverse_alpha_recipe.json' for s, f in expected_order[:len(folds)])
    for saved_fold in folds:
        seed, fold = saved_fold['seed'], saved_fold['fold']
        fit = split_masks(N_TRIPLES, seed)[fold]
        base = current_scores(features, pair, receipt, meta, seed, fold)
        np.testing.assert_array_equal(rank_rows(base[~fit]), old_ranks[seed, ~fit])
        recipe = fit_beta(base[target], normalized, fit[target], meta['relation'][target], meta['family'][target], meta['direction'][target])
        assert recipe == json.loads((out/f'seed{seed}_fold{fold}_reverse_alpha_recipe.json').read_text())
        control, treatment = apply_targeted(base, normalized, target, recipe, meta, ~fit)
        result = fold_stats(control, treatment, fit, meta['family'], seed, fold)
        result['global_alpha'] = recipe['global_choice']['beta']
        result['directed_group_alpha_counts'] = {str(a): sum(g['beta'] == a for g in recipe['groups'].values()) for a in BETAS}
        result['reverse_alone_drug_drug_report'] = metrics(rank_rows(normalized[(~fit)[target]]))
        assert result == saved_fold
        expected_folds.append(result)
        for arm, values in (('baseline', control), ('reverse_pipeline', treatment)):
            insert_report(ranks[arm][seed], coverage[arm][seed], fit, values)
        del base
        print(json.dumps(dict(stage='audit_fold', seed=seed, fold=fold, passed=True, lr=0.,
                              baseline_mrr=result['metrics']['baseline']['mrr'], mrr=result['metrics']['reverse_pipeline']['mrr'])), flush=True)
    with np.load(out/'ranks.npz', allow_pickle=False) as saved:
        assert set(saved.files) == set(ranks) | {a+'_coverage' for a in ranks}
        for a in ranks:
            np.testing.assert_array_equal(saved[a], ranks[a])
            np.testing.assert_array_equal(saved[a+'_coverage'], coverage[a])
    for key, value in summarize(ranks, coverage, expected_folds, meta['family']).items():
        assert summary[key] == value, key
    expected_opened = ['raw/num-node-dict.csv.gz', 'split/random/train.pt', 'split/random/valid.pt']
    assert sorted(opened) == recorded['dataset_files_opened'] == summary['dataset_files_opened'] == expected_opened
    verify_hashes(receipt['hashes'])
    result = dict(audit_passed=True, source_input_artifact_hashes_verified=True,
        model_unchanged=True, gradients_absent=True, direct_reverse_queries=len(scoring),
        direct_candidates_per_query=501, sampled_raw_score_maximum_error=error,
        all_normalized_reverse_scores_exact=True, evaluated_fit_only_recipes_reproduced=len(folds),
        all_evaluated_baseline_and_treatment_ranks_exact=True, non_drug_drug_unchanged=True,
        staging_coverage_metrics_clustered_intervals_reproduced=True,
        unused_folds_not_evaluated=len(folds) == 1, train_deserialized=False, valid_deserialized=True,
        test_loaded=False, single_checkpoint=True, model_updates=0, lr=0., device='cpu')
    write_json(out/'endpoint_audit.json', result)
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    audit(parser.parse_args().directory)
