"""TF2 saved-record replay; direct gradient/feature checks are in the runner."""

import argparse
import json
from pathlib import Path

import numpy as np

from biokg.confirm_feature_blend import verify_hashes, write_json
from biokg.objective_forensics import COMPONENTS, GROUPS, blend, install_guard, sample_batch, select_cases, summarize
from biokg.gradient_diagnostic import train_only
from biokg.confirm_feature_blend import DATASET
from biokg.train_forensics import catalog_rank, neighbors, symmetric_graph


def audit(directory):
    out = Path(directory).resolve()
    receipt = json.loads((out/'prerun.json').read_text())
    opened = set()
    install_guard(receipt, out, opened)
    audit = json.loads((out/'audit.json').read_text())
    summary = json.loads((out/'summary.json').read_text())
    cases = json.loads((out/'cases.json').read_text())
    records = [json.loads(line) for line in (out/'gradients.jsonl').read_text().splitlines()]
    assert receipt['protocol'] == 'TF2' and receipt['model_updates'] == receipt['lr'] == 0
    assert audit['passed'] and audit['source_input_hashes_unchanged'] and audit['cpu_model_unchanged']
    assert audit['student_before'] == audit['student_after'] and audit['teacher_before'] == audit['teacher_after']
    assert len(audit['teacher_before']) == 10 and audit['all_parameter_grads_absent']
    assert audit['gradients_train_only'] and audit['optimizer_updates'] == 0
    verify_hashes(receipt['hashes'])
    verify_hashes({str(out/name): digest for name, digest in audit['artifact_sha256'].items()})
    original = json.loads((Path(receipt['tf1'])/'findings.json').read_text())
    expected_cases = select_cases(original['records'])
    assert len(cases) == len(expected_cases) and len(records) == 2*len(cases)
    train, offsets, counts = train_only(DATASET)
    offset, count = offsets['drug'], counts['drug']
    h, r, t = (np.asarray(train[k]) for k in ('head', 'relation', 'tail'))
    bc, cs = [json.loads(Path(receipt[k]).read_text()) for k in ('bc_recipe', 'cs_recipe')]
    with np.load(out/'features.npz', allow_pickle=False) as features, np.load(out/'batches.npz', allow_pickle=False) as batches:
        raw, normalized, pipeline = features['raw'], features['normalized'], features['pipeline']
        assert raw.shape == normalized.shape == (len(cases), 7, count)
        assert pipeline.shape == (len(cases), count)
        assert raw.dtype == normalized.dtype == pipeline.dtype == np.float32
        assert all(np.isfinite(a).all() for a in (raw, normalized, pipeline))
        for i, (case, base) in enumerate(zip(cases, expected_cases)):
            for key, value in base.items():
                assert case[key] == value
            rows = np.flatnonzero(r == case['relation'])
            source, positive, winner = case['source_local_id'], case['positive_local_id'], case['winner_local_id']
            known = neighbors(symmetric_graph(h[rows], t[rows], count), source)
            np.testing.assert_array_equal(known, batches[f'case{i}_known'])
            assert positive in known and winner not in known
            unknown = np.ones(count, bool)
            unknown[known] = False
            assert winner == int(np.flatnonzero(unknown)[np.argmax(raw[i, 0, unknown])])
            score, z = blend(raw[i], bc, cs, case['relation'], case['direction'])
            np.testing.assert_array_equal(score, pipeline[i])
            np.testing.assert_array_equal(z, normalized[i])
            assert catalog_rank(raw[i, 0], positive, known) == case['model_rank'] == case['filtered_rank']
            assert catalog_rank(score, positive, known) == case['pipeline_rank']
            assert float(score[positive]-score[winner]) == case['pipeline_pair_margin']
            np.testing.assert_array_equal(raw[i][:, [positive, winner]], np.asarray(case['feature_pair_values'], np.float32))
            expected_batch = sample_batch(train, rows, case, i, offset, count)
            for key, value in expected_batch.items():
                np.testing.assert_array_equal(value, batches[f'case{i}_{key}'])
            assert case['pools_identical'] == bool(np.array_equal(expected_batch['negatives_random'], expected_batch['negatives_forced']))
            for pool_index, pool in enumerate(('random', 'forced')):
                record = records[2*i+pool_index]
                assert record['case_index'] == i and record['sample_index'] == case['sample_index'] and record['pool'] == pool
                negatives = expected_batch['negatives_'+pool]-offset
                for key, value in dict(competitor_draws=int((negatives == winner).sum()), positive_draws=int((negatives == positive).sum()),
                                       known_positive_draws=int(np.isin(negatives, known).sum())).items():
                    assert record[key] == case[pool+'_pool'][key] == value
                assert record['summed_gradient_parity_passed'] and record['parameter_grads_absent']
                assert record['unscaled_positive_minus_competitor'] < 0
                assert record['teacher_mean_pair_margin'] == float(np.mean(record['teacher_pair_margins']))
                for scope in ('focal', 'batch'):
                    for group in GROUPS:
                        values = [record['scopes'][scope][c]['groups'][group]['descent_margin_change'] for c in COMPONENTS]
                        np.testing.assert_allclose(sum(values[:3]), values[3], rtol=1e-4, atol=3e-6)
                    for component in COMPONENTS:
                        groups = record['scopes'][scope][component]['groups']
                        assert groups['temperature']['descent_margin_change'] == 0.
                        np.testing.assert_allclose(groups['representation']['descent_margin_change'],
                            groups['entities']['descent_margin_change']+groups['operators']['descent_margin_change'], atol=1e-12)
                        for data in groups.values():
                            assert data['helps_margin'] == (data['descent_margin_change'] > 1e-10)
                            assert data['harms_margin'] == (data['descent_margin_change'] < -1e-10)
            print(json.dumps(dict(stage='audit_case', case=i+1, cases=len(cases), passed=True, lr=0.)), flush=True)
    for key, value in summarize(cases, records).items():
        assert summary[key] == value, key
    assert sorted(opened) == audit['dataset_files_opened'] == summary['dataset_files_opened'] == ['raw/num-node-dict.csv.gz', 'split/random/train.pt']
    verify_hashes(receipt['hashes'])
    result = dict(audit_passed=True, source_input_artifact_hashes_verified=True, selection_and_train_batches_reproduced=True,
        saved_normalization_and_fixed_pipeline_exact=True, train_filtered_ranks_and_summary_reproduced=True,
        gradient_component_projection_sums_verified=True, unchanged_model_hashes_recorded=True,
        direct_gradients_recomputed_by_endpoint=False, direct_graph_features_recomputed_by_endpoint=False,
        direct_gradient_and_feature_checks_in_runner=True, validation_loaded=False, test_loaded=False,
        model_updates=0, lr=0.)
    write_json(out/'endpoint_audit.json', result)
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    audit(parser.parse_args().directory)
