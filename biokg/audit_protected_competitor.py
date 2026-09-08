"""TF3 saved-record audit with direct CPU frozen-model probe reconstruction."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from biokg.confirm_feature_blend import DATASET, verify_hashes, write_json
from biokg.gradient_diagnostic import train_only
from biokg.mixed_operator import restore_mixed
from biokg.objective_forensics import install_guard
from biokg.protected_competitor import (
    ARMS, candidate_rows, complete_probes, group_stats, matched_random, probe_readouts,
    probe_specs, probe_tensors, protection_mask, summarize,
)
from biokg.train_dual_operator import tensor_digest
from biokg.train_forensics import neighbors, symmetric_graph


def audit(directory):
    torch.set_num_threads(4)
    out = Path(directory).resolve()
    receipt = json.loads((out/'prerun.json').read_text())
    opened = set()
    install_guard(receipt, out, opened)
    evidence = json.loads((out/'audit.json').read_text())
    summary = json.loads((out/'summary.json').read_text())
    plans = json.loads((out/'plans.json').read_text())
    records = [json.loads(line) for line in (out/'records.jsonl').read_text().splitlines()]
    assert receipt['protocol'] == 'TF3' and receipt['arms'] == list(ARMS)
    assert receipt['lr'] == receipt['model_updates'] == 0
    assert receipt['random_seed_base'] == 36830 and receipt['background_probe_limit'] == 16 and receipt['known_probe_limit'] == 8
    assert receipt['ce_protection_only'] and receipt['kd_trajectory_unchanged']
    assert evidence['passed'] and evidence['source_input_hashes_unchanged']
    assert evidence['student_before'] == evidence['student_after']
    assert evidence['teacher_before'] == evidence['teacher_after'] and len(evidence['teacher_before']) == 10
    assert evidence['all_parameter_grads_absent'] and evidence['optimizer_updates'] == 0
    assert evidence['tf2_projections_reproduced']
    verify_hashes(receipt['hashes'])
    verify_hashes({str(out/name): digest for name, digest in evidence['artifact_sha256'].items()})
    prior = Path(receipt['prior_tf2'])
    cases = json.loads((prior/'cases.json').read_text())
    old_records = [json.loads(line) for line in (prior/'gradients.jsonl').read_text().splitlines()]
    old = {(r['case_index'], r['pool']): r for r in old_records}
    assert len(cases) == len(plans) == 16 and len(records) == 80
    train, offsets, counts = train_only(DATASET)
    offset, count = offsets['drug'], counts['drug']
    h, r, t = (np.asarray(train[k]) for k in ('head', 'relation', 'tail'))
    checkpoint = torch.load(receipt['checkpoint'], map_location='cpu', weights_only=False)
    assert checkpoint['offset'] == offsets and checkpoint['n_rel'] == 102 and checkpoint['mode'] == 'single'
    model = restore_mixed(checkpoint, 'cpu').a
    before = tensor_digest(model.state_dict())
    assert before == evidence['student_before']
    graphs, direct_probes = {}, 0
    with np.load(prior/'batches.npz', allow_pickle=False) as batches:
        for i, (case, plan) in enumerate(zip(cases, plans)):
            rel = case['relation']
            if rel not in graphs:
                rows = r == rel
                assert set(np.asarray(train['head_type'])[rows]) == {'drug'}
                assert set(np.asarray(train['tail_type'])[rows]) == {'drug'}
                graphs[rel] = symmetric_graph(h[rows], t[rows], count)
            graph = graphs[rel]
            source, positive, baseline, hard_pool = [batches[f'case{i}_{k}'] for k in (
                'source', 'positive', 'negatives_random', 'negatives_forced')]
            hard = case['winner_local_id']+offset
            random_id, random_pool = matched_random(hard_pool, hard, graph, case['source_local_id'], offset, 36830+i)
            hard_known = candidate_rows(graph, source-offset, hard-offset)
            random_known = candidate_rows(graph, source-offset, random_id-offset)
            assert not hard_known[0] and not random_known[0]
            specs = probe_specs(case, source, positive, hard_known, random_known, random_id, offset)
            probes = complete_probes(model, graph, copy.deepcopy(specs), case['directed_relation'], offset)
            expected = dict(case_index=i, sample_index=case['sample_index'], hard_global_id=hard, random_global_id=random_id,
                random_pool=random_pool.tolist(), hard_known=hard_known.tolist(), random_known=random_known.tolist(), probes=probes,
                hard_train_degree=len(neighbors(graph, hard-offset)), random_train_degree=len(neighbors(graph, random_id-offset)),
                matched_slots=np.flatnonzero(hard_pool == hard).tolist())
            assert plan == expected, f'Pool, TRAIN coverage, probe ID or CPU rank mismatch in case {i}'
            direct_probes += len(probes)
            with torch.no_grad():
                readouts = probe_readouts(model.E, model.H_b, *probe_tensors(probes, 'cpu')).numpy()
            per_arm = {}
            for j, arm in enumerate(ARMS):
                record = records[len(ARMS)*i+j]
                assert record['case_index'] == i and record['arm'] == arm and record['sample_index'] == case['sample_index']
                assert record['direct_masked_ce_gradient_zero'] and record['jvp_reverse_mode_parity'] and record['parameter_grads_absent']
                pool = baseline if arm == 'baseline' else (hard_pool if arm.startswith('hard') else random_pool)
                known, chosen = (hard_known, hard) if arm.startswith('hard') else (random_known, random_id)
                mask = protection_mask(known, pool, chosen, 'cpu') if arm.endswith('protected') else torch.zeros((len(source), len(pool)+1), dtype=torch.bool)
                assert record['protected_rows'] == int(mask.any(1).sum())
                assert record['protected_entries'] == int(mask.sum())
                arrays = {key: np.asarray(record[key], np.float64) for key in ('readouts', 'entity_change', 'operator_change', 'total_change')}
                assert all(a.shape == (len(probes), 5) and np.isfinite(a).all() for a in arrays.values())
                np.testing.assert_allclose(arrays['readouts'], readouts, atol=2e-6, rtol=2e-4)
                np.testing.assert_allclose(arrays['entity_change']+arrays['operator_change'], arrays['total_change'], atol=2e-6, rtol=2e-4)
                for key, values in arrays.items():
                    np.testing.assert_allclose(values[:, 0]-values[:, 1], values[:, 2], atol=3e-6, rtol=2e-4)
                np.testing.assert_allclose(arrays['operator_change'][:, 3:], 0., atol=1e-10)
                np.testing.assert_allclose(arrays['total_change'][0, 2], record['focal_reverse_mode_projection'], atol=2e-6, rtol=2e-4)
                np.testing.assert_allclose(record['total_loss'], sum(record['losses'].values()), atol=2e-6, rtol=2e-6)
                assert record['groups'] == group_stats(probes, record['total_change'])
                if arm in ('baseline', 'hard_raw'):
                    previous = old[i, 'random' if arm == 'baseline' else 'forced']['scopes']['batch']['total']['groups']['representation']['descent_margin_change']
                    np.testing.assert_allclose(record['focal_reverse_mode_projection'], previous, atol=2e-6, rtol=2e-4)
                if arm.endswith('protected'):
                    raw = per_arm[arm.replace('protected', 'raw')]
                    assert all(record['losses'][key] == raw['losses'][key] for key in ('kd', 'trajectory'))
                    assert record['losses']['ce'] <= raw['losses']['ce']+1e-6
                if per_arm:
                    np.testing.assert_array_equal(record['readouts'], per_arm['baseline']['readouts'])
                per_arm[arm] = record
            print(json.dumps(dict(stage='audit_case', case=i+1, cases=len(cases), probes=len(probes), passed=True, lr=0.)), flush=True)
    for key, value in summarize(cases, records).items():
        assert summary[key] == value, key
    assert tensor_digest(model.state_dict()) == before and all(p.grad is None for p in model.parameters())
    assert sorted(opened) == evidence['dataset_files_opened'] == summary['dataset_files_opened'] == ['raw/num-node-dict.csv.gz', 'split/random/train.pt']
    verify_hashes(receipt['hashes'])
    result = dict(audit_passed=True, source_input_artifact_hashes_verified=True,
        train_masks_matched_random_pools_and_probe_selection_reconstructed=True,
        direct_cpu_probe_scores_and_ranks_reproduced=True, direct_probe_occurrences=direct_probes,
        saved_derivative_arithmetic_and_summary_reproduced=True,
        tf2_focal_projection_parity=True, unchanged_model_hashes_verified=True,
        direct_gradients_recomputed_by_endpoint=False, direct_gradient_checks_in_runner=True,
        validation_loaded=False, test_loaded=False, model_updates=0, lr=0.)
    write_json(out/'endpoint_audit.json', result)
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    audit(parser.parse_args().directory)
