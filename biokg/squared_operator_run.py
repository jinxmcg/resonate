"""H37 finite pilot, matched training, frozen evaluation and endpoint audit."""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from biokg.masked_neighborhood import (
    FIT_SHA, HOLDOUT_SHA, NeighborIndex, Stream, install_guard, load_part,
    make_optimizers, ranks, sha, state_digest, verify, write_json,
)
from biokg.masked_neighborhood_run import candidates, metrics, pair_interval
from biokg.squared_operator import (
    NAMES, SOURCE_FILES, SPEC, gram_normalizer, make_pair, restore, snapshot,
    squared_weight, update,
)


@torch.no_grad()
def evaluate_models(models, part, stream, index, seed, log=lambda **event: None):
    modes = [m.training for m in models]
    for m in models:
        m.eval()
    result = np.full((len(models), 2, len(part['head'])), np.nan, np.float32)
    digest = hashlib.sha256()
    last = time.monotonic()
    device = next(models[0].parameters()).device
    for r in range(stream.n_rel):
        rows = np.flatnonzero(part['relation'] == r)
        for direction in (0, 1):
            directed = r+direction*stream.n_rel
            rng = np.random.default_rng(np.random.SeedSequence([seed, r, direction]))
            for start in range(0, len(rows), SPEC['eval_chunk']):
                ids = rows[start:start+SPEC['eval_chunk']]
                source, positive = (part['head'][ids], part['tail'][ids]) if direction == 0 else (part['tail'][ids], part['head'][ids])
                cand = candidates(source, positive, directed, index, stream.ranges[directed], rng)
                for array in (np.array([directed], np.int64), part['row_id'][ids], cand):
                    digest.update(array.tobytes())
                ss, cc = [torch.as_tensor(a, device=device) for a in (source, cand)]
                rr = torch.full((len(ids),), directed, device=device, dtype=torch.long)
                for j, m in enumerate(models):
                    result[j, direction, ids] = ranks(m.candidate_scores(ss, rr, cc)).cpu().numpy()
                if time.monotonic()-last >= 35:
                    log(stage='evaluation_progress', relation=r, direction=direction,
                        relation_rows_done=min(start+SPEC['eval_chunk'], len(rows)))
                    last = time.monotonic()
    assert np.isfinite(result).all() and (result >= 1).all() and (result <= 501).all()
    for m, mode in zip(models, modes):
        m.train(mode)
    return result, digest.hexdigest()


def summary(ranked, held, fit, manifest):
    rr = 1/ranked.astype(np.float64)
    delta = (rr[1]-rr[0]).mean(0)
    n = sum(manifest['counts'].values())
    pair = np.minimum(held['head'], held['tail'])*n+np.maximum(held['head'], held['tail'])
    interval = pair_interval(delta, pair)
    result = dict(protocol='H37', metrics={name: metrics(v) for name, v in zip(NAMES, ranked)},
        primary=dict(delta_mrr=float(delta.mean()), pair_cluster_95=interval,
            resampling_unit='unordered global entity pair, all rows/directions/relations grouped',
            model_seed_uncertainty_measured=False, internal_holdout_not_official_mrr=True),
        advance_gate=bool(delta.mean() >= .001 and interval[0] > 0))
    result['directions'] = {str(d): {name: metrics(ranked[j, d]) for j, name in enumerate(NAMES)} for d in (0, 1)}
    result['relations'] = {str(int(r)): {name: metrics(ranked[j][:, held['relation'] == r]) for j, name in enumerate(NAMES)}
                           for r in np.unique(held['relation'])}
    tid = manifest['entity_types'].index('drug')
    drug = (held['head_type_id'] == tid) & (held['tail_type_id'] == tid)
    seen = np.zeros(n, bool)
    seen[np.r_[fit['head'], fit['tail']]] = True
    cold = ~seen[held['head']] | ~seen[held['tail']]
    result['slices'] = {label: {name: metrics(ranked[j][:, mask]) for j, name in enumerate(NAMES)}
        for label, mask in (('drug_drug', drug), ('cold_endpoint', cold), ('seen_endpoints', ~cold)) if mask.any()}
    result.update(teachers=0, distillation=False, inference_ensemble=False, extra_parameters=0,
        official_validation_loaded=False, official_test_loaded=False, holdout_gradients=False,
        cold_in_fit_rows_may_receive_normalizer_gradients=True, submission_changed=False)
    return result


def source_receipt(root):
    return dict(spec=SPEC, source_sha256={name: sha(root/name) for name in SOURCE_FILES},
        torch=torch.__version__, numpy=np.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(), cpu_threads=torch.get_num_threads(), tf32=False)


def verify_sources(receipt, root):
    assert receipt['spec'] == SPEC
    verify({str(root/n): h for n, h in receipt['source_sha256'].items()})


def verify_receipt(receipt, root, data):
    verify_sources(receipt, root)
    verify({str(data/n): h for n, h in receipt['data_sha256'].items()})


def pilot(root, out):
    if out.exists():
        raise FileExistsError('Do not overwrite a pilot outcome')
    out.parent.mkdir(parents=True, exist_ok=True)
    receipt = source_receipt(root)
    torch.manual_seed(SPEC['pilot_seed'])
    # Independent direct enumeration checks every query, entity and scale gradient.
    q = torch.randn(8, 144, dtype=torch.complex64, device='cuda', requires_grad=True)
    v = torch.randn(31, 144, dtype=torch.complex64, device='cuda', requires_grad=True)
    scale = torch.tensor(.2, device='cuda', requires_grad=True)
    explicit = squared_weight((q @ v.conj().T).real, scale).sum(-1)
    compact = gram_normalizer(q, v, scale)
    torch.testing.assert_close(compact, explicit, rtol=3e-5, atol=2e-3)
    direct_grad = torch.autograd.grad(explicit.log().mean(), (q, v, scale), retain_graph=True)
    compact_grad = torch.autograd.grad(compact.log().mean(), (q, v, scale))
    for a, b in zip(direct_grad, compact_grad):
        torch.testing.assert_close(a, b, rtol=2e-4, atol=2e-6)
    del q, v, scale, explicit, compact, direct_grad, compact_grad
    models = make_pair(93773, 102, 'cuda')
    opts = [make_optimizers(m, SPEC['adam_lr'], SPEC['table_lr']) for m in models]
    batch = (torch.arange(2048, device='cuda'), torch.zeros(2048, dtype=torch.long, device='cuda'),
             torch.arange(2048, device='cuda')+21220, torch.arange(4096, device='cuda')+21220)
    support = (21220, 21220+45085)
    for _ in range(3):
        for m, oo in zip(models, opts):
            update(m, oo, batch, support)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    timings = []
    for _ in range(2):
        start = time.monotonic()
        for _ in range(20):
            for m, oo in zip(models, opts):
                update(m, oo, batch, support)
        torch.cuda.synchronize()
        timings.append((time.monotonic()-start)/20)
    peak = torch.cuda.max_memory_allocated()
    finite = all(m.n_params() == 27124129 and all(torch.isfinite(p).all() for p in m.parameters()) for m in models)
    passed = bool(finite and max(timings) <= SPEC['pilot_max_pair_seconds'] and peak <= SPEC['pilot_max_allocated_bytes'])
    verify_sources(receipt, root)
    result = dict(passed=passed, receipt=receipt, real_data_loaded=False,
        explicit_probability_gradient_check_passed=True, largest_eligible_type=45085,
        timed_paired_steps=40, seconds_per_pair_blocks=timings, peak_allocated_bytes=peak,
        parameters_each=27124129, finite=bool(finite), ideal_training_seconds=max(timings)*SPEC['steps'])
    write_json(out, result)
    print(json.dumps(dict(stage='pilot_complete', **result), allow_nan=False), flush=True)
    if not passed:
        raise RuntimeError('H37 practicality gate failed: do not launch real training')


def train(root, data, out, pilot_path):
    pilot_result = json.loads(pilot_path.read_text())
    assert pilot_result['passed'] and not pilot_result['real_data_loaded']
    assert pilot_result['explicit_probability_gradient_check_passed']
    verify_sources(pilot_result['receipt'], root)
    out.mkdir(parents=True, exist_ok=False)
    opened = set()
    install_guard(data, out, 'train', opened)
    before = source_receipt(root)
    before.update(data_sha256={'fit_train.npz': FIT_SHA, 'manifest.json': sha(data/'manifest.json')},
        reserved_holdout_sha256_not_opened=HOLDOUT_SHA, pilot_sha256=sha(pilot_path),
        teachers=0, distillation=False, official_validation_loaded=False, official_test_loaded=False)
    verify_receipt(before, root, data)
    write_json(out/'prerun.json', before)
    start = time.monotonic()
    def log(**event):
        event['seconds'] = time.monotonic()-start
        line = json.dumps(event, allow_nan=False)
        print(line, flush=True)
        with (out/'progress.jsonl').open('a') as handle:
            handle.write(line+'\n')
    manifest = json.loads((data/'manifest.json').read_text())
    fit = load_part(data, 'fit', manifest)
    stream = Stream(fit, manifest, SPEC['stream_seed'])
    n = sum(manifest['counts'].values())
    index = NeighborIndex(fit, n, stream.n_rel)
    pi = np.sort(np.random.default_rng(SPEC['probe_seed']).choice(len(fit['head']), SPEC['probe_rows'], replace=False))
    probe = {k: v[pi] for k, v in fit.items()}
    np.savez_compressed(out/'fit_probe_rows.npz', row_id=probe['row_id'])
    torch.manual_seed(SPEC['seed'])
    models = make_pair(n, 2*stream.n_rel, 'cuda')
    opts = [make_optimizers(m, SPEC['adam_lr'], SPEC['table_lr']) for m in models]
    assert all(m.n_params() == 27124129 for m in models)
    initial = state_digest(models[0].state_dict())
    initial_opt = state_digest([o.state_dict() for o in opts[0]])
    assert initial_opt == state_digest([o.state_dict() for o in opts[1]])
    scheds = [[torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=SPEC['steps']) for o in oo] for oo in opts]
    log(stage='fresh_pair_verified', initial_model=initial, initial_optimizer=initial_opt,
        independent_storage=True, parameters_each=27124129, holdout_opened=False,
        adam_lr=SPEC['adam_lr'], table_lr=SPEC['table_lr'])
    torch.cuda.reset_peak_memory_stats()
    last, last_mrr = time.monotonic(), {name: 'not_evaluated_yet' for name in NAMES}
    for step in range(1, SPEC['steps']+1):
        if time.monotonic()-start > SPEC['train_max_seconds']:
            log(stage='practical_time_cap_exceeded', step=step, holdout_opened=False)
            raise RuntimeError('Do not evaluate partial endpoints after practical time-cap stop')
        batch = stream.sample(SPEC['batch'], SPEC['negatives'], 'cuda')
        support = stream.ranges[int(batch[1][0])]
        used = [float(o.param_groups[0]['lr']) for o in opts[0]]
        losses = {}
        for name, m, oo, sched in zip(NAMES, models, opts, scheds):
            assert used == [float(o.param_groups[0]['lr']) for o in oo]
            value, parts = update(m, oo, batch, support)
            losses[name] = dict(loss=value, **parts)
            for sc in sched:
                sc.step()
        if step % SPEC['probe_every'] == 0 or step == SPEC['steps']:
            cpu_rng, gpu_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
            np_rng = copy.deepcopy(stream.rng.bit_generator.state)
            values, _ = evaluate_models(models, probe, stream, index, SPEC['probe_seed'])
            last_mrr = {name: dict(step=step, value=metrics(v)['mrr'], kind='in_sample_fit_probe_not_holdout') for name, v in zip(NAMES, values)}
            assert torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(gpu_rng, torch.cuda.get_rng_state())
            assert np_rng == stream.rng.bit_generator.state
        if step % 1000 == 0 or time.monotonic()-last >= 35 or step == SPEC['steps']:
            log(stage='paired_training', step=step, steps=SPEC['steps'], adam_lr=used[0], table_lr=used[1],
                losses=losses, fit_probe_mrr=last_mrr)
            last = time.monotonic()
    endpoints = {name: snapshot(m, oo, out, SPEC['steps']) for name, m, oo in zip(NAMES, models, opts)}
    assert len({v['model'] for v in endpoints.values()} | {initial}) == 3
    for m, oo in zip(models, opts):
        assert all(bool(torch.isfinite(p).all()) for p in m.parameters())
        assert all(int(v['step']) == SPEC['steps'] for v in oo[0].state.values())
    verify_receipt(before, root, data)
    assert sha(pilot_path) == before['pilot_sha256']
    assert sorted(opened) == ['fit_train.npz', 'manifest.json']
    log(stage='training_complete', lr=0., holdout_opened=False)
    files = ('linear.pt', 'squared.pt', 'prerun.json', 'fit_probe_rows.npz', 'progress.jsonl')
    write_json(out/'training_audit.json', dict(passed=True, endpoints=endpoints,
        initial_model=initial, initial_optimizer=initial_opt, identical_initial_model_optimizer=True,
        independent_storage=True, identical_positive_batch_stream_by_construction=True,
        exact_normalizer_recomputed_with_gradients_every_squared_step=True,
        batch_stream_sha256=stream.digest.hexdigest(), paired_updates_each=SPEC['steps'], parameters_each=27124129,
        dataset_files_opened=sorted(opened), holdout_opened=False, teacher_count=0, source_inputs_unchanged=True,
        seconds=time.monotonic()-start, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        artifact_sha256={name: sha(out/name) for name in files}))


def evaluate(root, data, out, pilot_path, auditing=False):
    opened = set()
    install_guard(data, out, 'audit' if auditing else 'evaluate', opened)
    before = json.loads((out/'prerun.json').read_text())
    verify_receipt(before, root, data)
    assert sha(pilot_path) == before['pilot_sha256']
    pilot_result = json.loads(pilot_path.read_text())
    assert pilot_result['passed']
    verify_sources(pilot_result['receipt'], root)
    trained = json.loads((out/'training_audit.json').read_text())
    assert trained['passed'] and not trained['holdout_opened']
    assert trained['dataset_files_opened'] == ['fit_train.npz', 'manifest.json']
    assert trained['identical_initial_model_optimizer'] and trained['independent_storage']
    assert trained['identical_positive_batch_stream_by_construction'] and trained['teacher_count'] == 0
    assert trained['exact_normalizer_recomputed_with_gradients_every_squared_step']
    assert trained['paired_updates_each'] == SPEC['steps'] and trained['parameters_each'] == 27124129
    verify({str(out/n): h for n, h in trained['artifact_sha256'].items()})
    if auditing:
        records = [json.loads(line) for line in (out/'progress.jsonl').read_text().splitlines()]
        progress = [r for r in records if r['stage'] == 'paired_training']
        assert progress and progress[-1]['step'] == SPEC['steps']
        assert all(a['step'] < b['step'] for a, b in zip(progress, progress[1:]))
        for record in progress:
            factor = .5*(1+math.cos(math.pi*(record['step']-1)/SPEC['steps']))
            for name in ('adam', 'table'):
                np.testing.assert_allclose(record[name+'_lr'], SPEC[name+'_lr']*factor, atol=1e-11, rtol=1e-5)
        initial = next(r for r in records if r['stage'] == 'fresh_pair_verified')
        assert initial['initial_model'] == trained['initial_model'] and initial['initial_optimizer'] == trained['initial_optimizer']
        for name in NAMES:
            ck = torch.load(out/(name+'.pt'), map_location='cpu', weights_only=False)
            record = trained['endpoints'][name]
            assert ck['steps'] == record['steps'] == SPEC['steps'] and ck['arm'] == record['arm'] == name
            assert state_digest(ck['optimizers']) == record['optimizer']
            assert all(int(v['step']) == SPEC['steps'] for v in ck['optimizers'][0]['state'].values())
            for state in ck['optimizers'][0]['state'].values():
                assert all(torch.isfinite(v).all() for v in state.values() if isinstance(v, torch.Tensor))
            acc = next(iter(ck['optimizers'][1]['state'].values()))['acc']
            assert acc.shape == (ck['n_entities'],) and torch.isfinite(acc).all() and (acc >= 0).all()
            assert all(float(o['param_groups'][0]['lr']) == 0. for o in ck['optimizers'])
            del ck
    manifest = json.loads((data/'manifest.json').read_text())
    fit, held = [load_part(data, name, manifest) for name in ('fit', 'holdout')]
    stream = Stream(fit, manifest, 0)
    n = sum(manifest['counts'].values())
    if auditing:
        torch.manual_seed(SPEC['seed'])
        initial_models = make_pair(n, 2*stream.n_rel, 'cuda')
        assert all(state_digest(m.state_dict()) == trained['initial_model'] for m in initial_models)
        for m in initial_models:
            oo = make_optimizers(m, SPEC['adam_lr'], SPEC['table_lr'])
            assert state_digest([o.state_dict() for o in oo]) == trained['initial_optimizer']
        del initial_models, m, oo
    pairs = [np.unique(np.minimum(p['head'], p['tail'])*n+np.maximum(p['head'], p['tail'])) for p in (fit, held)]
    assert not np.intersect1d(*pairs).size
    index = NeighborIndex(fit, n, stream.n_rel)
    models = [restore(out/(name+'.pt'), 'cuda') for name in NAMES]
    hashes = [state_digest(m.state_dict()) for m in models]
    assert hashes == [trained['endpoints'][name]['model'] for name in NAMES]
    assert all(m.n_params() == 27124129 for m in models)
    log = lambda **event: print(json.dumps(event, allow_nan=False), flush=True)
    start = time.monotonic()
    ranked, candidate_hash = evaluate_models(models, held, stream, index, SPEC['evaluation_seed'], log)
    result = summary(ranked, held, fit, manifest)
    assert all(p.grad is None for m in models for p in m.parameters())
    assert [state_digest(m.state_dict()) for m in models] == hashes
    verify_receipt(before, root, data)
    assert sorted(opened) == ['fit_train.npz', 'holdout_train.npz', 'manifest.json']
    if auditing:
        saved = json.loads((out/'evaluation_audit.json').read_text())
        verify({str(out/name): h for name, h in saved['artifact_sha256'].items()})
        assert json.loads((out/'summary.json').read_text()) == result
        with np.load(out/'ranks.npz', allow_pickle=False) as arrays:
            np.testing.assert_array_equal(arrays['ranks'], ranked)
            for k in ('row_id', 'head', 'tail', 'relation'):
                np.testing.assert_array_equal(arrays[k], held[k])
        assert candidate_hash == saved['candidate_stream_sha256']
        audit = dict(audit_passed=True, full_endpoint_scores_ranks_and_candidates_replayed=True,
            pair_cluster_summary_reproduced=True, source_input_artifact_hashes_verified=True,
            fresh_initial_weights_and_optimizer_state_regenerated=True,
            saved_optimizer_counters_accumulators_and_lr_schedule_verified=True,
            training_holdout_access_absent_recorded=True, no_optimizer_replay_claim=True,
            model_weights_unchanged=True, holdout_gradient_updates=0, official_validation_loaded=False,
            official_test_loaded=False, parameters_each=27124129, seconds=time.monotonic()-start)
        if (out/'endpoint_audit.json').exists():
            raise FileExistsError('Do not overwrite an existing audit')
        write_json(out/'endpoint_audit.json', audit)
        log(**audit)
    else:
        if (out/'summary.json').exists():
            raise FileExistsError('Do not overwrite existing endpoint results')
        np.savez_compressed(out/'ranks.npz', ranks=ranked, **{k: held[k] for k in ('row_id', 'head', 'tail', 'relation')})
        write_json(out/'summary.json', result)
        write_json(out/'evaluation_audit.json', dict(passed=True, candidate_stream_sha256=candidate_hash,
            model_before=hashes, model_after=hashes, dataset_files_opened=sorted(opened), pair_disjoint=True,
            model_updates=0, seconds=time.monotonic()-start,
            artifact_sha256={name: sha(out/name) for name in ('ranks.npz', 'summary.json', 'training_audit.json')}))
        log(stage='evaluation_complete', metrics=result['metrics'], primary=result['primary'], advance_gate=result['advance_gate'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='biokg/data')
    parser.add_argument('--out', default='biokg/results/h37/s0')
    parser.add_argument('--pilot-path', default='biokg/results/h37/pilot.json')
    phases = parser.add_mutually_exclusive_group()
    for flag in ('pilot', 'evaluate', 'audit'):
        phases.add_argument('--'+flag, action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(SPEC['cpu_threads'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available() or '5090' not in torch.cuda.get_device_name():
        raise RuntimeError('H37 requires the authorized RTX 5090')
    root = Path(__file__).resolve().parents[1]
    data, out, pilot_path = [Path(p).resolve() for p in (args.data, args.out, args.pilot_path)]
    if args.pilot:
        pilot(root, pilot_path)
    elif args.audit or args.evaluate:
        evaluate(root, data, out, pilot_path, args.audit)
    else:
        train(root, data, out, pilot_path)


if __name__ == '__main__':
    main()
