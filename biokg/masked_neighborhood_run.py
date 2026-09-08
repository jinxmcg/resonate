"""MN1 execution, sealed endpoint evaluation and replay audit."""

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
    FIT_SHA, HOLDOUT_SHA, SOURCE_FILES, SPEC, NeighborIndex, Stream, install_guard,
    load_part, make_model, make_optimizers, ranks, score_candidates, sha, state_digest,
    update, verify, write_json,
)


def candidates(source, positive, directed, index, typed_range, rng, count=500):
    lo, hi = typed_range
    result = rng.integers(lo, hi, size=(len(source), count))
    for _ in range(100):
        bad = (result == source[:, None]) | (result == positive[:, None]) | index.known(source[:, None], result, directed)
        if not bad.any():
            return np.c_[positive, result]
        result[bad] = rng.integers(lo, hi, size=int(bad.sum()))
    raise RuntimeError('Typed candidate rejection failed; do not change the pool policy')


@torch.no_grad()
def evaluate_models(models, part, stream, index, seed, log=lambda **event: None):
    states = [m.training for m in models]
    for model in models:
        model.eval()
    n = len(part['head'])
    result = np.full((len(models), 2, n), np.nan, np.float32)
    digest = hashlib.sha256()
    last = time.monotonic()
    for r in range(stream.n_rel):
        rows = np.flatnonzero(part['relation'] == r)
        for direction in (0, 1):
            directed = r+direction*stream.n_rel
            rng = np.random.default_rng(np.random.SeedSequence([seed, r, direction]))
            for start in range(0, len(rows), 256):
                ids = rows[start:start+256]
                source, positive = (part['head'][ids], part['tail'][ids]) if direction == 0 else (part['tail'][ids], part['head'][ids])
                cand = candidates(source, positive, directed, index, stream.ranges[directed], rng)
                for array in (np.array([directed], np.int64), part['row_id'][ids], cand):
                    digest.update(array.tobytes())
                dev = next(models[0].parameters()).device
                ss, cc = [torch.as_tensor(a, device=dev) for a in (source, cand)]
                rr = torch.full((len(ids),), directed, device=dev, dtype=torch.long)
                for j, model in enumerate(models):
                    result[j, direction, ids] = ranks(score_candidates(model, ss, rr, cc)).cpu().numpy()
                if time.monotonic()-last >= 35:
                    log(stage='evaluation_progress', relation=r, direction=direction, relation_rows_done=min(start+256, len(rows)))
                    last = time.monotonic()
    assert np.isfinite(result).all() and (result >= 1).all() and (result <= 501).all()
    for model, state in zip(models, states):
        model.train(state)
    return result, digest.hexdigest()


def metrics(values):
    values = np.asarray(values, np.float64)
    return dict(mrr=float((1/values).mean()), hits1=float((values == 1).mean()), hits10=float((values <= 10).mean()), queries=values.size)


def pair_interval(delta, pair_key, replicates=2000):
    _, inverse = np.unique(pair_key, return_inverse=True)
    sums = np.bincount(inverse, weights=delta)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(SPEC['bootstrap_seed'])
    results = []
    for _ in range(replicates):
        ids = rng.integers(len(sums), size=len(sums))
        results.append(sums[ids].sum()/counts[ids].sum())
    return np.quantile(results, [.025, .975]).tolist()


def summary(ranked, part, fit, manifest):
    names = ('base', 'control', 'masked')
    result = dict(protocol='MN1', metrics={n: metrics(v) for n, v in zip(names, ranked)})
    rr = 1/ranked.astype(np.float64)
    delta = (rr[2]-rr[1]).mean(0)
    n = sum(manifest['counts'].values())
    pair = np.minimum(part['head'], part['tail'])*n+np.maximum(part['head'], part['tail'])
    interval = pair_interval(delta, pair)
    result['primary'] = dict(delta_mrr=float(delta.mean()), pair_cluster_95=interval,
        resampling_unit='unordered global entity pair, all original rows and directions grouped',
        model_seed_uncertainty_measured=False, internal_holdout_not_official_mrr=True)
    result['advance_gate'] = bool(delta.mean() >= .001 and interval[0] > 0 and result['metrics']['masked']['mrr'] >= result['metrics']['base']['mrr'])
    result['versus_base'] = {name: float((rr[j]-rr[0]).mean()) for j, name in ((1, 'control'), (2, 'masked'))}
    result['directions'] = {str(d): {name: metrics(ranked[j, d]) for j, name in enumerate(names)} for d in (0, 1)}
    result['relations'] = {str(int(r)): {name: metrics(ranked[j][:, part['relation'] == r]) for j, name in enumerate(names)}
                           for r in np.unique(part['relation'])}
    tid = manifest['entity_types'].index('drug')
    drug = (part['head_type_id'] == tid) & (part['tail_type_id'] == tid)
    seen = np.zeros(n, bool)
    seen[np.r_[fit['head'], fit['tail']]] = True
    cold = ~seen[part['head']] | ~seen[part['tail']]
    result['slices'] = {label: {name: metrics(ranked[j][:, mask]) for j, name in enumerate(names)}
                       for label, mask in (('drug_drug', drug), ('cold_endpoint', cold), ('seen_endpoints', ~cold)) if mask.any()}
    result.update(teachers=0, distillation=False, inference_ensemble=False, neighborhood_at_inference=False,
        official_validation_loaded=False, official_test_loaded=False, holdout_gradients=False, submission_changed=False)
    return result


def snapshot(model, opts, out, name, steps):
    for opt in opts:
        opt.zero_grad(set_to_none=True)
    record = dict(model_type='MN1SparseSingle', spec=SPEC, model=model.state_dict(),
        optimizers=[o.state_dict() for o in opts], steps=steps, n_entities=model.n_entities, n_relations=model.n_relations)
    torch.save(record, out/(name+'.pt'))
    return dict(model=state_digest(record['model']), optimizer=state_digest(record['optimizers']),
                checkpoint_sha256=sha(out/(name+'.pt')), steps=steps)


def restore(path, device='cuda', with_optimizers=False):
    ck = torch.load(path, map_location=device, weights_only=False)
    if ck['model_type'] != 'MN1SparseSingle' or ck['spec'] != SPEC:
        raise ValueError('Not an own MN1 single checkpoint')
    model = make_model(ck['n_entities'], ck['n_relations'], device)
    model.load_state_dict(ck['model'])
    if not with_optimizers:
        return model.eval().requires_grad_(False)
    opts = make_optimizers(model)
    for opt, state in zip(opts, ck['optimizers']):
        opt.load_state_dict(state)
    return model.train(), opts


def receipt(root, data):
    return dict(spec=SPEC, source_sha256={name: sha(root/name) for name in SOURCE_FILES},
        data_sha256={'fit_train.npz': FIT_SHA, 'manifest.json': sha(data/'manifest.json')},
        reserved_holdout_sha256_not_opened=HOLDOUT_SHA, torch=torch.__version__, numpy=np.__version__,
        gpu=torch.cuda.get_device_name(), cuda=torch.version.cuda, cpu_threads=torch.get_num_threads(),
        tf32=False, teachers=0, distillation=False, official_validation_loaded=False, official_test_loaded=False)


def verify_receipt(saved, root, data):
    assert saved['spec'] == SPEC
    verify({str(root/n): h for n, h in saved['source_sha256'].items()})
    verify({str(data/n): h for n, h in saved['data_sha256'].items()})


def train(root, data, out):
    out.mkdir(parents=True, exist_ok=False)
    opened = set()
    install_guard(data, out, 'train', opened)
    before = receipt(root, data)
    verify_receipt(before, root, data)
    write_json(out/'prerun.json', before)
    start = time.monotonic()
    def log(**event):
        event.update(seconds=time.monotonic()-start)
        print(json.dumps(event, allow_nan=False), flush=True)
        with (out/'progress.jsonl').open('a') as handle:
            handle.write(json.dumps(event, allow_nan=False)+'\n')
    manifest = json.loads((data/'manifest.json').read_text())
    fit = load_part(data, 'fit', manifest)
    stream = Stream(fit, manifest, SPEC['seed'])
    n = sum(manifest['counts'].values())
    log(stage='fit_loaded', rows=len(fit['head']), holdout_opened=False, lr=0.)
    index = NeighborIndex(fit, n, stream.n_rel)
    pi = np.sort(np.random.default_rng(SPEC['probe_seed']).choice(len(fit['head']), SPEC['probe_rows'], replace=False))
    probe = {k: v[pi] for k, v in fit.items()}
    np.savez_compressed(out/'fit_probe_rows.npz', row_id=probe['row_id'])
    torch.manual_seed(SPEC['seed'])
    base = make_model(n, 2*stream.n_rel)
    assert base.n_params() == 27124129
    opts = make_optimizers(base)
    sched = [torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=SPEC['base_steps']) for o in opts]
    torch.cuda.reset_peak_memory_stats()
    last, last_mrr = time.monotonic(), None
    for step in range(1, SPEC['base_steps']+1):
        batch = stream.sample(SPEC['batch'], SPEC['negatives'], 'cuda')
        used = [float(o.param_groups[0]['lr']) for o in opts]
        loss, parts = update(base, opts, batch)
        for sc in sched:
            sc.step()
        if step % SPEC['probe_every'] == 0 or step == SPEC['base_steps']:
            values, _ = evaluate_models([base], probe, stream, index, SPEC['probe_seed'])
            last_mrr = dict(step=step, value=metrics(values)['mrr'], kind='in_sample_fit_probe_not_holdout')
        if step % 1000 == 0 or time.monotonic()-last >= 35 or step == SPEC['base_steps']:
            log(stage='base_training', step=step, steps=SPEC['base_steps'], loss=loss,
                adam_lr=used[0], table_lr=used[1], fit_probe_mrr=last_mrr if last_mrr is not None else 'not_evaluated_yet')
            last = time.monotonic()
    base_record = snapshot(base, opts, out, 'base', SPEC['base_steps'])
    base_stream = stream.digest.hexdigest()
    del base, opts, sched
    arms = [restore(out/'base.pt', with_optimizers=True) for _ in range(2)]
    assert state_digest(arms[0][0].state_dict()) == state_digest(arms[1][0].state_dict()) == base_record['model']
    assert state_digest([o.state_dict() for o in arms[0][1]]) == state_digest([o.state_dict() for o in arms[1][1]]) == base_record['optimizer']
    assert all(a.data_ptr() != b.data_ptr() for a, b in zip(arms[0][0].parameters(), arms[1][0].parameters()))
    schedules = []
    for model, opts in arms:
        for opt, lr in zip(opts, (SPEC['continuation_adam_lr'], SPEC['continuation_table_lr'])):
            for group in opt.param_groups:
                group['lr'] = lr
                group['initial_lr'] = lr
        schedules.append([torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=SPEC['continuation_steps']) for o in opts])
    stream = Stream(fit, manifest, SPEC['stream_seed'])
    generator = torch.Generator(device='cuda').manual_seed(SPEC['context_seed'])
    ctx_rows = ctx_batches = 0
    last_mrr = ['not_evaluated_yet']*2
    log(stage='fork_verified', initial_model=base_record['model'], initial_optimizer=base_record['optimizer'],
        independent_storage=True, adam_lr=SPEC['continuation_adam_lr'], table_lr=SPEC['continuation_table_lr'])
    for step in range(1, SPEC['continuation_steps']+1):
        batch = stream.sample(SPEC['batch'], SPEC['negatives'], 'cuda')
        directed = int(batch[1][0])
        context = valid = None
        if directed in stream.drug_relations:
            context, valid = index.sample_without_target(batch[0], batch[2], directed, SPEC['context_count'], generator)
            ctx_rows += int(valid.sum())
            ctx_batches += 1
        losses = []
        used = [float(o.param_groups[0]['lr']) for o in arms[0][1]]
        for j, (model, opts) in enumerate(arms):
            assert used == [float(o.param_groups[0]['lr']) for o in opts]
            loss, parts = update(model, opts, batch, context if j else None, valid if j else None)
            losses.append(loss)
            for sc in schedules[j]:
                sc.step()
        if step % SPEC['probe_every'] == 0 or step == SPEC['continuation_steps']:
            state = generator.get_state().clone()
            values, _ = evaluate_models([a[0] for a in arms], probe, stream, index, SPEC['probe_seed'])
            assert torch.equal(state, generator.get_state())
            last_mrr = [dict(step=step, value=metrics(v)['mrr'], kind='in_sample_fit_probe_not_holdout') for v in values]
        if step % 1000 == 0 or time.monotonic()-last >= 35 or step == SPEC['continuation_steps']:
            log(stage='paired_continuation', step=step, steps=SPEC['continuation_steps'],
                adam_lr=used[0], table_lr=used[1], control_loss=losses[0], masked_loss=losses[1],
                control_fit_probe_mrr=last_mrr[0], masked_fit_probe_mrr=last_mrr[1], context_rows=ctx_rows, context_batches=ctx_batches)
            last = time.monotonic()
    endpoints = {name: snapshot(model, opts, out, name, SPEC['base_steps']+SPEC['continuation_steps'])
                 for name, (model, opts) in zip(('control', 'masked'), arms)}
    for name in endpoints:
        assert endpoints[name]['model'] != base_record['model']
    assert endpoints['control']['model'] != endpoints['masked']['model'] and ctx_rows > 0
    for model, opts in arms:
        assert all(bool(torch.isfinite(p).all()) for p in model.parameters())
        assert all(int(state['step']) == SPEC['base_steps']+SPEC['continuation_steps'] for state in opts[0].state.values())
    verify_receipt(before, root, data)
    assert sorted(opened) == ['fit_train.npz', 'manifest.json']
    log(stage='training_complete', lr=0., holdout_opened=False, seconds_training=time.monotonic()-start)
    audit = dict(passed=True, base=base_record, endpoints=endpoints,
        identical_initial_model_optimizer=True, independent_storage=True, identical_batch_stream_by_construction=True,
        base_stream_sha256=base_stream, continuation_stream_sha256=stream.digest.hexdigest(),
        context_batches=ctx_batches, context_rows=ctx_rows, sampled_context_slots=ctx_rows*SPEC['context_count'],
        target_exclusion_checked_every_context_batch=True, adam_steps_each=62500,
        paired_updates_each=SPEC['continuation_steps'], dataset_files_opened=sorted(opened),
        holdout_opened=False, teacher_count=0, source_inputs_unchanged=True,
        seconds=time.monotonic()-start, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        artifact_sha256={name: sha(out/name) for name in ('base.pt', 'control.pt', 'masked.pt', 'prerun.json', 'fit_probe_rows.npz', 'progress.jsonl')})
    write_json(out/'training_audit.json', audit)


def evaluate(root, data, out, auditing=False):
    opened = set()
    install_guard(data, out, 'audit' if auditing else 'evaluate', opened)
    before = json.loads((out/'prerun.json').read_text())
    verify_receipt(before, root, data)
    trained = json.loads((out/'training_audit.json').read_text())
    assert trained['passed'] and not trained['holdout_opened']
    assert trained['dataset_files_opened'] == ['fit_train.npz', 'manifest.json']
    assert trained['identical_initial_model_optimizer'] and trained['independent_storage']
    assert trained['identical_batch_stream_by_construction'] and trained['target_exclusion_checked_every_context_batch']
    assert trained['paired_updates_each'] == SPEC['continuation_steps'] and trained['teacher_count'] == 0
    verify({str(out/n): h for n, h in trained['artifact_sha256'].items()})
    if auditing:
        progress = [json.loads(line) for line in (out/'progress.jsonl').read_text().splitlines()]
        for stage, prefix, total in (('base_training', 'base', SPEC['base_steps']),
                                     ('paired_continuation', 'continuation', SPEC['continuation_steps'])):
            records = [r for r in progress if r['stage'] == stage]
            assert records and records[-1]['step'] == total
            assert all(a['step'] < b['step'] for a, b in zip(records, records[1:]))
            for record in records:
                factor = .5*(1+math.cos(math.pi*(record['step']-1)/total))
                for name in ('adam', 'table'):
                    np.testing.assert_allclose(record[name+'_lr'], SPEC[prefix+'_'+name+'_lr']*factor, atol=1e-11, rtol=1e-5)
        base_acc = None
        for name in ('base', 'control', 'masked'):
            ck = torch.load(out/(name+'.pt'), map_location='cpu', weights_only=False)
            expected_steps = SPEC['base_steps']+(0 if name == 'base' else SPEC['continuation_steps'])
            recorded = trained['base'] if name == 'base' else trained['endpoints'][name]
            assert ck['steps'] == recorded['steps'] == expected_steps
            assert state_digest(ck['optimizers']) == recorded['optimizer']
            assert all(int(v['step']) == expected_steps for v in ck['optimizers'][0]['state'].values())
            acc = next(iter(ck['optimizers'][1]['state'].values()))['acc']
            assert acc.shape == (ck['n_entities'],) and torch.isfinite(acc).all() and (acc >= 0).all()
            if name == 'base':
                base_acc = acc.clone()
            else:
                assert (acc >= base_acc).all()
            assert all(float(o['param_groups'][0]['lr']) == 0. for o in ck['optimizers'])
            del ck
    manifest = json.loads((data/'manifest.json').read_text())
    fit, held = [load_part(data, name, manifest) for name in ('fit', 'holdout')]
    stream = Stream(fit, manifest, 0)
    n = sum(manifest['counts'].values())
    a = np.unique(np.minimum(fit['head'], fit['tail'])*n+np.maximum(fit['head'], fit['tail']))
    b = np.unique(np.minimum(held['head'], held['tail'])*n+np.maximum(held['head'], held['tail']))
    assert not np.intersect1d(a, b).size
    index = NeighborIndex(fit, n, stream.n_rel)
    models = [restore(out/(name+'.pt')) for name in ('base', 'control', 'masked')]
    model_hashes = [state_digest(m.state_dict()) for m in models]
    expected = [trained['base']['model']]+[trained['endpoints'][n]['model'] for n in ('control', 'masked')]
    assert model_hashes == expected
    log = lambda **event: print(json.dumps(event), flush=True)
    start = time.monotonic()
    ranked, candidate_hash = evaluate_models(models, held, stream, index, SPEC['evaluation_seed'], log)
    result = summary(ranked, held, fit, manifest)
    assert all(p.grad is None for m in models for p in m.parameters())
    assert [state_digest(m.state_dict()) for m in models] == model_hashes
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
            saved_optimizer_counters_accumulators_and_lr_schedule_verified=True,
            training_holdout_access_absent_recorded=True, no_optimizer_replay_claim=True,
            model_weights_unchanged=True, holdout_gradient_updates=0, official_validation_loaded=False,
            official_test_loaded=False, seconds=time.monotonic()-start)
        write_json(out/'endpoint_audit.json', audit)
        log(**audit)
    else:
        if (out/'summary.json').exists():
            raise FileExistsError('Do not overwrite existing endpoint results')
        np.savez_compressed(out/'ranks.npz', ranks=ranked, **{k: held[k] for k in ('row_id', 'head', 'tail', 'relation')})
        write_json(out/'summary.json', result)
        write_json(out/'evaluation_audit.json', dict(passed=True, candidate_stream_sha256=candidate_hash,
            model_before=model_hashes, model_after=model_hashes, dataset_files_opened=sorted(opened),
            pair_disjoint=True, model_updates=0, seconds=time.monotonic()-start,
            artifact_sha256={name: sha(out/name) for name in ('ranks.npz', 'summary.json', 'training_audit.json')}))
        log(stage='evaluation_complete', metrics=result['metrics'], primary=result['primary'], advance_gate=result['advance_gate'])


def smoke():
    torch.manual_seed(36855)
    n, nr = 93773, 102
    model = make_model(n, nr)
    opts = make_optimizers(model)
    h = np.repeat(np.arange(1024), 12)
    t = 1024+np.tile(np.arange(12), 1024)
    fit = dict(head=h, tail=t, relation=np.zeros(len(h), np.int64))
    index = NeighborIndex(fit, n, 1)
    generator = torch.Generator(device='cuda').manual_seed(36851)
    batch = (torch.arange(2048, device='cuda')%1024, torch.zeros(2048, device='cuda', dtype=torch.long),
             1024+torch.arange(2048, device='cuda')%12, torch.arange(4096, device='cuda')+1024)
    for _ in range(2):
        update(model, opts, batch)
    control = make_model(n, nr)
    control.load_state_dict(model.state_dict())
    copied = make_optimizers(control)
    for dst, src in zip(copied, opts):
        dst.load_state_dict(copy.deepcopy(src.state_dict()))
    assert state_digest(model.state_dict()) == state_digest(control.state_dict())
    assert state_digest([o.state_dict() for o in copied]) == state_digest([o.state_dict() for o in opts])
    torch.cuda.synchronize()
    start = time.monotonic()
    for _ in range(20):
        context, valid = index.sample_without_target(batch[0], batch[2], 0, 8, generator)
        update(control, copied, batch)
        update(model, opts, batch, context, valid)
    torch.cuda.synchronize()
    seconds = time.monotonic()-start
    assert state_digest(control.state_dict()) != state_digest(model.state_dict())
    assert seconds/20 < 1., 'Synthetic paired steps too slow for bounded pilot'
    print(json.dumps(dict(smoke_passed=True, paired_steps=20, seconds=seconds,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), real_data_loaded=False,
        model_parameters=model.n_params(), target_exclusion_verified=True)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='biokg/data')
    parser.add_argument('--out', default='biokg/results/masked_neighborhood/s0')
    for flag in ('smoke', 'evaluate', 'audit'):
        parser.add_argument('--'+flag, action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available() or '5090' not in torch.cuda.get_device_name():
        raise RuntimeError('MN1 requires the authorized 5090')
    root, data, out = Path(__file__).resolve().parents[1], Path(args.data).resolve(), Path(args.out).resolve()
    if args.smoke:
        smoke()
    elif args.audit or args.evaluate:
        evaluate(root, data, out, args.audit)
    else:
        train(root, data, out)


if __name__ == '__main__':
    main()
