"""NL1: the same complex 4x4 operators, with a fixed phase-preserving activation."""

import argparse
import copy
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from resonate import cnorm
from resonate_wiki import SparseTableResonatE
from biokg.masked_neighborhood import (
    FIT_SHA, HOLDOUT_SHA, NeighborIndex, Stream, install_guard, load_part,
    make_optimizers, score_candidates, sha, state_digest, update, verify, write_json,
)
from biokg.masked_neighborhood_run import evaluate_models, metrics, pair_interval


SPEC = dict(protocol='NL1', steps=50000, seed=0, stream_seed=0, batch=2048,
    negatives=4096, alpha=1., k=12, block_size=4, trajectory=.1,
    adam_lr=.005, table_lr=.3, probe_seed=36852, probe_rows=1024,
    probe_every=5000, evaluation_seed=36853, bootstrap_seed=36854,
    eval_chunk=256, cpu_threads=4)
NAMES = ('linear', 'nonlinear')
SOURCE_FILES = ('resonate.py', 'resonate_wiki.py', 'rowadagrad.py',
    'biokg/masked_neighborhood.py', 'biokg/masked_neighborhood_run.py',
    'biokg/test_masked_neighborhood.py', 'biokg/nonlinear_operator.py',
    'biokg/test_nonlinear_operator.py', 'biokg/NONLINEAR_OPERATOR.md',
    'biokg/run_nonlinear_vast.py', 'biokg/scripts/run_nonlinear.sh',
    'biokg/scripts/nonlinear.supervisor.conf')


def activate(z, alpha):
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError('Expected a finite nonnegative activation strength')
    return z if alpha == 0 else z/(1+alpha*z.abs())


class NonlinearOperator(SparseTableResonatE):
    def __init__(self, n_entities, n_relations, alpha, device='cpu'):
        if alpha not in (0., SPEC['alpha']):
            raise ValueError('NL1 only permits its fixed control and intervention')
        super().__init__(n_entities, n_relations, k=12, block_size=4,
                         sparse_grad=True, device=device)
        self.alpha = float(alpha)  # configuration, not a learned parameter

    def preactivation(self, z, relation):
        blocks = z.reshape(len(z), -1, self.block_size)
        return torch.einsum('bkij,bkj->bki', self.H[relation], blocks).reshape_as(z)

    def hop(self, z, relation):
        return cnorm(activate(self.preactivation(z, relation), self.alpha))


def make_pair(n, nr, device):
    linear = NonlinearOperator(n, nr, 0., device)
    nonlinear = copy.deepcopy(linear)
    nonlinear.alpha = SPEC['alpha']
    assert state_digest(linear.state_dict()) == state_digest(nonlinear.state_dict())
    assert all(a.data_ptr() != b.data_ptr() for a, b in zip(linear.parameters(), nonlinear.parameters()))
    return [linear, nonlinear]


@torch.no_grad()
def activation_diagnostic(model, source, relation):
    z = model.preactivation(model.embed(source), relation)
    magnitude = z.abs().flatten()
    gain = 1/(1+model.alpha*magnitude)
    quantiles = torch.tensor([.1, .5, .9], device=z.device)
    q, plain = cnorm(activate(z, model.alpha)), cnorm(z)
    return dict(alpha=model.alpha,
        preactivation_magnitude_q10_q50_q90=torch.quantile(magnitude, quantiles).cpu().tolist(),
        attenuation_q10_q50_q90=torch.quantile(gain, quantiles).cpu().tolist(),
        mean_query_cosine_to_same_weights_linear=float((q*plain.conj()).sum(-1).real.mean()),
        mean_operator_frobenius=float(model.H.abs().square().sum((-1, -2)).sqrt().mean()))


def snapshot(model, opts, out, name, steps):
    for opt in opts:
        opt.zero_grad(set_to_none=True)
    record = dict(model_type='NL1Single', spec=SPEC, alpha=model.alpha,
        arm=name, steps=steps, n_entities=model.n_entities, n_relations=model.n_relations,
        model=model.state_dict(), optimizers=[o.state_dict() for o in opts])
    path = Path(out)/(name+'.pt')
    if path.exists():
        raise FileExistsError(path)
    torch.save(record, path)
    return dict(model=state_digest(record['model']), optimizer=state_digest(record['optimizers']),
                checkpoint_sha256=sha(path), steps=steps, alpha=model.alpha)


def restore(path, device='cpu'):
    ck = torch.load(path, map_location=device, weights_only=False)
    if ck['model_type'] != 'NL1Single' or ck['spec'] != SPEC or ck['arm'] not in NAMES:
        raise ValueError('Not an own NL1 fixed-recipe checkpoint')
    alpha = 0. if ck['arm'] == 'linear' else SPEC['alpha']
    if ck['alpha'] != alpha:
        raise ValueError('Checkpoint activation does not match its arm')
    model = NonlinearOperator(ck['n_entities'], ck['n_relations'], alpha, device)
    model.load_state_dict(ck['model'], strict=True)
    return model.eval().requires_grad_(False)


def receipt(root, data):
    return dict(spec=SPEC, source_sha256={n: sha(root/n) for n in SOURCE_FILES},
        data_sha256={'fit_train.npz': FIT_SHA, 'manifest.json': sha(data/'manifest.json')},
        reserved_holdout_sha256_not_opened=HOLDOUT_SHA, torch=torch.__version__,
        numpy=np.__version__, gpu=torch.cuda.get_device_name(), cuda=torch.version.cuda,
        cpu_threads=torch.get_num_threads(), tf32=False, teachers=0, distillation=False)


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
    schedules = [[torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=SPEC['steps']) for o in oo] for oo in opts]
    ss = torch.as_tensor(probe['head'], device='cuda')
    rr = torch.as_tensor(probe['relation'], device='cuda')
    log(stage='fresh_pair_verified', initial_model=initial, initial_optimizer=initial_opt,
        independent_storage=True, parameters_each=27124129, holdout_opened=False,
        adam_lr=SPEC['adam_lr'], table_lr=SPEC['table_lr'],
        activation={name: activation_diagnostic(m, ss, rr) for name, m in zip(NAMES, models)})
    torch.cuda.reset_peak_memory_stats()
    last = time.monotonic()
    last_mrr = {name: 'not_evaluated_yet' for name in NAMES}
    for step in range(1, SPEC['steps']+1):
        batch = stream.sample(SPEC['batch'], SPEC['negatives'], 'cuda')
        used = [float(o.param_groups[0]['lr']) for o in opts[0]]
        losses = {}
        for name, model, oo, sched in zip(NAMES, models, opts, schedules):
            assert used == [float(o.param_groups[0]['lr']) for o in oo]
            loss, parts = update(model, oo, batch)
            losses[name] = dict(loss=loss, **parts)
            for sc in sched:
                sc.step()
        diagnostic = None
        if step % SPEC['probe_every'] == 0 or step == SPEC['steps']:
            state = torch.get_rng_state().clone()
            cuda_state = torch.cuda.get_rng_state().clone()
            numpy_state = copy.deepcopy(stream.rng.bit_generator.state)
            ranked, _ = evaluate_models(models, probe, stream, index, SPEC['probe_seed'])
            last_mrr = {name: dict(step=step, value=metrics(v)['mrr'], kind='in_sample_fit_probe_not_holdout')
                        for name, v in zip(NAMES, ranked)}
            diagnostic = {name: activation_diagnostic(m, ss, rr) for name, m in zip(NAMES, models)}
            assert torch.equal(state, torch.get_rng_state()) and torch.equal(cuda_state, torch.cuda.get_rng_state())
            assert numpy_state == stream.rng.bit_generator.state
        if step % 1000 == 0 or time.monotonic()-last >= 35 or step == SPEC['steps']:
            log(stage='paired_training', step=step, steps=SPEC['steps'], adam_lr=used[0], table_lr=used[1],
                losses=losses, fit_probe_mrr=last_mrr, activation=diagnostic if diagnostic else 'not_measured_this_step')
            last = time.monotonic()
    endpoints = {name: snapshot(m, oo, out, name, SPEC['steps']) for name, m, oo in zip(NAMES, models, opts)}
    assert len({v['model'] for v in endpoints.values()} | {initial}) == 3
    for m, oo in zip(models, opts):
        assert all(bool(torch.isfinite(p).all()) for p in m.parameters())
        assert all(int(v['step']) == SPEC['steps'] for v in oo[0].state.values())
    verify_receipt(before, root, data)
    assert sorted(opened) == ['fit_train.npz', 'manifest.json']
    log(stage='training_complete', lr=0., holdout_opened=False)
    files = ('linear.pt', 'nonlinear.pt', 'prerun.json', 'fit_probe_rows.npz', 'progress.jsonl')
    write_json(out/'training_audit.json', dict(passed=True, endpoints=endpoints,
        initial_model=initial, initial_optimizer=initial_opt, identical_initial_model_optimizer=True,
        independent_storage=True, identical_batch_stream_by_construction=True, batch_stream_sha256=stream.digest.hexdigest(),
        paired_updates_each=SPEC['steps'], parameters_each=27124129, dataset_files_opened=sorted(opened),
        holdout_opened=False, teacher_count=0, source_inputs_unchanged=True,
        seconds=time.monotonic()-start, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        artifact_sha256={name: sha(out/name) for name in files}))


def summary(ranked, held, fit, manifest):
    assert ranked.shape[0] == 2
    rr = 1/ranked.astype(np.float64)
    delta = (rr[1]-rr[0]).mean(0)
    n = sum(manifest['counts'].values())
    pair = np.minimum(held['head'], held['tail'])*n+np.maximum(held['head'], held['tail'])
    interval = pair_interval(delta, pair)
    result = dict(protocol='NL1', metrics={name: metrics(v) for name, v in zip(NAMES, ranked)},
        primary=dict(delta_mrr=float(delta.mean()), pair_cluster_95=interval,
            resampling_unit='unordered global entity pair, all original rows and directions grouped',
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
        official_validation_loaded=False, official_test_loaded=False, holdout_gradients=False, submission_changed=False)
    return result


def evaluate(root, data, out, auditing=False):
    opened = set()
    install_guard(data, out, 'audit' if auditing else 'evaluate', opened)
    before = json.loads((out/'prerun.json').read_text())
    verify_receipt(before, root, data)
    trained = json.loads((out/'training_audit.json').read_text())
    assert trained['passed'] and not trained['holdout_opened']
    assert trained['dataset_files_opened'] == ['fit_train.npz', 'manifest.json']
    assert trained['identical_initial_model_optimizer'] and trained['independent_storage']
    assert trained['identical_batch_stream_by_construction'] and trained['teacher_count'] == 0
    assert trained['paired_updates_each'] == SPEC['steps'] and trained['parameters_each'] == 27124129
    verify({str(out/n): h for n, h in trained['artifact_sha256'].items()})
    if auditing:
        progress = [json.loads(line) for line in (out/'progress.jsonl').read_text().splitlines()]
        records = [r for r in progress if r['stage'] == 'paired_training']
        assert records and records[-1]['step'] == SPEC['steps']
        assert all(a['step'] < b['step'] for a, b in zip(records, records[1:]))
        for record in records:
            factor = .5*(1+math.cos(math.pi*(record['step']-1)/SPEC['steps']))
            for name in ('adam', 'table'):
                np.testing.assert_allclose(record[name+'_lr'], SPEC[name+'_lr']*factor, atol=1e-11, rtol=1e-5)
        initial = next(r for r in progress if r['stage'] == 'fresh_pair_verified')
        assert initial['initial_model'] == trained['initial_model']
        assert initial['initial_optimizer'] == trained['initial_optimizer']
        for name in NAMES:
            ck = torch.load(out/(name+'.pt'), map_location='cpu', weights_only=False)
            record = trained['endpoints'][name]
            assert ck['steps'] == record['steps'] == SPEC['steps'] and ck['alpha'] == record['alpha']
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
            model_before=hashes, model_after=hashes, dataset_files_opened=sorted(opened),
            pair_disjoint=True, model_updates=0, seconds=time.monotonic()-start,
            artifact_sha256={name: sha(out/name) for name in ('ranks.npz', 'summary.json', 'training_audit.json')}))
        log(stage='evaluation_complete', metrics=result['metrics'], primary=result['primary'], advance_gate=result['advance_gate'])


def smoke():
    torch.manual_seed(36955)
    models = make_pair(93773, 102, 'cuda')
    opts = [make_optimizers(m) for m in models]
    batch = (torch.arange(2048, device='cuda'), torch.zeros(2048, dtype=torch.long, device='cuda'),
             torch.arange(2048, device='cuda')+2048, torch.arange(4096, device='cuda')+4096)
    for _ in range(2):
        for m, oo in zip(models, opts):
            update(m, oo, batch)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    for _ in range(20):
        for m, oo in zip(models, opts):
            update(m, oo, batch)
    torch.cuda.synchronize()
    seconds = time.monotonic()-start
    assert seconds/20 < 1., 'Synthetic paired steps too slow for bounded pilot'
    assert all(m.n_params() == 27124129 and all(torch.isfinite(p).all() for p in m.parameters()) for m in models)
    assert state_digest(models[0].state_dict()) != state_digest(models[1].state_dict())
    print(json.dumps(dict(smoke_passed=True, paired_steps=20, seconds=seconds,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), parameters_each=27124129,
        real_data_loaded=False, alpha=SPEC['alpha'])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='biokg/data')
    parser.add_argument('--out', default='biokg/results/nonlinear_operator/s0')
    phases = parser.add_mutually_exclusive_group()
    for flag in ('smoke', 'evaluate', 'audit'):
        phases.add_argument('--'+flag, action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(SPEC['cpu_threads'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available() or '5090' not in torch.cuda.get_device_name():
        raise RuntimeError('NL1 requires the authorized RTX 5090')
    root, data, out = Path(__file__).resolve().parents[1], Path(args.data).resolve(), Path(args.out).resolve()
    if args.smoke:
        smoke()
    elif args.audit or args.evaluate:
        evaluate(root, data, out, args.audit)
    else:
        train(root, data, out)


if __name__ == '__main__':
    main()
