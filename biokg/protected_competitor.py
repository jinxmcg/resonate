"""TF3: candidate-specific TRAIN CE protection and collateral gradient effects."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from resonate import ResonatE, cnorm
from biokg.confirm_feature_blend import DATASET, verify_hashes, write_json
from biokg.gradient_diagnostic import real_flat, sha256, train_only
from biokg.joint_operator import retained_loss
from biokg.mixed_operator import from_student, restore_mixed
from biokg.objective_forensics import install_guard
from biokg.train_biokg_comp import load_teacher, teacher_logits
from biokg.train_dual_operator import tensor_digest
from biokg.train_forensics import catalog_rank, neighbors, symmetric_graph


ARMS = ('baseline', 'hard_raw', 'hard_protected', 'random_raw', 'random_protected')
KINDS = ('focal_model', 'focal_pipeline', 'background', 'hard_known', 'random_known')
READOUTS = ('positive_score', 'negative_score', 'margin', 'positive_norm', 'negative_norm')


def candidate_rows(graph, sources, candidate):
    sources = np.asarray(sources, np.int64)
    return np.asarray(graph[sources, int(candidate)].toarray()).ravel() > 0


def matched_random(pool, hard, graph, source, offset, seed):
    ids = np.arange(graph.shape[0])+offset
    eligible = ~np.isin(ids, pool)
    eligible[neighbors(graph, source)] = False
    choices = ids[eligible]
    if not len(choices):
        raise ValueError('No eligible matched random candidate')
    chosen = int(np.random.default_rng(seed).choice(choices))
    result = np.asarray(pool, np.int64).copy()
    slots = result == hard
    assert slots.any() and chosen != hard
    result[slots] = chosen
    assert (result == chosen).sum() == slots.sum() and not (result == hard).any()
    np.testing.assert_array_equal(result[~slots], np.asarray(pool)[~slots])
    return chosen, result


def protection_mask(row_known, negatives, candidate, device):
    mask = np.zeros((len(row_known), len(negatives)+1), bool)
    mask[:, 1:] = np.asarray(row_known, bool)[:, None] & (np.asarray(negatives) == candidate)[None, :]
    return torch.as_tensor(mask, device=device)


def protected_loss(outputs, target, mask):
    total, parts = retained_loss(outputs, target)
    logits = outputs[0]
    if mask.shape != logits.shape or mask.dtype != torch.bool or mask[:, 0].any():
        raise ValueError('Invalid CE protection mask or masked designated positive')
    if mask.any():
        ce = F.cross_entropy(logits.masked_fill(mask, -torch.inf), torch.zeros(len(logits), device=logits.device, dtype=torch.long))
        total = ce+parts['kd_t_squared']+.1*parts['trajectory']
    else:
        ce = parts['ce']
    return total, dict(ce=ce, kd=parts['kd_t_squared'], trajectory=.1*parts['trajectory'])


def probe_readouts(entity, operator, source, relation, positive, negative):
    block = operator.shape[-1]
    x = cnorm(entity[source]).reshape(len(source), -1, block)
    q = cnorm(torch.einsum('bkij,bkj->bki', operator[relation], x).reshape(len(source), -1))
    p, n = entity[positive], entity[negative]
    ps, ns = (q*p.conj()).sum(-1).real, (q*n.conj()).sum(-1).real
    # Subtract embeddings before contraction, as in TF2's original margin.
    margin = (q*(p-n).conj()).sum(-1).real
    return torch.stack([ps, ns, margin, p.abs().square().sum(-1).sqrt(), n.abs().square().sum(-1).sqrt()], dim=1)


def probe_specs(case, sources, positives, hard_known, random_known, random_id, offset, background_limit=16, known_limit=8):
    source, positive = case['source_local_id']+offset, case['positive_local_id']+offset
    probes = [dict(kind='focal_model', source=source, positive=positive, negative=case['winner_local_id']+offset),
              dict(kind='focal_pipeline', source=source, positive=positive, negative=case['pipeline_winner_local_id']+offset)]
    for kind, mask, fixed_positive, limit in (
        ('background', np.ones(len(sources), bool), None, background_limit),
        ('hard_known', hard_known, case['winner_local_id']+offset, known_limit),
        ('random_known', random_known, random_id, known_limit),
    ):
        seen = {source}
        for row, (s, p) in enumerate(zip(sources, positives)):
            if not mask[row] or int(s) in seen:
                continue
            probes.append(dict(kind=kind, source=int(s), positive=int(p if fixed_positive is None else fixed_positive),
                               negative=None, batch_row=row))
            seen.add(int(s))
            if len(seen)-1 == limit:
                break
    return probes


@torch.no_grad()
def complete_probes(model, graph, specs, relation, offset):
    dev = model.E.device
    cache = {}
    for probe in specs:
        source = probe['source']
        if source not in cache:
            q = model.query(torch.tensor([source], device=dev), torch.tensor([relation], device=dev), model.H_b)
            scores = (q @ model.E[offset:offset+graph.shape[0]].conj().T).real[0].cpu().numpy()
            known = neighbors(graph, source-offset)
            eligible = np.ones(graph.shape[0], bool)
            eligible[known] = False
            if not eligible.any():
                raise ValueError('No TRAIN-unknown competitor for a probe')
            winner = int(np.flatnonzero(eligible)[np.argmax(scores[eligible])])+offset
            cache[source] = scores, known, winner
        scores, known, winner = cache[source]
        assert probe['positive']-offset in known
        if probe['negative'] is None:
            probe['negative'] = winner
        assert probe['negative']-offset not in known
        probe['relation'] = relation
        probe['baseline_rank'] = catalog_rank(scores, probe['positive']-offset, known)
    return specs


def probe_tensors(probes, device):
    return tuple(torch.tensor([p[key] for p in probes], device=device) for key in ('source', 'relation', 'positive', 'negative'))


def inspect_arm(model, batch, target, mask, probes):
    outputs = model.training_outputs(*batch)
    loss, parts = protected_loss(outputs, target, mask)
    ce_logits = torch.autograd.grad(parts['ce'], outputs[0], retain_graph=True)[0]
    assert torch.isfinite(loss) and torch.isfinite(ce_logits).all()
    assert torch.count_nonzero(ce_logits[mask]) == 0
    params = model.E, model.H_b, model.log_tau
    probe_ids = probe_tensors(probes, model.E.device)
    focal = probe_readouts(model.E, model.H_b, *(v[:1] for v in probe_ids))[0, 2]
    gm = torch.autograd.grad(focal, params, retain_graph=True, allow_unused=True)
    gradients = torch.autograd.grad(loss, params)
    e, h = model.E.detach(), model.H_b.detach()
    fn = lambda ent, op: probe_readouts(ent, op, *probe_ids)
    value, entity_change = torch.func.jvp(lambda ent: fn(ent, h), (e,), (-gradients[0].detach(),))
    _, operator_change = torch.func.jvp(lambda op: fn(e, op), (h,), (-gradients[1].detach(),))
    total_change = entity_change+operator_change
    direct = -sum(float(torch.dot(real_flat(a), real_flat(b))) for a, b in zip(gm[:2], gradients[:2]))
    np.testing.assert_allclose(float(total_change[0, 2]), direct, atol=2e-6, rtol=2e-4)
    np.testing.assert_allclose(total_change[:, 2].detach().cpu(), (total_change[:, 0]-total_change[:, 1]).detach().cpu(), atol=3e-6, rtol=2e-4)
    assert all(p.grad is None for p in params)
    return dict(losses={k: float(v.detach()) for k, v in parts.items()}, total_loss=float(loss.detach()),
        protected_entries=int(mask.sum()), protected_rows=int(mask.any(1).sum()),
        direct_masked_ce_gradient_zero=True, focal_reverse_mode_projection=direct,
        readouts=value.detach().cpu().numpy().tolist(),
        entity_change=entity_change.detach().cpu().numpy().tolist(),
        operator_change=operator_change.detach().cpu().numpy().tolist(),
        total_change=total_change.detach().cpu().numpy().tolist(),
        jvp_reverse_mode_parity=True, parameter_grads_absent=True)


def group_stats(probes, values):
    result = {}
    values = np.asarray(values, np.float64)
    for kind in KINDS:
        ids = [i for i, p in enumerate(probes) if p['kind'] == kind]
        if not ids:
            result[kind] = dict(probes=0, mean_margin_change=None, helps=0, harms=0)
            continue
        v = values[ids]
        result[kind] = dict(probes=len(ids), mean_margin_change=float(v[:, 2].mean()),
            helps=int((v[:, 2] > 1e-10).sum()), harms=int((v[:, 2] < -1e-10).sum()),
            mean_changes={key: float(v[:, j].mean()) for j, key in enumerate(READOUTS)})
    return result


def summarize(cases, records):
    arms = {}
    for arm in ARMS:
        selected = [r for r in records if r['arm'] == arm]
        assert len(selected) == len(cases)
        groups = {}
        for kind in KINDS:
            data = [r['groups'][kind] for r in selected if r['groups'][kind]['probes']]
            groups[kind] = dict(cases=len(data), probe_occurrences=sum(v['probes'] for v in data),
                cases_mean_helpful=sum(v['mean_margin_change'] > 1e-10 for v in data),
                cases_mean_harmful=sum(v['mean_margin_change'] < -1e-10 for v in data),
                equal_case_mean_margin_change=float(np.mean([v['mean_margin_change'] for v in data])) if data else None,
                equal_case_mean_harm_fraction=float(np.mean([v['harms']/v['probes'] for v in data])) if data else None)
        arms[arm] = dict(groups=groups, protected_rows=sum(r['protected_rows'] for r in selected))
    comparisons = {}
    by_key = {(r['case_index'], r['arm']): r for r in records}
    for treatment, control in (('hard_protected', 'random_protected'), ('hard_protected', 'hard_raw'),
                               ('hard_protected', 'baseline'), ('random_protected', 'random_raw')):
        groups = {}
        for kind in KINDS:
            delta = [by_key[i, treatment]['groups'][kind]['mean_margin_change']-by_key[i, control]['groups'][kind]['mean_margin_change']
                     for i in range(len(cases)) if by_key[i, treatment]['groups'][kind]['probes']]
            groups[kind] = dict(cases=len(delta), equal_case_mean_delta=float(np.mean(delta)) if delta else None,
                                cases_improved=sum(d > 1e-10 for d in delta), cases_worsened=sum(d < -1e-10 for d in delta))
        comparisons[treatment+'_minus_'+control] = groups
    return dict(protocol='TF3', cases=len(cases), records=len(records), arms=arms, comparisons=comparisons,
        lr=0., model_updates=0, optimizer_constructed=False, gradient_only=True,
        validation_loaded=False, test_loaded=False, external_models_loaded=False,
        inference_model_unchanged=True, submission_changed=False, training_started=False,
        repeated_probe_occurrences_not_independent=True, not_adam_or_mrr_improvement=True)


def make_receipt(repo):
    prior_dir = repo/'biokg/results/objective_forensics/s0'
    prior = json.loads((prior_dir/'prerun.json').read_text())
    audit = json.loads((prior_dir/'audit.json').read_text())
    assert audit['passed'] and json.loads((prior_dir/'endpoint_audit.json').read_text())['audit_passed']
    hashes = dict(prior['hashes'])
    for name in ('prerun.json', 'cases.json', 'batches.npz', 'gradients.jsonl', 'summary.json'):
        hashes[str(prior_dir/name)] = audit['artifact_sha256'][name]
    for path in [prior_dir/'audit.json', prior_dir/'endpoint_audit.json'] + [repo/'biokg'/n for n in (
        'PROTECTED_COMPETITOR.md', 'protected_competitor.py', 'test_protected_competitor.py', 'audit_protected_competitor.py')]:
        hashes[str(path)] = sha256(path)
    return dict(protocol='TF3', hashes=hashes, prior_tf2=str(prior_dir), checkpoint=prior['checkpoint'],
        teacher_paths=prior['teacher_paths'], arms=list(ARMS), random_seed_base=36830,
        background_probe_limit=16, known_probe_limit=8, lr=0., model_updates=0,
        ce_protection_only=True, kd_trajectory_unchanged=True, device='cuda',
        torch=torch.__version__, numpy=np.__version__)


def smoke():
    torch.manual_seed(3683)
    base = ResonatE(64, 4, k=12, block=True, block_size=4)
    model = from_student(dict(model=base.state_dict(), args=dict(k=12, block_size=4)), 'single', 'cuda').a.requires_grad_(True)
    before = tensor_digest(model.state_dict())
    batch = (torch.arange(16, device='cuda'), torch.zeros(16, device='cuda', dtype=torch.long),
             torch.arange(16, 32, device='cuda'), torch.arange(32, 64, device='cuda'))
    probes = [dict(source=i, positive=16+i, negative=32+i, relation=0, kind='background') for i in range(8)]
    mask = protection_mask(np.arange(16)%2 == 0, np.arange(32, 64), 32, 'cuda')
    result = inspect_arm(model, batch, torch.randn(16, 33, device='cuda'), mask, probes)
    assert tensor_digest(model.state_dict()) == before
    print(json.dumps(dict(smoke_passed=True, jvp_reverse_mode_parity=result['jvp_reverse_mode_parity'],
                         protected_entries=result['protected_entries'], real_data_loaded=False, lr=0.)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='biokg/results/protected_competitor/s0')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available() or '1080 Ti' not in torch.cuda.get_device_name():
        raise RuntimeError('TF3 requires authorized local1080Ti access')
    if args.smoke:
        smoke()
        return
    repo, out = Path(__file__).resolve().parents[1], Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    receipt = make_receipt(repo)
    opened = set()
    install_guard(receipt, out, opened)
    started = time.monotonic()
    def log(**event):
        event.update(seconds=time.monotonic()-started, lr=0., prior_validation_pipeline_mrr_unchanged=.8582166512116752)
        print(json.dumps(event, allow_nan=False), flush=True)
        with (out/'progress.jsonl').open('a') as handle:
            handle.write(json.dumps(event, allow_nan=False)+'\n')
    log(stage='verifying_inputs')
    verify_hashes(receipt['hashes'])
    write_json(out/'prerun.json', receipt)
    prior_dir = Path(receipt['prior_tf2'])
    cases = json.loads((prior_dir/'cases.json').read_text())
    old_audit = json.loads((prior_dir/'audit.json').read_text())
    old_records = [json.loads(line) for line in (prior_dir/'gradients.jsonl').read_text().splitlines()]
    old = {(r['case_index'], r['pool']): r for r in old_records}
    with np.load(prior_dir/'batches.npz', allow_pickle=False) as saved:
        batches = {key: saved[key] for key in saved.files}
    train, offsets, counts = train_only(DATASET)
    offset, count = offsets['drug'], counts['drug']
    h, r, t = (np.asarray(train[k]) for k in ('head', 'relation', 'tail'))
    ck = torch.load(receipt['checkpoint'], map_location='cpu', weights_only=False)
    assert ck['offset'] == offsets and ck['n_rel'] == 102 and ck['mode'] == 'single'
    model = restore_mixed(ck, 'cuda').a.requires_grad_(True)
    before = tensor_digest(model.state_dict())
    assert before == old_audit['student_before']
    teachers = []
    log(stage='loading_own_teachers')
    for seed, path in enumerate(receipt['teacher_paths']):
        meta = torch.load(path, map_location='cpu', weights_only=False)
        assert meta['offset'] == offsets and meta['n_rel'] == 102 and meta['args']['seed'] == seed and not meta['args'].get('distill')
        del meta
        teachers.append(load_teacher(path, sum(counts.values()), 102, torch.device('cuda')))
    teacher_before = [tensor_digest(tm.state_dict()) for tm in teachers]
    assert teacher_before == old_audit['teacher_before']
    graphs, records, plans = {}, [], []
    torch.cuda.reset_peak_memory_stats()
    for i, case in enumerate(cases):
        rel = case['relation']
        if rel not in graphs:
            rows = r == rel
            assert set(np.asarray(train['head_type'])[rows]) == {'drug'}
            assert set(np.asarray(train['tail_type'])[rows]) == {'drug'}
            graphs[rel] = symmetric_graph(h[rows], t[rows], count)
        graph = graphs[rel]
        source, positive, relation, baseline, hard_pool = [batches[f'case{i}_{k}'] for k in (
            'source', 'positive', 'relation', 'negatives_random', 'negatives_forced')]
        hard = case['winner_local_id']+offset
        random_id, random_pool = matched_random(hard_pool, hard, graph, case['source_local_id'], offset, 36830+i)
        hard_known = candidate_rows(graph, source-offset, hard-offset)
        random_known = candidate_rows(graph, source-offset, random_id-offset)
        assert not hard_known[0] and not random_known[0]
        probes = complete_probes(model, graph, probe_specs(case, source, positive, hard_known, random_known, random_id, offset),
                                 case['directed_relation'], offset)
        plan = dict(case_index=i, sample_index=case['sample_index'], hard_global_id=hard, random_global_id=random_id,
            random_pool=random_pool.tolist(), hard_known=hard_known.tolist(), random_known=random_known.tolist(), probes=probes,
            hard_train_degree=len(neighbors(graph, hard-offset)), random_train_degree=len(neighbors(graph, random_id-offset)),
            matched_slots=np.flatnonzero(hard_pool == hard).tolist())
        plans.append(plan)
        write_json(out/'plans.json', plans)
        src, rr, pos = [torch.as_tensor(a, device='cuda') for a in (source, relation, positive)]
        targets, arms_this_case = {}, {}
        for arm in ARMS:
            pool_key = 'baseline' if arm == 'baseline' else ('hard' if arm.startswith('hard') else 'random')
            pool = {'baseline': baseline, 'hard': hard_pool, 'random': random_pool}[pool_key]
            batch = src, rr, pos, torch.as_tensor(pool, device='cuda')
            if pool_key not in targets:
                targets[pool_key] = teacher_logits(teachers, *batch)
            mask = torch.zeros((len(src), len(pool)+1), dtype=torch.bool, device='cuda')
            if arm.endswith('protected'):
                known, candidate = (hard_known, hard) if pool_key == 'hard' else (random_known, random_id)
                mask = protection_mask(known, pool, candidate, 'cuda')
            result = inspect_arm(model, batch, targets[pool_key], mask, probes)
            result.update(case_index=i, sample_index=case['sample_index'], arm=arm)
            result['groups'] = group_stats(probes, result['total_change'])
            if arm in ('baseline', 'hard_raw'):
                previous = old[i, 'random' if arm == 'baseline' else 'forced']['scopes']['batch']['total']['groups']['representation']['descent_margin_change']
                np.testing.assert_allclose(result['focal_reverse_mode_projection'], previous, atol=2e-6, rtol=2e-4)
            if arm.endswith('protected'):
                raw = arms_this_case[pool_key+'_raw']
                assert all(result['losses'][k] == raw['losses'][k] for k in ('kd', 'trajectory'))
            arms_this_case[arm] = result
            records.append(result)
            with (out/'records.jsonl').open('a') as handle:
                handle.write(json.dumps(result, allow_nan=False)+'\n')
            log(stage='arm_complete', case=i+1, cases=len(cases), arm=arm,
                focal_change=result['groups']['focal_model']['mean_margin_change'],
                background_change=result['groups']['background']['mean_margin_change'],
                hard_known_change=result['groups']['hard_known']['mean_margin_change'], protected_rows=result['protected_rows'])
        del targets, arms_this_case
    after = tensor_digest(model.state_dict())
    teacher_after = [tensor_digest(tm.state_dict()) for tm in teachers]
    assert after == before and teacher_after == teacher_before
    assert all(p.grad is None for tm in [model, *teachers] for p in tm.parameters())
    assert all(not p.requires_grad for tm in teachers for p in tm.parameters())
    summary = summarize(cases, records)
    summary.update(seconds=time.monotonic()-started, peak_allocated_bytes=torch.cuda.max_memory_allocated(), dataset_files_opened=sorted(opened))
    write_json(out/'summary.json', summary)
    verify_hashes(receipt['hashes'])
    assert sorted(opened) == ['raw/num-node-dict.csv.gz', 'split/random/train.pt']
    names = ('prerun.json', 'plans.json', 'records.jsonl', 'summary.json')
    write_json(out/'audit.json', dict(passed=True, student_before=before, student_after=after,
        teacher_before=teacher_before, teacher_after=teacher_after, source_input_hashes_unchanged=True,
        all_parameter_grads_absent=True, optimizer_updates=0, tf2_projections_reproduced=True,
        dataset_files_opened=sorted(opened), artifact_sha256={name: sha256(out/name) for name in names}))
    log(stage='complete', cases=len(cases), records=len(records))


if __name__ == '__main__':
    main()
