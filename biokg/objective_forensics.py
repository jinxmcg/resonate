"""TF2: TRAIN-only objective projections and existing retrieval on near misses."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import sparse

from resonate import cnorm
from biokg.candidate_retrieval import baseline_scores, cosine_cache, direct_features, pool_cached
from biokg.compare_feature_pipeline import metrics, score_batch
from biokg.confirm_feature_blend import CHECKPOINT_SHA, DATASET, verify_hashes, views, write_json
from biokg.gradient_diagnostic import check_dataset_path, real_flat, sha256, train_only
from biokg.joint_operator import retained_loss
from biokg.mixed_operator import restore_mixed
from biokg.relation_analogy import normalize
from biokg.reverse_scoring import cs_scores
from biokg.train_biokg_comp import load_teacher, teacher_logits
from biokg.train_dual_operator import tensor_digest
from biokg.train_forensics import catalog_rank, neighbors, symmetric_graph


COMPONENTS = ('ce', 'kd', 'trajectory', 'total')
GROUPS = ('entities', 'operators', 'temperature', 'representation')


def select_cases(records, per_direction=8):
    selected, pairs = [], set()
    for direction in (0, 1):
        count = 0
        for record in records:
            pair = tuple(sorted((record['source_local_id'], record['positive_local_id'])))
            if record['direction'] != direction or not 1 < record['filtered_rank'] <= 10 or pair in pairs:
                continue
            selected.append(dict(record))
            pairs.add(pair)
            count += 1
            if count == per_direction:
                break
    return selected


def pair_removed_graph(head, tail, count, source, positive, direction):
    head, tail = np.asarray(head), np.asarray(tail)
    remove = ((head == source) & (tail == positive)) | ((head == positive) & (tail == source))
    h, t = (head[~remove], tail[~remove]) if direction == 0 else (tail[~remove], head[~remove])
    graph = sparse.csr_matrix((np.ones(len(h), np.float32), (h, t)), shape=(count, count))
    graph.sum_duplicates()
    graph.data.fill(1)
    graph.sort_indices()
    assert graph[source, positive] == graph[positive, source] == 0
    return graph, int(remove.sum())


def blend(raw, bc_recipe, cs_recipe, relation, direction):
    """Seven channels: model, holder max/top3, Jaccard max/top3, candidate max/top3."""
    normalized = normalize(np.asarray(raw, np.float32))
    old = np.zeros((8, 1, raw.shape[-1]), np.float32)
    old[3:8, 0] = normalized[:5]
    rel, direct = np.array([relation]), np.array([direction])
    base = baseline_scores(views(old)['half_strength'], bc_recipe, rel, direct)
    pair = (normalized[5:6]+normalized[6:7])*np.float32(.5)
    return cs_scores(base, pair, cs_recipe, rel, direct)[0], normalized


@torch.inference_mode()
def retrieval_features(model, graph, source, directed, offset, cosine):
    count = graph.shape[0]
    global_sources = np.arange(count)+offset
    local_query = np.array([source])
    table = cnorm(model.E[offset:offset+count])
    # Preserve the existing scorer's arithmetic and coordinate indexing.
    global_graph = sparse.hstack([sparse.csr_matrix((count, offset)), graph,
        sparse.csr_matrix((count, len(model.E)-offset-count))], format='csr')
    degree = np.diff(graph.indptr).astype(np.float32)
    candidates = global_sources[None, :]
    old = score_batch([model, model], [table, table], global_graph, global_graph.tocsc(),
                       degree, local_query, global_sources, directed, candidates)
    known = neighbors(graph, source)
    extra = pool_cached(cosine, known, np.arange(count))
    raw = np.concatenate([old[3:8, 0], extra])
    cols = np.unique(np.r_[np.linspace(0, count-1, min(17, count), dtype=np.int64), known[:3]])
    reference = direct_features(table.numpy(), known, cols)
    np.testing.assert_allclose(extra[:, cols], reference, atol=1e-5, rtol=1e-5)
    permuted = score_batch([model, model], [table, table], global_graph, global_graph.tocsc(),
        degree, local_query, global_sources, directed, candidates[:, cols[::-1]].copy())
    np.testing.assert_allclose(old[:, 0, cols[::-1]], permuted[:, 0], atol=1e-5, rtol=1e-5)
    np.testing.assert_array_equal(extra[:, cols[::-1]], pool_cached(cosine, known, cols[::-1], 7, 11))
    return raw


def sample_batch(train, relation_rows, case, index, offset, count, batch_size=2048, negative_count=4096):
    rng = np.random.default_rng(36820+index)
    rows = np.r_[case['train_row'], rng.choice(relation_rows, batch_size-1, replace=True)]
    source_key, positive_key = ('head', 'tail') if case['direction'] == 0 else ('tail', 'head')
    source = np.asarray(train[source_key])[rows]+offset
    positive = np.asarray(train[positive_key])[rows]+offset
    negatives = rng.integers(offset, offset+count, size=negative_count)
    forced = negatives.copy()
    winner = case['winner_local_id']+offset
    if winner not in forced:
        forced[0] = winner
    return dict(train_rows=rows, source=source, positive=positive, negatives_random=negatives,
                negatives_forced=forced, relation=np.full(batch_size, case['directed_relation'], np.int64))


def gradient_measure(margin_gradients, loss_gradients):
    norm_m, norm_g, gains = [], [], []
    for gm, gl in zip(margin_gradients, loss_gradients):
        norm_m.append(float(real_flat(gm).square().sum()) if gm is not None else 0.)
        norm_g.append(float(real_flat(gl).square().sum()) if gl is not None else 0.)
        gains.append(-float(torch.dot(real_flat(gm), real_flat(gl))) if gm is not None and gl is not None else 0.)
    result = {}
    for group, ids in (('entities', [0]), ('operators', [1]), ('temperature', [2]), ('representation', [0, 1])):
        m, g, gain = np.sqrt(sum(norm_m[i] for i in ids)), np.sqrt(sum(norm_g[i] for i in ids)), sum(gains[i] for i in ids)
        result[group] = dict(descent_margin_change=gain, margin_gradient_norm=float(m), loss_gradient_norm=float(g),
            descent_cosine=float(np.clip(gain/(m*g), -1, 1)) if m*g else None,
            helps_margin=bool(gain > 1e-10), harms_margin=bool(gain < -1e-10))
    return result


def inspect_objective(model, batch, teacher_target, winner):
    """No backward(), optimizer, parameter edits, or finite update."""
    outputs = model.training_outputs(*batch)
    source, relation, positive, negatives = batch
    # Temperature-independent positive-minus-competitor margin.
    margin = (outputs[1][0]*(model.E[positive[0]]-model.E[winner]).conj()).sum().real
    params = (model.E, model.H_b, model.log_tau)
    gm = torch.autograd.grad(margin, params, retain_graph=True, allow_unused=True)
    scopes = {}
    for scope in ('focal', 'batch'):
        out = tuple(x[:1] for x in outputs) if scope == 'focal' else outputs
        target = teacher_target[:1] if scope == 'focal' else teacher_target
        total, parts = retained_loss(out, target)
        losses = dict(ce=parts['ce'], kd=parts['kd_t_squared'], trajectory=.1*parts['trajectory'], total=total)
        projected, summed = {}, [torch.zeros_like(p) for p in params]
        for name, loss in losses.items():
            grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
            if name != 'total':
                for i, value in enumerate(grads):
                    if value is not None:
                        summed[i].add_(value)
            else:
                for actual, expected in zip(summed, grads):
                    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
            projected[name] = dict(loss=float(loss.detach()), groups=gradient_measure(gm, grads))
        scopes[scope] = projected
        del summed, grads
    assert all(p.grad is None for p in params)
    return dict(unscaled_positive_minus_competitor=float(margin.detach()), scopes=scopes,
                summed_gradient_parity_passed=True, parameter_grads_absent=True)


def summarize(cases, records):
    summary = dict(protocol='TF2', cases=len(cases), gradient_records=len(records),
        cases_by_direction={str(d): sum(c['direction'] == d for c in cases) for d in (0, 1)},
        retrieval=dict(model=metrics(np.array([c['model_rank'] for c in cases])),
            pipeline=metrics(np.array([c['pipeline_rank'] for c in cases])),
            recovered_top1=sum(c['pipeline_rank'] == 1 for c in cases),
            ranks_improved=sum(c['pipeline_rank'] < c['model_rank'] for c in cases),
            ranks_worsened=sum(c['pipeline_rank'] > c['model_rank'] for c in cases),
            original_competitor_overtaken=sum(c['pipeline_pair_margin'] > 0 for c in cases)),
        pools={}, model_updates=0, lr=0., no_new_hyperparameter_fitting=True,
        validation_edges_loaded=False, test_loaded=False, external_models_loaded=False,
        gradients_computed_on_train_only=True, optimizer_constructed=False,
        not_adam_update_or_generalization_estimate=True, submission_changed=False)
    for pool in ('random', 'forced'):
        chosen = [r for r in records if r['pool'] == pool]
        report = dict(queries=len(chosen), competitor_present=sum(r['competitor_draws'] > 0 for r in chosen),
            teacher_mean_prefers_positive=sum(r['teacher_mean_pair_margin'] > 0 for r in chosen), scopes={})
        for scope in ('focal', 'batch'):
            report['scopes'][scope] = {component: {group: dict(
                helps=sum(r['scopes'][scope][component]['groups'][group]['helps_margin'] for r in chosen),
                harms=sum(r['scopes'][scope][component]['groups'][group]['harms_margin'] for r in chosen),
                mean_change=float(np.mean([r['scopes'][scope][component]['groups'][group]['descent_margin_change'] for r in chosen])))
                for group in GROUPS} for component in COMPONENTS}
        summary['pools'][pool] = report
    return summary


def make_receipt(repo):
    checkpoint = repo/'biokg/results/h35f/campaign_s0/single.pt'
    prior_path = checkpoint.parent/'prerun.json'
    prior = json.loads(prior_path.read_text())
    tf1 = repo/'biokg/results/train_forensics/s0'
    tf1_audit = json.loads((tf1/'audit.json').read_text())
    cs = repo/'biokg/results/candidate_retrieval/s0_retry1'
    cs_audit = json.loads((cs/'audit.json').read_text())
    bc = repo/'biokg/results/blend_confirmation/s0'
    bc_audit = json.loads((bc/'audit.json').read_text())
    assert tf1_audit['passed'] and json.loads((cs/'endpoint_audit.json').read_text())['audit_passed']
    teacher_paths = [Path('/mnt/geocore/wiki_pull/h24/dense')/f'sparse_s{s}.pt' for s in range(10)]
    hashes = {str(checkpoint): CHECKPOINT_SHA, str(DATASET/'split/random/train.pt'): prior['input_sha256'][str(DATASET/'split/random/train.pt')]}
    hashes.update({str(p): prior['teacher_sha256'][p.name] for p in teacher_paths})
    for name in ('findings.json', 'catalog_scores.npy', 'prerun.json'):
        hashes[str(tf1/name)] = tf1_audit['artifact_sha256'][name]
    bc_recipe = bc/'seed0_fold0_half_strength_recipe.json'
    cs_recipe = cs/'seed0_fold0_beta_recipe.json'
    hashes[str(bc_recipe)] = bc_audit['artifact_sha256'][bc_recipe.name]
    hashes[str(cs_recipe)] = cs_audit['artifact_sha256'][cs_recipe.name]
    for name in ('resonate.py', 'biokg/dual_operator.py', 'biokg/joint_operator.py', 'biokg/mixed_operator.py',
                 'biokg/train_biokg_comp.py', 'biokg/train_dual_operator.py', 'biokg/branch_warmup.py',
                 'biokg/relation_analogy.py', 'biokg/train_joint_operator.py'):
        hashes[str(repo/name)] = prior['source_hashes'][name]
    paths = [DATASET/'raw/num-node-dict.csv.gz', prior_path, tf1/'audit.json', bc/'audit.json', cs/'audit.json']
    paths += [repo/'biokg'/n for n in ('OBJECTIVE_FORENSICS.md', 'objective_forensics.py', 'test_objective_forensics.py',
        'audit_objective_forensics.py', 'train_forensics.py', 'gradient_diagnostic.py', 'candidate_retrieval.py',
        'compare_feature_pipeline.py', 'confirm_feature_blend.py', 'retrieval_followups.py', 'reverse_scoring.py')]
    hashes.update({str(p): sha256(p) for p in paths})
    return dict(protocol='TF2', hashes=hashes, checkpoint=str(checkpoint), teacher_paths=list(map(str, teacher_paths)),
        tf1=str(tf1), bc_recipe=str(bc_recipe), cs_recipe=str(cs_recipe), cases_per_direction=8,
        batch_size=2048, negatives=4096, batch_seed_base=36820, lr=0., model_updates=0,
        gradient_device='cuda', retrieval_device='cpu', torch=torch.__version__, numpy=np.__version__)


def install_guard(receipt, out, opened):
    allowed = {str(Path(p).resolve()) for p in receipt['hashes']}
    def guard(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes)):
            path = Path(args[0].decode() if isinstance(args[0], bytes) else args[0]).resolve()
            relative = check_dataset_path(path, DATASET)
            if relative:
                opened.add(relative)
            if path.suffix in ('.pt', '.pth', '.pkl', '.npy', '.npz', '.safetensors', '.bin'):
                if str(path) not in allowed and not path.is_relative_to(out):
                    raise PermissionError(f'Unapproved TF2 binary: {path}')
    sys.addaudithook(guard)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='biokg/results/objective_forensics/s0')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available() or '1080 Ti' not in torch.cuda.get_device_name():
        raise RuntimeError('TF2 requires the authorized local1080Ti')
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
    train, offsets, counts = train_only(DATASET)
    offset, count = offsets['drug'], counts['drug']
    tf1 = json.loads((Path(receipt['tf1'])/'findings.json').read_text())
    cases = select_cases(tf1['records'])
    assert cases and cases[0]['sample_index'] == tf1['case']['sample_index']
    old_scores = np.load(Path(receipt['tf1'])/'catalog_scores.npy', mmap_mode='r', allow_pickle=False)
    ck = torch.load(receipt['checkpoint'], map_location='cpu', weights_only=False)
    assert ck['offset'] == offsets and ck['n_rel'] == 102 and ck['mode'] == 'single'
    model = restore_mixed(ck, 'cpu')
    before = tensor_digest(model.state_dict())
    with torch.inference_mode():
        cosine = cosine_cache(cnorm(model.E[offset:offset+count]))
    bc_recipe, cs_recipe = [json.loads(Path(receipt[k]).read_text()) for k in ('bc_recipe', 'cs_recipe')]
    h, r, t = (np.asarray(train[k]) for k in ('head', 'relation', 'tail'))
    by_rel, arrays, raw_all, norm_all, pipeline_all = {}, {}, [], [], []
    for i, case in enumerate(cases):
        rel, direction = case['relation'], case['direction']
        if rel not in by_rel:
            by_rel[rel] = np.flatnonzero(r == rel)
        rows = by_rel[rel]
        assert np.all(np.asarray(train['head_type'])[rows] == 'drug') and np.all(np.asarray(train['tail_type'])[rows] == 'drug')
        source, positive = case['source_local_id'], case['positive_local_id']
        graph = symmetric_graph(h[rows], t[rows], count)
        known = neighbors(graph, source)
        catalog = old_scores[case['sample_index']]
        unknown = np.ones(count, bool)
        unknown[known] = False
        winner = int(np.flatnonzero(unknown)[np.argmax(catalog[unknown])])
        assert catalog[winner] > catalog[positive]
        case.update(winner_local_id=winner, known_train_positives=int(len(known)))
        graph, removed = pair_removed_graph(h[rows], t[rows], count, source, positive, direction)
        raw = retrieval_features(model, graph, source, case['directed_relation'], offset, cosine)
        np.testing.assert_allclose(raw[0], catalog, atol=1e-5, rtol=1e-5)
        pipeline, normalized = blend(raw, bc_recipe, cs_recipe, rel, direction)
        mr, pr = catalog_rank(raw[0], positive, known), catalog_rank(pipeline, positive, known)
        assert mr == case['filtered_rank']
        allowed = unknown.copy()
        allowed[positive] = True
        best = int(np.flatnonzero(allowed)[np.argmax(pipeline[allowed])])
        case.update(model_rank=mr, pipeline_rank=pr, pipeline_winner_local_id=best,
            focal_edge_copies_removed=removed, pipeline_pair_margin=float(pipeline[positive]-pipeline[winner]),
            feature_pair_values=raw[:, [positive, winner]].tolist(),
            bc_weights=bc_recipe['groups'][f'{rel}/{direction}']['weights'], cs_beta=cs_recipe['groups'][f'{rel}/{direction}']['beta'])
        batch = sample_batch(train, rows, case, i, offset, count)
        assert batch['source'][0] == source+offset and batch['positive'][0] == positive+offset
        for pool in ('random', 'forced'):
            values = batch['negatives_'+pool]-offset
            case[pool+'_pool'] = dict(competitor_draws=int((values == winner).sum()),
                positive_draws=int((values == positive).sum()), known_positive_draws=int(np.isin(values, known).sum()))
        case['pools_identical'] = bool(np.array_equal(batch['negatives_random'], batch['negatives_forced']))
        arrays.update({f'case{i}_{k}': v for k, v in batch.items()})
        arrays[f'case{i}_known'] = known.copy()
        raw_all.append(raw)
        norm_all.append(normalized)
        pipeline_all.append(pipeline)
        log(stage='retrieval_case', case=i+1, cases=len(cases), model_rank=mr, pipeline_rank=pr,
            diagnostic_train_pipeline_mrr=float(np.mean([1/c['pipeline_rank'] for c in cases[:i+1]])))
    assert tensor_digest(model.state_dict()) == before
    write_json(out/'cases.json', cases)
    np.savez_compressed(out/'batches.npz', **arrays)
    np.savez_compressed(out/'features.npz', raw=np.stack(raw_all), normalized=np.stack(norm_all), pipeline=np.stack(pipeline_all))
    del cosine, old_scores, raw_all, norm_all, pipeline_all
    student = model.a.to('cuda').requires_grad_(True)
    student_before = tensor_digest(student.state_dict())
    teachers = []
    log(stage='loading_own_teachers', count=len(receipt['teacher_paths']))
    for seed, path in enumerate(receipt['teacher_paths']):
        meta = torch.load(path, map_location='cpu', weights_only=False)
        assert meta['offset'] == offsets and meta['n_rel'] == 102 and meta['args']['seed'] == seed and not meta['args'].get('distill')
        del meta
        teachers.append(load_teacher(path, sum(counts.values()), 102, torch.device('cuda')))
    teacher_before = [tensor_digest(tm.state_dict()) for tm in teachers]
    records = []
    torch.cuda.reset_peak_memory_stats()
    for i, case in enumerate(cases):
        source, rel, positive = [torch.as_tensor(arrays[f'case{i}_{k}'], device='cuda') for k in ('source', 'relation', 'positive')]
        winner = case['winner_local_id']+offset
        pair_margins = []
        for tm in teachers:
            pair = teacher_logits([tm], source[:1], rel[:1], positive[:1], torch.tensor([winner], device='cuda'))
            pair_margins.append(float(pair[0, 0]-pair[0, 1]))
        for pool in ('random', 'forced'):
            neg = torch.as_tensor(arrays[f'case{i}_negatives_{pool}'], device='cuda')
            batch = source, rel, positive, neg
            target = teacher_logits(teachers, *batch)
            record = dict(case_index=i, sample_index=case['sample_index'], pool=pool,
                **case[pool+'_pool'], pools_identical=case['pools_identical'],
                teacher_pair_margins=pair_margins, teacher_mean_pair_margin=float(np.mean(pair_margins)),
                **inspect_objective(student, batch, target, winner))
            records.append(record)
            with (out/'gradients.jsonl').open('a') as handle:
                handle.write(json.dumps(record, allow_nan=False)+'\n')
            log(stage='gradient_case', case=i+1, cases=len(cases), pool=pool,
                focal_change=record['scopes']['focal']['total']['groups']['representation']['descent_margin_change'],
                batch_change=record['scopes']['batch']['total']['groups']['representation']['descent_margin_change'],
                diagnostic_train_pipeline_mrr=float(np.mean([1/c['pipeline_rank'] for c in cases])))
            del target
    after = tensor_digest(student.state_dict())
    teacher_after = [tensor_digest(tm.state_dict()) for tm in teachers]
    assert after == student_before and teacher_after == teacher_before
    assert all(p.grad is None for tm in [student, *teachers] for p in tm.parameters())
    assert all(not p.requires_grad for tm in teachers for p in tm.parameters())
    summary = summarize(cases, records)
    summary.update(seconds=time.monotonic()-started, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   dataset_files_opened=sorted(opened))
    write_json(out/'summary.json', summary)
    verify_hashes(receipt['hashes'])
    assert sorted(opened) == ['raw/num-node-dict.csv.gz', 'split/random/train.pt']
    names = ('prerun.json', 'cases.json', 'batches.npz', 'features.npz', 'gradients.jsonl', 'summary.json')
    write_json(out/'audit.json', dict(passed=True, source_input_hashes_unchanged=True,
        cpu_model_unchanged=True, student_before=student_before, student_after=after,
        teacher_before=teacher_before, teacher_after=teacher_after, all_parameter_grads_absent=True,
        gradients_train_only=True, optimizer_updates=0, dataset_files_opened=sorted(opened),
        artifact_sha256={name: sha256(out/name) for name in names}))
    log(stage='complete', cases=len(cases), gradient_records=len(records))


if __name__ == '__main__':
    main()
