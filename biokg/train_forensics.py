"""TF1: inspect a TRAIN near miss without changing the model or scoring recipe."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import sparse

from resonate import cnorm
from biokg.confirm_feature_blend import CHECKPOINT_SHA, DATASET, verify_hashes, write_json
from biokg.gradient_diagnostic import check_dataset_path, sha256, train_only
from biokg.mixed_operator import restore_mixed
from biokg.train_joint_operator import parameter_hashes


def catalog_rank(scores, target, known):
    scores = np.asarray(scores)
    keep = np.ones(len(scores), bool)
    keep[np.asarray(known, np.int64)] = False
    keep[int(target)] = False
    competitors = scores[keep]
    return float(1+((competitors > scores[target]).sum()+(competitors >= scores[target]).sum())/2)


def choose_case(records):
    for low, high in ((1, 10), (1, np.inf)):
        for row in records:
            if low < row['filtered_rank'] <= high:
                return row
    return None


def symmetric_graph(head, tail, count):
    h, t = np.asarray(head), np.asarray(tail)
    graph = sparse.csr_matrix((np.ones(2*len(h), np.float32), (np.r_[h, t], np.r_[t, h])), shape=(count, count))
    graph.sum_duplicates()
    graph.data.fill(1)
    graph.sort_indices()
    return graph


def neighbors(graph, node, exclude_pair=None):
    result = graph.indices[graph.indptr[node]:graph.indptr[node+1]]
    if exclude_pair is not None:
        a, b = exclude_pair
        if node == a:
            result = result[result != b]
        if node == b:
            result = result[result != a]
    return result


def margin_parts(norms, cosines, scale):
    """Exact symmetric decomposition of winner-minus-positive score."""
    p, w = map(float, norms)
    cp, cw = map(float, cosines)
    return dict(norm=float(scale*(cw+cp)*.5*(w-p)),
                alignment=float(scale*(w+p)*.5*(cw-cp)))


@torch.inference_mode()
def geometry(model, source, relation, targets):
    q = model.a.query(torch.tensor([source]), torch.tensor([relation]), model.a.H_b)[0].to(torch.complex128)
    e = model.E[torch.as_tensor(targets)].to(torch.complex128)
    tau = float(model.a.log_tau.exp())
    qnorm, norms = q.abs().square().sum().sqrt(), e.abs().square().sum(-1).sqrt()
    dot = (q[None, :]*e.conj()).sum(-1).real
    cosines = dot/(qnorm*norms)
    block = model.a.H_b.shape[-1]
    contributions = (q[None, :]*e.conj()).real.reshape(len(e), -1, block).sum(-1)*tau
    api = model.candidate_outputs(torch.tensor([source]), torch.tensor([relation]), torch.as_tensor(targets)[None, :])[0][0].numpy()
    np.testing.assert_allclose(api, (dot*tau).numpy(), atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(api, contributions.sum(-1).numpy(), atol=2e-5, rtol=2e-5)
    parts = margin_parts(norms.numpy(), cosines.numpy(), float(tau*qnorm))
    np.testing.assert_allclose(sum(parts.values()), float((dot[1]-dot[0])*tau), atol=1e-10)
    unit_source = cnorm(model.E[torch.tensor([source])])[0].to(torch.complex128)
    pre_operator_cosine = ((unit_source[None, :]*e.conj()).sum(-1).real/norms).numpy()
    inverse = (relation+len(model.a.H_b)//2) % len(model.a.H_b)
    reverse = model.candidate_outputs(torch.as_tensor(targets), torch.full((len(targets),), inverse), torch.full((len(targets), 1), source))[0][:, 0].numpy()
    return dict(scores=api.tolist(), target_norms=norms.tolist(), query_norm=float(qnorm),
        temperature=tau, normalized_alignment=cosines.tolist(), pre_operator_alignment=pre_operator_cosine.tolist(),
        reverse_scores=reverse.tolist(), winner_minus_positive=float(api[1]-api[0]),
        symmetric_margin_decomposition=parts, block_size=block, blocks=contributions.shape[1],
        block_contributions=contributions.tolist(),
        block_winner_minus_positive=(contributions[1]-contributions[0]).tolist(),
        direct_score_reconstruction_passed=True)


def best_support(values, ids, limit=3):
    order = np.argsort(-values, kind='stable')[:limit]
    return [dict(drug_local_id=int(ids[i]), similarity=float(values[i])) for i in order]


@torch.inference_mode()
def structural_evidence(model, graph, source, positive, candidate, relation, offset):
    pair = (source, positive)
    source_neighbors = neighbors(graph, source, pair)
    holders = neighbors(graph, candidate, pair)
    unit = cnorm(model.E[offset:offset+graph.shape[0]]).numpy()
    cosine = (unit[holders]*unit[source].conj()).sum(-1).real
    candidate_cosine = (unit[source_neighbors]*unit[candidate].conj()).sum(-1).real
    transformed = model.a.query(torch.as_tensor(np.r_[source, holders]+offset),
        torch.full((1+len(holders),), relation), model.a.H_b).numpy()
    relation_cosine = (transformed[1:]*transformed[:1].conj()).sum(-1).real
    jaccards = []
    for holder in holders:
        other = neighbors(graph, int(holder), pair)
        common = np.intersect1d(source_neighbors, other).size
        union = len(source_neighbors)+len(other)-common
        jaccards.append(common/union if union else 0.)
    return dict(focal_pair_removed=True, source_neighbors=len(source_neighbors), candidate_holders=len(holders),
        shared_source_candidate_neighbors=np.intersect1d(source_neighbors, holders).tolist(),
        holder_cosine_top3=best_support(cosine, holders),
        holder_relation_cosine_top3=best_support(relation_cosine, holders),
        holder_jaccard_top3=best_support(np.asarray(jaccards), holders),
        candidate_analogy_top3=best_support(candidate_cosine, source_neighbors))


@torch.inference_mode()
def inspect(model, train, offset, counts, log=lambda **kw: None):
    drug = (np.asarray(train['head_type']) == 'drug') & (np.asarray(train['tail_type']) == 'drug')
    ddi = np.flatnonzero(drug)
    selected = np.random.default_rng(3681).choice(ddi, min(128, len(ddi)), replace=False)
    h, r, t = (np.asarray(train[k]) for k in ('head', 'relation', 'tail'))
    count, start = counts['drug'], offset['drug']
    candidates = torch.arange(start, start+count)[None, :]
    records, graphs, all_scores = [], {}, []
    for direction in (0, 1):
        for row in selected:
            relation = int(r[row])
            if relation not in graphs:
                edge = drug & (r == relation)
                graphs[relation] = symmetric_graph(h[edge], t[edge], count)
            source, positive = (int(h[row]), int(t[row])) if direction == 0 else (int(t[row]), int(h[row]))
            directed = relation+51*direction
            scores = model.candidate_outputs(torch.tensor([source+start]), torch.tensor([directed]), candidates)[0][0].numpy()
            assert np.isfinite(scores).all()
            known = neighbors(graphs[relation], source)
            assert positive in known
            record = dict(sample_index=len(records), train_row=int(row), direction=direction, relation=relation,
                directed_relation=directed, source_local_id=source, positive_local_id=positive,
                raw_rank=catalog_rank(scores, positive, []), filtered_rank=catalog_rank(scores, positive, known),
                known_train_positives=len(known), raw_winner_local_id=int(scores.argmax()),
                raw_winner_known_positive=bool(int(scores.argmax()) in known))
            records.append(record)
            all_scores.append(scores)
        log(stage='catalog_direction_complete', direction=direction, queries=len(records),
            diagnostic_train_mrr=float(np.mean([1/x['filtered_rank'] for x in records])))
    chosen = choose_case(records)
    if chosen is None:
        return dict(records=records, case=None, sample_seed=3681), np.stack(all_scores)
    case = dict(chosen)
    source, positive, relation = case['source_local_id'], case['positive_local_id'], case['relation']
    scores, graph = all_scores[case['sample_index']], graphs[relation]
    known = neighbors(graph, source)
    mask = np.ones(count, bool)
    mask[known] = False
    mask[positive] = True
    ids = np.flatnonzero(mask)
    order = ids[np.argsort(-scores[ids], kind='stable')]
    winner = int(order[0])
    assert winner != positive and winner not in known
    norms = model.E[start:start+count].abs().square().sum(-1).sqrt().numpy()
    case.update(winner_local_id=winner, drug_offset=start, drug_count=count,
        filtered_top10=[dict(drug_local_id=int(i), score=float(scores[i]), norm=float(norms[i]),
                            is_positive=bool(i == positive), known_train_positive=bool(i in known)) for i in order[:10]],
        unit_target_norm_filtered_rank=catalog_rank(scores/norms, positive, known),
        target_norm_percentile=[float((norms <= norms[i]).mean()*100) for i in (positive, winner)],
        geometry=geometry(model, source+start, case['directed_relation'], [positive+start, winner+start]),
        train_evidence={name: structural_evidence(model, graph, source, positive, candidate, case['directed_relation'], start)
                        for name, candidate in (('positive', positive), ('winner', winner))})
    # Independent scalar rank replay for the selected example.
    expected = 1+sum(float(scores[i] > scores[positive])+.5*float(scores[i] == scores[positive])
                     for i in ids if i != positive)
    assert expected == case['filtered_rank']
    return dict(records=records, case=case, sample_seed=3681), np.stack(all_scores)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='biokg/results/train_forensics/s0')
    args = parser.parse_args()
    torch.set_num_threads(4)
    repo, out = Path(__file__).resolve().parents[1], Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    checkpoint = repo/'biokg/results/h35f/campaign_s0/single.pt'
    prior = json.loads((checkpoint.parent/'prerun.json').read_text())
    train_path = DATASET/'split/random/train.pt'
    hashes = {str(checkpoint): CHECKPOINT_SHA, str(train_path): prior['input_sha256'][str(train_path)]}
    paths = [DATASET/'raw/num-node-dict.csv.gz', Path(__file__), repo/'biokg/test_train_forensics.py',
             repo/'biokg/TRAIN_FORENSICS.md', checkpoint.parent/'prerun.json']
    paths += [repo/name for name in ('resonate.py', 'biokg/mixed_operator.py', 'biokg/joint_operator.py',
        'biokg/dual_operator.py', 'biokg/train_dual_operator.py', 'biokg/gradient_diagnostic.py',
        'biokg/train_joint_operator.py', 'biokg/train_biokg_comp.py', 'biokg/confirm_feature_blend.py')]
    hashes.update({str(p): sha256(p) for p in paths})
    opened = set()
    def guard(event, arguments):
        if event == 'open' and isinstance(arguments[0], (str, bytes)):
            path = Path(arguments[0].decode() if isinstance(arguments[0], bytes) else arguments[0]).resolve()
            relative = check_dataset_path(path, DATASET)
            if relative:
                opened.add(relative)
            if path.suffix in ('.pt', '.pth', '.pkl', '.npy', '.npz', '.safetensors', '.bin'):
                if str(path) not in hashes and not path.is_relative_to(out):
                    raise PermissionError(f'Unapproved artifact: {path}')
    sys.addaudithook(guard)
    started = time.monotonic()
    def log(**event):
        event.update(seconds=time.monotonic()-started, lr=0.)
        print(json.dumps(event), flush=True)
    verify_hashes(hashes)
    write_json(out/'prerun.json', dict(protocol='TF1', sample_seed=3681, train_rows=128,
        hashes=hashes, device='cpu', lr=0., model_updates=0, valid_loaded=False, test_loaded=False))
    train, offset, counts = train_only(DATASET)
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert ck['offset'] == offset and ck['n_rel'] == 102 and ck['mode'] == 'single'
    model = restore_mixed(ck, 'cpu')
    before = parameter_hashes(model)
    log(stage='train_catalog_scoring_start')
    result, scores = inspect(model, train, offset, counts, log)
    result.update(protocol='TF1', full_catalog_train_diagnostic=True, submission_pipeline_evaluated=False,
        not_generalization_mrr=True, lr=0., model_updates=0, valid_loaded=False, test_loaded=False,
        no_external_trained_artifacts=True, no_submission_change=True)
    write_json(out/'findings.json', result)
    np.save(out/'catalog_scores.npy', scores, allow_pickle=False)
    after = parameter_hashes(model)
    assert before == after and all(p.grad is None and not p.requires_grad for p in model.parameters())
    verify_hashes(hashes)
    assert sorted(opened) == ['raw/num-node-dict.csv.gz', 'split/random/train.pt']
    write_json(out/'audit.json', dict(passed=True, source_input_hashes_unchanged=True,
        model_before=before, model_after=after, gradients_absent=True, dataset_files_opened=sorted(opened),
        candidate_identity_checks_passed=True, selected_rank_and_score_reconstruction_passed=True,
        artifact_sha256={n: sha256(out/n) for n in ('prerun.json', 'findings.json', 'catalog_scores.npy')},
        seconds=time.monotonic()-started))
    log(stage='complete', case_found=result['case'] is not None)


if __name__ == '__main__':
    main()
