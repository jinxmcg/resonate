"""H34: frozen single-student relation-aware retrieval, TRAIN/VALID only.

Run from the repository root with ``python -m biokg.relation_analogy``.
Protocol / finite hyperparameter grid: optimizations.md. No training, no
test option, no legacy caches, and no inference-time teacher ensemble.
"""

import argparse
import csv
import gzip
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy import sparse

from biokg.analyze_errors import restore
from biokg.train_biokg_comp import globalize, load
from resonate import cnorm


FEATURES = ("student", "global_max", "global_top3", "relation_max",
            "relation_top3", "jaccard_max", "jaccard_top3")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validation_only(root):
    """The only dataset entry point; do not replace with get_edge_split()."""
    return load(root, include_test=False)


def rank_rows(scores):
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite scores")
    greater = (scores[:, 1:] > scores[:, :1]).sum(1)
    greater_equal = (scores[:, 1:] >= scores[:, :1]).sum(1)
    return 1.0 + (greater + greater_equal) / 2.0


def stats(scores):
    rank = rank_rows(scores)
    return dict(mrr=float((1 / rank).mean()), hits1=float((rank <= 1).mean()),
                hits10=float((rank <= 10).mean()),
                tied_rows=int((scores[:, 1:] == scores[:, :1]).any(1).sum()))


def holder_pool(similarities, holders, source, candidates):
    """Exact max/top3 for several metrics using the same ragged holder lists.

    similarities: (metrics, sources). holders: CSC source->candidate graph.
    Exclude the query source. Empty candidates use -1. Duplicate candidates
    get identical scores. The ragged reduction avoids a Python candidate loop.
    """
    count = np.diff(holders.indptr)[candidates]
    keep = np.flatnonzero(count)
    out_max = np.full((len(similarities), len(candidates)), -1, np.float32)
    out_top3 = out_max.copy()
    if not len(keep):
        return out_max, out_top3
    lengths = count[keep]
    start = np.r_[0, np.cumsum(lengths)[:-1]]
    owner = np.repeat(np.arange(len(keep)), lengths)
    position = np.arange(lengths.sum())
    graph_pos = (np.repeat(holders.indptr[candidates[keep]], lengths)
                 + position - np.repeat(start, lengths))
    hs = holders.indices[graph_pos]
    values = similarities[:, hs].copy()
    values[:, hs == source] = -np.inf
    maxima, total = None, np.zeros((len(similarities), len(keep)), np.float32)
    used = np.zeros_like(total, dtype=np.int32)
    for _ in range(3):
        best = np.maximum.reduceat(values, start, axis=1)
        good = np.isfinite(best)
        if maxima is None:
            maxima = np.where(good, best, -1)
        total += np.where(good, best, 0)
        used += good
        # Remove exactly one maximizing holder per candidate, even on ties.
        first = np.minimum.reduceat(
            np.where(values == best[:, owner], position[None, :], len(position)),
            start, axis=1)
        metric, group = np.nonzero(good)
        values[metric, first[metric, group]] = -np.inf
    out_max[:, keep] = maxima
    out_top3[:, keep] = np.where(used > 0, total / np.maximum(used, 1), -1)
    return out_max, out_top3


@torch.inference_mode()
def embedding_table(model, ids, relation=None, chunk=2048):
    dev = next(model.parameters()).device
    parts = []
    for start in range(0, len(ids), chunk):
        x = model.embed(torch.as_tensor(ids[start:start + chunk], device=dev))
        if relation is not None:
            r = torch.full((len(x),), relation, device=dev, dtype=torch.long)
            x = cnorm(model.hop(x, r))
        parts.append(x)
    return torch.cat(parts)


@torch.inference_mode()
def candidate_scores(model, sources, relation, candidates):
    dev = next(model.parameters()).device
    source = torch.as_tensor(sources, device=dev)
    target = torch.as_tensor(candidates, device=dev)
    rel = torch.full((len(source),), relation, device=dev, dtype=torch.long)
    query = model.out(model.hop(model.embed(source), rel), rel)
    scores = torch.einsum("bm,bcm->bc", query, model.rows(target).conj()).real
    scores = scores * model.log_tau.exp()
    if model.b is not None:
        scores = scores + model.b[target]
    return scores.float().cpu().numpy()


@torch.inference_mode()
def build_features(model, train, part, offset, n_ent, n_rel, chunk=32):
    h, r, t = globalize(part, offset)
    th, tr, tt = globalize(train, offset)
    n = len(r)
    result = np.empty((len(FEATURES), 2 * n, 501), np.float32)
    audits = []
    started, last_progress, processed = time.monotonic(), 0.0, 0
    for direction in (0, 1):
        train_source, train_target = (th, tt) if direction == 0 else (tt, th)
        query_source, query_target = (h, t) if direction == 0 else (t, h)
        target_type = part["tail_type"] if direction == 0 else part["head_type"]
        neg = part["tail_neg"] if direction == 0 else part["head_neg"]
        for relation in np.unique(r):
            rows = np.flatnonzero(r == relation)
            train_mask = tr == relation
            sources = np.unique(np.r_[train_source[train_mask], query_source[rows]])
            local_train = np.searchsorted(sources, train_source[train_mask])
            adjacency = sparse.csr_matrix(
                (np.ones(len(local_train), np.float32),
                 (local_train, train_target[train_mask])),
                shape=(len(sources), n_ent))
            duplicate_edges = int(train_mask.sum() - adjacency.nnz)
            adjacency.data.fill(1)  # set semantics, not edge multiplicities
            holders = adjacency.tocsc()
            degree = np.diff(adjacency.indptr).astype(np.float32)
            directed = int(relation + direction * (n_rel // 2))
            global_e = embedding_table(model, sources)
            relation_e = embedding_table(model, sources, directed)
            cosine_delta_sum, cosine_count, cosine_delta_max = 0.0, 0, 0.0
            for start in range(0, len(rows), chunk):
                indices = rows[start:start + chunk]
                local_query = np.searchsorted(sources, query_source[indices])
                dev_index = torch.as_tensor(local_query, device=global_e.device)
                sim_global = (global_e[dev_index] @ global_e.conj().T).real.cpu().numpy()
                sim_relation = (relation_e[dev_index] @ relation_e.conj().T).real.cpu().numpy()
                delta = np.abs(sim_global - sim_relation)
                cosine_delta_sum += float(delta.sum(dtype=np.float64))
                cosine_count += delta.size
                cosine_delta_max = max(cosine_delta_max, float(delta.max()))
                inter = (adjacency[local_query] @ adjacency.T).toarray()
                union = degree[local_query, None] + degree[None, :] - inter
                jac = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
                candidate_offset = np.array([offset[x] for x in target_type[indices]])
                candidates = np.concatenate(
                    [query_target[indices, None], neg[indices] + candidate_offset[:, None]], axis=1)
                output_rows = direction * n + indices
                result[0, output_rows] = candidate_scores(
                    model, query_source[indices], directed, candidates)
                for j, output_row in enumerate(output_rows):
                    similarities = np.stack([sim_global[j], sim_relation[j], jac[j]])
                    mx, top = holder_pool(similarities, holders, local_query[j], candidates[j])
                    # Legacy Jaccard leaves every candidate missing for a source
                    # with no TRAIN neighbours, rather than assigning zero overlap.
                    if degree[local_query[j]] == 0:
                        mx[2] = top[2] = -1
                    result[1, output_row], result[2, output_row] = mx[0] ** 3, top[0] ** 3
                    result[3, output_row], result[4, output_row] = mx[1] ** 3, top[1] ** 3
                    result[5, output_row], result[6, output_row] = mx[2], top[2]
                processed += len(indices)
                elapsed = time.monotonic() - started
                if elapsed - last_progress >= 45:
                    print(f"H34 {processed}/{2*n} queries, elapsed={elapsed:.0f}s, "
                          "LR=n/a (frozen inference); paired MRR pending", flush=True)
                    last_progress = elapsed
            operator_defect = None
            if model.block and not getattr(model, "tied_reverse", False):
                op = model.H[directed]
                gram = op.conj().transpose(-1, -2) @ op
                eye = torch.eye(op.shape[-1], device=op.device)
                operator_defect = float((gram - eye).abs().square().mean().sqrt())
            audits.append(dict(relation=int(relation), direction=direction,
                               queries=len(rows), duplicate_train_edges=duplicate_edges,
                               cosine_abs_delta_mean=cosine_delta_sum / cosine_count,
                               cosine_abs_delta_max=cosine_delta_max,
                               operator_gram_rms_from_identity=operator_defect))
            print(f"dir={direction} rel={relation} queries={len(rows)} "
                  f"cosine_change={cosine_delta_sum / cosine_count:.5f}", flush=True)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite features")
    return result, audits


def normalize(x):
    return (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-6)


def weight_grid():
    candidates = [("uniform", np.full(5, .2, np.float32))]
    for model_weight in (.5, .75, .9, 1.):
        for analogy_share in (0., .5, 1.):
            analogy = (1 - model_weight) * analogy_share / 2
            jac = (1 - model_weight) * (1 - analogy_share) / 2
            weights = np.array([model_weight, analogy, analogy, jac, jac], np.float32)
            if not any(np.array_equal(weights, w) for _, w in candidates):
                candidates.append((f"model{model_weight}_analogy{analogy_share}", weights))
    return candidates


def combine(features, weights):
    out = np.zeros(features.shape[1:], np.float32)
    for feature, weight in zip(features, weights):
        if weight:
            out += feature * weight
    return out


def choose_mixture(features, fit):
    best_score, best_label, best_weights = -1., None, None
    for label, weights in weight_grid():
        score = float((1 / rank_rows(combine(features[:, fit], weights))).mean())
        if score > best_score:
            best_score, best_label, best_weights = score, label, weights
    return dict(label=best_label, weights=best_weights.tolist(), fit_mrr=best_score), \
        combine(features, best_weights)


def bootstrap_delta(delta, report_triples, seed=3402, replicates=1000):
    n = len(delta) // 2
    paired = ((delta[:n] + delta[n:]) / 2)[report_triples]
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(paired, size=len(paired), replace=True).mean()
                      for _ in range(replicates)])
    return np.quantile(means, [.025, .975]).tolist()


def evaluate(features, relations, families):
    n = len(relations)
    fit_tri = np.random.default_rng(3401).random(n) < .5
    if not fit_tri.any() or fit_tri.all():
        raise ValueError("Screen too small for a paired validation split")
    fit = np.tile(fit_tri, 2)
    control = normalize(features[[0, 1, 2, 5, 6]])
    treatment = normalize(features[[0, 3, 4, 5, 6]])
    control_recipe, baseline = choose_mixture(control, fit)
    treatment_recipe, changed = choose_mixture(treatment, fit)
    rr_baseline, rr_changed = 1 / rank_rows(baseline), 1 / rank_rows(changed)
    delta = rr_changed - rr_baseline
    interval = bootstrap_delta(delta, ~fit_tri)
    held_delta = float(delta[~fit].mean())
    uniform = np.full(5, .2, np.float32)
    report = dict(
        screen_only=True, test_loaded=False, training_performed=False,
        fit_triples=int(fit_tri.sum()), report_triples=int((~fit_tri).sum()),
        model_full_screen=stats(features[0]),
        control=dict(recipe=control_recipe, held_out=stats(baseline[~fit])),
        relation=dict(recipe=treatment_recipe, held_out=stats(changed[~fit])),
        held_out_mrr_delta=held_delta, paired_triple_bootstrap_95=interval,
        advance_gate=bool(held_delta >= .001 and interval[0] > 0),
        uniform_control=stats(combine(control[:, ~fit], uniform)),
        uniform_relation=stats(combine(treatment[:, ~fit], uniform)),
        per_feature_held_out={name: stats(feature[~fit])
                              for name, feature in zip(FEATURES, features)},
        family_direction=[])
    rel_rows = np.tile(relations, 2)
    dirs = np.repeat([0, 1], n)
    family = np.array([families[int(r)] for r in rel_rows])
    for fam in np.unique(family):
        for d in (0, 1):
            mask = (~fit) & (family == fam) & (dirs == d)
            if mask.any():
                report["family_direction"].append(dict(
                    family=fam, direction=d, queries=int(mask.sum()),
                    control_mrr=float(rr_baseline[mask].mean()),
                    relation_mrr=float(rr_changed[mask].mean()),
                    delta=float(delta[mask].mean())))
    return report, dict(fit=fit, control_rr=rr_baseline, relation_rr=rr_changed,
                        model_rank=rank_rows(features[0]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--triples", type=int, default=5000)
    p.add_argument("--chunk", type=int, default=32)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    if args.triples < 10 or args.chunk < 1 or args.threads < 1:
        p.error("Need >=10 triples, positive chunk and thread count")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    root = Path(args.data_root) / "ogbl_biokg"
    provenance = dict(args=vars(args), torch_version=torch.__version__,
                      numpy_version=np.__version__, feature_names=FEATURES,
                      checkpoint_sha256=sha256(args.model),
                      input_hashes={name: sha256(root / name) for name in (
                          "split/random/train.pt", "split/random/valid.pt",
                          "processed/data_processed", "mapping/relidx2relname.csv.gz")},
                      source_sha256=sha256(__file__),
                      protocol_sha256=sha256(Path(__file__).parents[1] / "optimizations.md"))
    (out / "prerun.json").write_text(json.dumps(provenance, indent=2) + "\n")
    split, offset, n_ent, _, _ = validation_only(args.data_root)
    n_rel = 2 * (int(split["train"]["relation"].max()) + 1)
    model, config = restore(args.model, n_ent, n_rel, offset, torch.device(args.device))
    if config.get("low_rank", 0) or config.get("rel_gain", False):
        raise ValueError("H34 preregistration requires the plain block student")
    if config.get("seed") != 0 or config.get("distill_T") != 2.0:
        raise ValueError("H34 screen requires released seed-0 T=2 student")
    part = split["valid"]
    selected = np.sort(np.random.default_rng(3400).choice(
        len(part["head"]), min(args.triples, len(part["head"])), replace=False))
    part = {key: np.asarray(value)[selected] for key, value in part.items()}
    np.save(out / "valid_indices.npy", selected)
    print(f"H34: {len(selected)} triples / {2*len(selected)} queries; "
          f"params={model.n_params():,}; device={args.device}; frozen T=2 seed=0", flush=True)
    started = time.monotonic()
    features, audits = build_features(model, split["train"], part, offset,
                                      n_ent, n_rel, args.chunk)
    np.save(out / "features.npy", features)
    with gzip.open(root / "mapping/relidx2relname.csv.gz", "rt") as f:
        reader = csv.reader(f)
        next(reader)
        families = {int(i): name.split("_")[0] for i, name in reader}
    report, queries = evaluate(features, part["relation"], families)
    report.update(elapsed_seconds=time.monotonic() - started, metric_audit=audits,
                  checkpoint_sha256=provenance["checkpoint_sha256"])
    np.savez_compressed(out / "queries.npz", valid_index=np.tile(selected, 2),
                        relation=np.tile(part["relation"], 2), **queries)
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in (
        "control", "relation", "held_out_mrr_delta", "paired_triple_bootstrap_95",
        "advance_gate", "elapsed_seconds")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
