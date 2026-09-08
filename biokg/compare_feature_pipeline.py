"""FC1 frozen original/improved A feature comparison; no TEST or training option."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import sparse

from resonate import cnorm
from biokg.gradient_diagnostic import sha256
from biokg.mixed_operator import from_student, restore_mixed
from biokg.relation_analogy import holder_pool, normalize, rank_rows
from biokg.train_biokg_comp import globalize
from biokg.train_candidate_focus import check_data_path, train_valid
from biokg.train_joint_operator import parameter_hashes


CHANNELS = ("original_score", "original_analogy_max", "original_analogy_top3",
            "improved_score", "improved_analogy_max", "improved_analogy_top3",
            "jaccard_max", "jaccard_top3")
VIEWS = dict(original=(0, 1, 2, 6, 7), improved=(3, 4, 5, 6, 7),
             score_only_swap=(3, 1, 2, 6, 7), analogy_only_swap=(0, 4, 5, 6, 7))
MIN_ROWS = 2000


def paired_masks(n):
    fit = np.random.default_rng(0).random(n) < .5
    if not fit.any() or fit.all():
        raise ValueError("Need nonempty fit and report halves")
    return np.tile(fit, 2)


def graph_for(train_source, train_target, query_source, n_ent):
    sources = np.unique(np.r_[train_source, query_source])
    local = np.searchsorted(sources, train_source)
    graph = sparse.csr_matrix((np.ones(len(local), np.float32), (local, train_target)),
                              shape=(len(sources), n_ent))
    duplicates = len(local) - graph.nnz
    graph.data.fill(1)
    return sources, graph, duplicates


@torch.inference_mode()
def score_batch(models, tables, graph, holders, degree, local_query, sources, relation, candidates):
    """No positive labels/column assumptions: scores every candidate identically."""
    dev = models[0].E.device
    src = torch.as_tensor(sources[local_query], device=dev)
    rel = torch.full((len(src),), relation, device=dev, dtype=torch.long)
    cand = torch.as_tensor(candidates, device=dev)
    idx = torch.as_tensor(local_query, device=dev)
    similarities = [(table[idx] @ table.conj().T).real.cpu().numpy() for table in tables]
    inter = (graph[local_query] @ graph.T).toarray()
    union = degree[local_query, None] + degree[None, :] - inter
    jac = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    out = np.empty((8, len(src), candidates.shape[1]), np.float32)
    for i, model in enumerate(models):
        out[3 * i] = model.candidate_outputs(src, rel, cand)[0].float().cpu().numpy()
    for j, source in enumerate(local_query):
        sims = np.stack([similarities[0][j], similarities[1][j], jac[j]])
        mx, top = holder_pool(sims, holders, source, candidates[j])
        if degree[source] == 0:
            mx[2] = top[2] = -1
        out[1, j], out[2, j] = mx[0] ** 3, top[0] ** 3
        out[4, j], out[5, j] = mx[1] ** 3, top[1] ** 3
        out[6, j], out[7, j] = mx[2], top[2]
    if not np.isfinite(out).all():
        raise ValueError("Nonfinite features")
    return out


@torch.inference_mode()
def build_features(models, train, valid, offset, n_ent, n_rel, output, log, chunk=32):
    h, r, t = globalize(valid, offset)
    th, tr, tt = globalize(train, offset)
    n, processed, last = len(r), 0, time.monotonic()
    seen = np.zeros(2 * n, bool)
    totals = np.zeros(2, np.float64)
    audits = []
    started = time.monotonic()
    for direction in (0, 1):
        ts, td = (th, tt) if direction == 0 else (tt, th)
        qs, qt = (h, t) if direction == 0 else (t, h)
        types = np.asarray(valid["tail_type" if direction == 0 else "head_type"])
        negatives = valid["tail_neg" if direction == 0 else "head_neg"]
        for relation in np.unique(r):
            rows = np.flatnonzero(r == relation)
            mask = tr == relation
            sources, graph, duplicates = graph_for(ts[mask], td[mask], qs[rows], n_ent)
            holders, degree = graph.tocsc(), np.diff(graph.indptr).astype(np.float32)
            tables = [cnorm(model.E[torch.as_tensor(sources, device=model.E.device)]) for model in models]
            directed = int(relation + direction * (n_rel // 2))
            for start in range(0, len(rows), chunk):
                indices = rows[start:start + chunk]
                local = np.searchsorted(sources, qs[indices])
                offs = np.array([offset[name] for name in types[indices]])
                candidates = np.concatenate([qt[indices, None], negatives[indices] + offs[:, None]], axis=1)
                values = score_batch(models, tables, graph, holders, degree, local, sources, directed, candidates)
                if start == 0:
                    order = np.arange(candidates.shape[1])[::-1].copy()
                    permuted = score_batch(models, tables, graph, holders, degree, local, sources,
                                           directed, candidates[:, order].copy())
                    np.testing.assert_allclose(permuted, values[:, :, order], atol=1e-6, rtol=1e-6)
                out_rows = direction * n + indices
                assert not seen[out_rows].any()
                output[:, out_rows] = values
                seen[out_rows] = True
                totals += [(1 / rank_rows(values[channel])).sum() for channel in (0, 3)]
                processed += len(indices)
                now = time.monotonic()
                if now - last >= 35:
                    log(stage="features", queries=processed, total=2*n, seconds=now-started,
                        lr=0.0, lr_status="not applicable: frozen inference",
                        partial_model_mrr=dict(zip(("original", "improved"), (totals / processed).tolist())),
                        partial_warning="relation-ordered partial queries, not the final comparison")
                    last = now
            audits.append(dict(relation=int(relation), direction=direction, queries=len(rows),
                               train_unique_edges=int(graph.nnz), duplicate_edges_removed=int(duplicates),
                               candidate_permutation_passed=True))
            del tables
    assert seen.all()
    if hasattr(output, "flush"):
        output.flush()
    log(stage="features_complete", queries=processed, seconds=time.monotonic()-started,
        model_mrr=dict(zip(("original", "improved"), (totals / processed).tolist())), lr=0.0)
    return audits


def mixture_ranks(features, weights, rows, chunk=2048):
    rows = np.asarray(rows, np.int64)
    result = np.empty(len(rows), np.float64)
    weights = np.asarray(weights, np.float32)
    if len(features) != len(weights) or (weights < 0).any() or not np.isclose(weights.sum(), 1):
        raise ValueError("Invalid convex weights")
    for start in range(0, len(rows), chunk):
        ids = rows[start:start + chunk]
        scores = np.zeros((len(ids), features[0].shape[1]), np.float32)
        for feature, weight in zip(features, weights):
            if weight:
                scores += np.asarray(feature[ids], np.float32) * weight
        result[start:start + len(ids)] = rank_rows(scores)
    return result


def candidate_weights(member_mrr, local=False, inherited=()):
    m = len(member_mrr)
    candidates = [("uniform", np.full(m, 1/m, np.float32))]
    candidates.extend((label, np.asarray(weight, np.float32)) for label, weight in inherited)
    order = np.argsort(-np.asarray(member_mrr), kind="stable")
    for top in ((1, 3) if local else (3,)):
        top = min(top, m)
        weights = np.zeros(m, np.float32)
        weights[order[:top]] = 1/top
        candidates.append((f"top{top}", weights))
    for eta in (20, 50, 100):
        weights = np.exp(eta * (member_mrr - np.max(member_mrr)))
        candidates.append((f"soft{eta}", (weights / weights.sum()).astype(np.float32)))
    unique = []
    for label, weights in candidates:
        if not any(np.array_equal(weights, old) for _, old in unique):
            unique.append((label, weights))
    return unique


def fit_recipe(features, fit, relations, family, direction, min_rows=MIN_ROWS, log=lambda **x: None):
    """Only fit rows influence candidate generation or choice."""
    member_rr = np.empty((len(features), len(fit)), np.float64)
    member_rr.fill(np.nan)
    ids = np.flatnonzero(fit)
    if not len(ids):
        raise ValueError("No fit queries")
    for j in range(len(features)):
        unit = np.eye(len(features), dtype=np.float32)[j]
        member_rr[j, ids] = 1 / mixture_ranks(features, unit, ids)

    def choose(rows, local=False, inherited=()):
        per_member = member_rr[:, rows].mean(1)
        best = None
        for label, weights in candidate_weights(per_member, local, inherited):
            mrr = float((1 / mixture_ranks(features, weights, rows)).mean())
            if best is None or mrr > best["fit_mrr"]:
                best = dict(label=label, weights=weights.tolist(), fit_mrr=mrr, fit_queries=len(rows))
        return best

    global_choice = choose(ids)
    log(stage="global_recipe", **global_choice)
    families = {}
    for name in np.unique(family):
        for d in (0, 1):
            rows = np.flatnonzero(fit & (family == name) & (direction == d))
            if len(rows) >= min_rows:
                families[f"{name}/{d}"] = choose(rows, True, [("global", global_choice["weights"])])
                log(stage="family_recipe", family=str(name), direction=d)
    groups = {}
    for relation in np.unique(relations):
        for d in (0, 1):
            mask = (relations == relation) & (direction == d)
            if not mask.any():
                continue
            names = np.unique(family[mask])
            assert len(names) == 1
            key = f"{names[0]}/{d}"
            fallback = families.get(key, global_choice)
            rows = np.flatnonzero(fit & mask)
            if len(rows) >= min_rows:
                selected = choose(rows, True, [("global", global_choice["weights"]),
                                                ("family", fallback["weights"])])
                level = "relation"
            else:
                selected = fallback
                level = "family" if key in families else "global"
            groups[f"{relation}/{d}"] = dict(**selected, level=level, group_fit_queries=len(rows))
    return dict(global_choice=global_choice, families=families, groups=groups, min_rows=min_rows)


def apply_recipe(features, recipe, relations, direction):
    ranks = np.full(len(relations), np.nan)
    for relation in np.unique(relations):
        for d in (0, 1):
            rows = np.flatnonzero((relations == relation) & (direction == d))
            if len(rows):
                ranks[rows] = mixture_ranks(features, recipe["groups"][f"{relation}/{d}"]["weights"], rows)
    assert np.isfinite(ranks).all()
    return ranks


def metrics(rank):
    return dict(mrr=float((1 / rank).mean()), hits1=float((rank == 1).mean()),
                hits10=float((rank <= 10).mean()), queries=len(rank))


def delta_summary(treatment, control, fit, seed=3611):
    n = len(fit) // 2
    assert np.array_equal(fit[:n], fit[n:])
    delta = 1 / treatment - 1 / control
    paired = ((delta[:n] + delta[n:]) / 2)[~fit[:n]]
    rng = np.random.default_rng(seed)
    means = [rng.choice(paired, len(paired), replace=True).mean() for _ in range(1000)]
    out = ~fit
    return dict(delta_mrr=float(delta[out].mean()),
                bootstrap_95=np.quantile(means, [.025, .975]).tolist(),
                recovered_top1=int(((treatment == 1) & (control > 1) & out).sum()),
                lost_top1=int(((treatment > 1) & (control == 1) & out).sum()))


def summarize(ranks, fit, family, direction):
    result = dict(protocol="FC1", training_performed=False, test_loaded=False,
                  fit_triples=int(fit[:len(fit)//2].sum()), report_triples=int((~fit[:len(fit)//2]).sum()),
                  metrics={key: dict(fit=metrics(value[fit]), report=metrics(value[~fit]))
                           for key, value in ranks.items()})
    primary = delta_summary(ranks["improved_pipeline"], ranks["original_pipeline"], fit)
    result["primary"] = primary
    result["useful_gain_flag"] = primary["delta_mrr"] >= .001 and primary["bootstrap_95"][0] > 0
    result["model_only_full_valid"] = {arm: metrics(ranks[arm + "_model"]) for arm in ("original", "improved")}
    result["attribution_original_weights"] = {
        key: delta_summary(ranks[key], ranks["original_pipeline"], fit, 3612 + i)
        for i, key in enumerate(("score_only_swap", "analogy_only_swap", "both_swap_original_weights"))}
    result["pipeline_uplift_over_model"] = {
        arm: delta_summary(ranks[arm + "_pipeline"], ranks[arm + "_model"], fit, 3615 + i)
        for i, arm in enumerate(("original", "improved"))}
    result["family_direction"] = []
    for name in np.unique(family):
        for d in (0, 1):
            mask = (~fit) & (family == name) & (direction == d)
            if mask.any():
                result["family_direction"].append(dict(family=str(name), direction="tail" if d == 0 else "head",
                    queries=int(mask.sum()), mrr={key: metrics(value[mask])["mrr"] for key, value in ranks.items()}))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="biokg/results/feature_compare/s0")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if not torch.cuda.is_available() or "1080 Ti" not in torch.cuda.get_device_name():
        raise RuntimeError("FC1 requires the local GTX 1080 Ti")
    repo = Path(__file__).resolve().parents[1]
    dataset = Path("/mnt/geocore/geocore/data_ogb/ogbl_biokg")
    opened = set()

    def guard(event, arguments):
        if event == "open" and isinstance(arguments[0], (str, bytes)):
            path = arguments[0].decode() if isinstance(arguments[0], bytes) else arguments[0]
            relative = check_data_path(path, dataset)
            if relative:
                opened.add(relative)

    sys.addaudithook(guard)
    original = repo / "biokg/checkpoints/dist_T2_s0.pt"
    improved = repo / "biokg/results/h35f/campaign_s0/single.pt"
    expected = {str(original): "b463a8bcc431ada38f4834e235677464f9cd3a86346f88aecd2fd681a01260ef",
                str(improved): "df7c783e7b8a6110f126bca1c60d89209735accbaa5ffc0b362f2cab52d9d2f1"}
    for path, digest in expected.items():
        assert sha256(path) == digest, path
    family_path = repo / "biokg/results/candidate_focus/s0/relation_families.json"
    references = {"original": repo / "biokg/results/h35/campaign_s0/frozen_queries.npz",
                  "improved": repo / "biokg/results/h35f/campaign_s0/single_queries.npz"}
    inputs = [original, improved, family_path, *references.values(),
              *(dataset / name for name in ("split/random/train.pt", "split/random/valid.pt", "raw/num-node-dict.csv.gz"))]
    sources = [repo / name for name in ("biokg/FEATURE_COMPARE.md", "biokg/compare_feature_pipeline.py",
        "biokg/test_feature_pipeline.py", "biokg/audit_feature_pipeline.py", "resonate.py",
        "biokg/relation_analogy.py", "biokg/mixed_operator.py", "biokg/joint_operator.py",
        "biokg/dual_operator.py", "biokg/gradient_diagnostic.py", "biokg/train_candidate_focus.py",
        "biokg/train_joint_operator.py", "biokg/train_dual_operator.py", "biokg/train_biokg_comp.py")]
    hashes = {str(path): sha256(path) for path in inputs + sources}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)

    def write(name, value):
        (out / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

    def log(**event):
        event.setdefault("lr", 0.0)
        print(json.dumps(event, allow_nan=False), flush=True)
        with (out / "progress.jsonl").open("a") as handle:
            handle.write(json.dumps(event, allow_nan=False) + "\n")

    receipt = dict(protocol="FC1", hashes=hashes, checkpoints=expected, references={k: str(v) for k, v in references.items()},
                   channels=CHANNELS, views=VIEWS, min_rows=MIN_ROWS, mask_seed=0, bootstrap_seed=3611,
                   device=torch.cuda.get_device_name(), torch=torch.__version__, numpy=np.__version__,
                   training_performed=False, test_loaded=False, lr=0.0,
                   started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    write("prerun.json", receipt)
    train, valid, offset, counts = train_valid(dataset)
    n, n_ent, n_rel = len(valid["head"]), sum(counts.values()), 2 * (int(train["relation"].max()) + 1)
    assert (n, n_ent, n_rel) == (162886, 93773, 102)
    models = []
    for index, path in enumerate((original, improved)):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        assert ck["offset"] == offset and ck["n_rel"] == n_rel
        model = from_student(ck, "single", "cuda") if index == 0 else restore_mixed(ck, "cuda")
        model.eval().requires_grad_(False)
        assert model.mode == "single" and model.n_params() == 27124129
        models.append(model)
    before = [parameter_hashes(model) for model in models]
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    raw = np.lib.format.open_memmap(out / "raw_features.npy", mode="w+", dtype=np.float32, shape=(8, 2*n, 501))
    log(stage="start", triples=n, queries=2*n, model_updates=0)
    graph_audit = build_features(models, train, valid, offset, n_ent, n_rel, raw, log)
    after = [parameter_hashes(model) for model in models]
    assert before == after and all(not p.requires_grad and p.grad is None for model in models for p in model.parameters())
    peak = torch.cuda.max_memory_allocated()
    del models, model, ck
    torch.cuda.empty_cache()
    z = np.lib.format.open_memmap(out / "normalized_features.npy", mode="w+", dtype=np.float32, shape=raw.shape)
    feature_ranks = np.empty((8, 2*n), np.float64)
    for j, name in enumerate(CHANNELS):
        for start in range(0, 2*n, 2048):
            sl = slice(start, start + 2048)
            feature_ranks[j, sl] = rank_rows(raw[j, sl])
            z[j, sl] = normalize(raw[j, sl])
        log(stage="normalized", feature=name)
    z.flush()
    reference_audit = {}
    for arm, channel in (("original", 0), ("improved", 3)):
        with np.load(references[arm], allow_pickle=False) as saved:
            reference = saved["rank"].astype(np.float64)
        actual = feature_ranks[channel]
        discrepancy = float((1 / actual - 1 / reference).mean())
        reference_audit[arm] = dict(mrr_delta=discrepancy, differing_ranks=int((actual != reference).sum()))
        assert abs(discrepancy) <= 1e-6, reference_audit
    fit = paired_masks(n)
    relations, direction = np.tile(valid["relation"], 2), np.repeat([0, 1], n)
    families = json.loads(family_path.read_text())
    family = np.array([families[str(int(r))] for r in relations])
    np.savez_compressed(out / "metadata.npz", fit=fit, relation=relations, direction=direction, family=family)
    ranks, recipes = {}, {}
    for arm in ("original", "improved"):
        features = [z[j] for j in VIEWS[arm]]
        log(stage="selection_start", arm=arm)
        recipes[arm] = fit_recipe(features, fit, relations, family, direction,
                                  log=lambda **event: log(arm=arm, **event))
        write(f"{arm}_recipe.json", recipes[arm])  # frozen before report scoring
        ranks[arm + "_pipeline"] = apply_recipe(features, recipes[arm], relations, direction)
        ranks[arm + "_uniform"] = mixture_ranks(features, np.full(5, .2, np.float32), np.arange(2*n))
        ranks[arm + "_model"] = feature_ranks[VIEWS[arm][0]]
        log(stage="pipeline_scored", arm=arm, report=metrics(ranks[arm + "_pipeline"][~fit]))
    for key, view in (("score_only_swap", "score_only_swap"), ("analogy_only_swap", "analogy_only_swap"),
                      ("both_swap_original_weights", "improved")):
        ranks[key] = apply_recipe([z[j] for j in VIEWS[view]], recipes["original"], relations, direction)
    for j, name in enumerate(CHANNELS):
        ranks["feature_" + name] = feature_ranks[j]
    np.savez_compressed(out / "ranks.npz", **ranks)
    report = summarize(ranks, fit, family, direction)
    report.update(seconds=time.monotonic()-started, peak_allocated_bytes=peak,
                  dataset_files_opened=sorted(opened), models_unchanged=True,
                  reference_rank_audit=reference_audit)
    write("summary.json", report)
    for path, digest in hashes.items():
        assert sha256(path) == digest, path
    artifacts = {name: sha256(out / name) for name in ("raw_features.npy", "normalized_features.npy", "metadata.npz",
                 "original_recipe.json", "improved_recipe.json", "ranks.npz", "summary.json")}
    write("audit.json", dict(models_before=before, models_after=after, models_unchanged=True,
        gradients_absent=True, input_source_hashes_unchanged=True, dataset_files_opened=sorted(opened),
        graph_audit=graph_audit, reference_rank_audit=reference_audit, artifact_sha256=artifacts))
    log(stage="complete", primary=report["primary"], useful_gain_flag=report["useful_gain_flag"], seconds=report["seconds"])


if __name__ == "__main__":
    main()
