"""RF1: two fixed, separate frozen-A retrieval interventions. No TEST option."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from resonate import cnorm
from biokg.compare_feature_pipeline import (
    apply_recipe, fit_recipe, graph_for, metrics, mixture_ranks, paired_masks,
)
from biokg.gradient_diagnostic import sha256
from biokg.mixed_operator import restore_mixed
from biokg.relation_analogy import holder_pool, normalize, rank_rows
from biokg.train_biokg_comp import globalize
from biokg.train_candidate_focus import check_data_path, train_valid
from biokg.train_joint_operator import parameter_hashes


CHANNELS = ("weighted_jaccard_max", "weighted_jaccard_top3", "analogy_gate", "jaccard_gate")
ARMS = ("rarity", "support", "half_strength")
BASELINE_MRR = 0.8540538579928584


def rarity_weights(graph):
    """Document frequencies and active population derive from TRAIN edges only."""
    if not np.all(graph.data == 1):
        raise ValueError("Expected unique-edge adjacency")
    active = np.diff(graph.indptr) > 0
    df = np.asarray(graph.sum(0)).ravel()
    return (1 + np.log((int(active.sum()) + 1) / (df.astype(np.float64) + 1))).astype(np.float32)


def graph_similarities(graph, local_query, weights=None):
    weighted = graph if weights is None else graph.multiply(weights[None, :]).tocsr()
    degree = np.asarray(weighted.sum(1)).ravel().astype(np.float32)
    inter = (weighted[local_query] @ graph.T).toarray().astype(np.float32)
    union = degree[local_query, None] + degree[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def support_quality(similarities, active, local_query):
    """Shape (metrics, queries, TRAIN/query-source rows); no candidate labels."""
    values = similarities[:, :, active].astype(np.float64)
    total = values.sum(-1)
    square = np.square(values).sum(-1)
    own_active = active[local_query]
    own = similarities[:, np.arange(len(local_query)), local_query].astype(np.float64)
    total -= own * own_active[None, :]
    square -= np.square(own) * own_active[None, :]
    count = int(active.sum()) - own_active.astype(np.int64)
    mean64 = total / np.maximum(count, 1)[None, :]
    variance = np.maximum(square / np.maximum(count, 1)[None, :] - np.square(mean64), 0)
    mean, std = mean64.astype(np.float32), np.sqrt(variance).astype(np.float32)
    excess = np.maximum(similarities - mean[:, :, None], 0)
    quality = excess / (excess + std[:, :, None] + np.float32(1e-8))
    usable = (count[None, :] >= 2) & (std > 1e-8)
    quality *= usable[:, :, None]
    return quality.astype(np.float32), mean, std, count


def pool_followups(cosine, jaccard, weighted_jaccard, graph, holders, local_query):
    """Prepare query evidence; candidate-independent background statistics."""
    similarities = np.stack([cosine, jaccard])
    quality, mean, std, count = support_quality(similarities, np.diff(graph.indptr) > 0, local_query)
    return np.concatenate([weighted_jaccard[None], quality], axis=0), dict(mean=mean, std=std, count=count)


def candidate_followups(prepared, graph, holders, local_query, candidates):
    out = np.empty((4, len(local_query), candidates.shape[1]), np.float32)
    holder_count = np.diff(holders.indptr)
    for j, source in enumerate(local_query):
        mx, top = holder_pool(prepared[:, j], holders, source, candidates[j])
        own_targets = graph.indices[graph.indptr[source]:graph.indptr[source + 1]]
        count = holder_count[candidates[j]] - np.isin(candidates[j], own_targets).astype(np.int64)
        assert (count >= 0).all()
        gates = np.maximum(top[1:], 0) * (np.minimum(count, 3).astype(np.float32) / 3)[None, :]
        if not len(own_targets):
            mx[0] = top[0] = -1
            gates[1] = 0
        out[0, j], out[1, j] = mx[0], top[0]
        out[2:, j] = gates
    if not np.isfinite(out).all() or (out[2:] < 0).any() or (out[2:] > 1).any():
        raise ValueError("Invalid retrieval feature/gate")
    return out


def supported_feature(model, retrieval, gate):
    """Convex candidate-specific return-to-model; no second normalization."""
    gate = np.asarray(gate, np.float32)
    if not np.isfinite(gate).all() or (gate < 0).any() or (gate > 1).any():
        raise ValueError("Gate must be in [0,1]")
    return (gate * retrieval + (np.float32(1) - gate) * model).astype(np.float32)


class HalfFeature:
    def __init__(self, model, retrieval):
        self.model, self.retrieval, self.shape = model, retrieval, retrieval.shape

    def __getitem__(self, indices):
        return supported_feature(self.model[indices], self.retrieval[indices], .5)


def feature_views(base, weighted, supported):
    unchanged = [base[j] for j in (3, 4, 5, 6, 7)]
    return dict(baseline=unchanged, rarity=unchanged[:3] + [weighted[0], weighted[1]],
                support=[unchanged[0]] + [supported[j] for j in range(4)],
                half_strength=[unchanged[0]] + [HalfFeature(unchanged[0], f) for f in unchanged[1:]])


@torch.inference_mode()
def build_features(model, train, valid, offset, n_ent, n_rel, output, base_raw, log, chunk=32):
    h, r, t = globalize(valid, offset)
    th, tr, tt = globalize(train, offset)
    n, processed, last = len(r), 0, time.monotonic()
    started, seen = time.monotonic(), np.zeros(2*n, bool)
    audits = []
    for direction in (0, 1):
        ts, td = (th, tt) if direction == 0 else (tt, th)
        qs, qt = (h, t) if direction == 0 else (t, h)
        types = np.asarray(valid["tail_type" if direction == 0 else "head_type"])
        negatives = valid["tail_neg" if direction == 0 else "head_neg"]
        for relation in np.unique(r):
            rows, mask = np.flatnonzero(r == relation), tr == relation
            sources, graph, duplicates = graph_for(ts[mask], td[mask], qs[rows], n_ent)
            holders, weights = graph.tocsc(), rarity_weights(graph)
            # Form once per directed relation; expensive sparse products stay batched.
            weighted_graph = graph.multiply(weights[None, :]).tocsr()
            degree = np.diff(graph.indptr).astype(np.float32)
            weighted_degree = np.asarray(weighted_graph.sum(1)).ravel().astype(np.float32)
            table = cnorm(model.E[torch.as_tensor(sources, device=model.E.device)])
            directed = int(relation + direction * (n_rel // 2))
            maximum_reference_error = 0.
            gate_total, gate_zero, gate_rows = np.zeros(2), np.zeros(2, np.int64), 0
            for start in range(0, len(rows), chunk):
                indices = rows[start:start + chunk]
                local = np.searchsorted(sources, qs[indices])
                cosine = (table[torch.as_tensor(local, device=table.device)] @ table.conj().T).real.cpu().numpy()
                inter = (graph[local] @ graph.T).toarray()
                union = degree[local, None] + degree[None, :] - inter
                jac = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
                wi = (weighted_graph[local] @ graph.T).toarray()
                wu = weighted_degree[local, None] + weighted_degree[None, :] - wi
                wjac = np.divide(wi, wu, out=np.zeros_like(wi), where=wu > 0)
                prepared, background = pool_followups(cosine, jac, wjac, graph, holders, local)
                offs = np.array([offset[name] for name in types[indices]])
                candidates = np.concatenate([qt[indices, None], negatives[indices] + offs[:, None]], axis=1)
                values = candidate_followups(prepared, graph, holders, local, candidates)
                out_rows = direction * n + indices
                if start == 0:
                    order = np.arange(candidates.shape[1])[::-1].copy()
                    shuffled = candidate_followups(prepared, graph, holders, local, candidates[:, order])
                    np.testing.assert_array_equal(shuffled, values[:, :, order])
                    src = torch.as_tensor(qs[indices], device=model.E.device)
                    rel = torch.full((len(src),), directed, device=model.E.device, dtype=torch.long)
                    score = model.candidate_outputs(src, rel, torch.as_tensor(candidates, device=model.E.device))[0]
                    old = np.empty((5, len(indices), candidates.shape[1]), np.float32)
                    old[0] = score.cpu().numpy()
                    for j, source in enumerate(local):
                        mx, top = holder_pool(np.stack([cosine[j], jac[j]]), holders, source, candidates[j])
                        if degree[source] == 0:
                            mx[1] = top[1] = -1
                        old[1, j], old[2, j] = mx[0] ** 3, top[0] ** 3
                        old[3, j], old[4, j] = mx[1], top[1]
                    expected = np.stack([base_raw[c, out_rows] for c in (3, 4, 5, 6, 7)])
                    np.testing.assert_allclose(old, expected, atol=1e-6, rtol=1e-6)
                    maximum_reference_error = float(np.max(np.abs(old - expected)))
                    first_background = dict(mean_min=background["mean"].min(axis=1).tolist(),
                                            mean_max=background["mean"].max(axis=1).tolist(),
                                            std_min=background["std"].min(axis=1).tolist(),
                                            std_max=background["std"].max(axis=1).tolist(),
                                            count_min=int(background["count"].min()))
                assert not seen[out_rows].any()
                output[:, out_rows] = values
                seen[out_rows] = True
                gate_total += values[2:].sum((1, 2), dtype=np.float64)
                gate_zero += (values[2:] == 0).sum((1, 2))
                gate_rows += values.shape[1] * values.shape[2]
                processed += len(indices)
                now = time.monotonic()
                if now - last >= 35:
                    log(stage="feature_generation", queries=processed, total=2*n, seconds=now-started,
                        baseline_report_mrr=BASELINE_MRR, treatment_mrr="pending fixed evaluation")
                    last = now
            observed_weights = weights[np.asarray(graph.sum(0)).ravel() > 0]
            audits.append(dict(relation=int(relation), direction=direction, queries=len(rows),
                unique_train_edges=int(graph.nnz), duplicate_edges_removed=int(duplicates),
                active_train_sources=int((degree > 0).sum()), weight_min=float(observed_weights.min()),
                weight_max=float(observed_weights.max()), candidate_permutation_passed=True,
                unchanged_feature_max_error=maximum_reference_error, first_batch_background=first_background,
                mean_gate=(gate_total/gate_rows).tolist(), zero_gate_fraction=(gate_zero/gate_rows).tolist()))
            log(stage="relation_complete", relation=int(relation), direction=direction, queries=processed,
                baseline_report_mrr=BASELINE_MRR)
            del table
    assert seen.all()
    if hasattr(output, "flush"):
        output.flush()
    log(stage="features_complete", queries=processed, seconds=time.monotonic()-started)
    return audits


def comparison(treatment, control, fit, seed, adjusted=False):
    n = len(fit)//2
    assert np.array_equal(fit[:n], fit[n:])
    delta = 1/treatment.astype(np.float64) - 1/control.astype(np.float64)
    paired = ((delta[:n] + delta[n:])/2)[~fit[:n]]
    rng = np.random.default_rng(seed)
    means = [rng.choice(paired, len(paired), replace=True).mean() for _ in range(2000)]
    quantiles = [.0125, .9875] if adjusted else [.025, .975]
    report = ~fit
    return dict(delta_mrr=float(delta[report].mean()), bootstrap_interval=np.quantile(means, quantiles).tolist(),
                interval_percent=97.5 if adjusted else 95, replicates=2000, seed=seed,
                recovered_top1=int(((treatment==1) & (control>1) & report).sum()),
                lost_top1=int(((treatment>1) & (control==1) & report).sum()))


def summarize(ranks, fit, family, direction):
    primary = {arm: comparison(ranks[arm], ranks["baseline"], fit, 3621+i, True)
               for i, arm in enumerate(("rarity", "support"))}
    result = dict(protocol="RF1", training_performed=False, test_loaded=False, model_updates=0,
                  fit_triples=int(fit[:len(fit)//2].sum()), report_triples=int((~fit[:len(fit)//2]).sum()),
                  metrics={key: dict(fit=metrics(rank[fit]), report=metrics(rank[~fit])) for key, rank in ranks.items()},
                  primary=primary, primary_interval_correction="Bonferroni for two planned contrasts",
                  advance_flags={arm: row["delta_mrr"] >= .001 and row["bootstrap_interval"][0] > 0
                                 for arm, row in primary.items()},
                  half_strength_vs_baseline=comparison(ranks["half_strength"], ranks["baseline"], fit, 3623),
                  support_vs_half_strength=comparison(ranks["support"], ranks["half_strength"], fit, 3624))
    slices = []
    for name in np.unique(family):
        for d in (0, 1):
            mask = (~fit) & (family == name) & (direction == d)
            if mask.any():
                slices.append(dict(family=str(name), direction="tail" if d == 0 else "head", queries=int(mask.sum()),
                                   mrr={key: metrics(rank[mask])["mrr"] for key, rank in ranks.items()}))
    result["family_direction"] = slices
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="biokg/results/retrieval_followups/s0")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if not torch.cuda.is_available() or "1080 Ti" not in torch.cuda.get_device_name():
        raise RuntimeError("RF1 requires the local GTX 1080 Ti")
    repo = Path(__file__).resolve().parents[1]
    prior = repo / "biokg/results/feature_compare/s0_retry1"
    dataset = Path("/mnt/geocore/geocore/data_ogb/ogbl_biokg")
    opened = set()

    def guard(event, arguments):
        if event == "open" and isinstance(arguments[0], (str, bytes)):
            path = arguments[0].decode() if isinstance(arguments[0], bytes) else arguments[0]
            relative = check_data_path(path, dataset)
            if relative:
                opened.add(relative)

    sys.addaudithook(guard)
    old_receipt = json.loads((prior / "prerun.json").read_text())
    old_audit = json.loads((prior / "audit.json").read_text())
    assert json.loads((prior / "endpoint_audit.json").read_text())["audit_passed"]
    checkpoint = repo / "biokg/results/h35f/campaign_s0/single.pt"
    assert old_receipt["checkpoints"][str(checkpoint)] == "df7c783e7b8a6110f126bca1c60d89209735accbaa5ffc0b362f2cab52d9d2f1"
    # Verify FC1 source and data inputs, but do not reopen the unused original model.
    hashes = {path: digest for path, digest in old_receipt["hashes"].items()
              if "/checkpoints/dist_T2_s0.pt" not in path and "/h35/campaign_s0/" not in path}
    hashes.update({str(prior/name): old_audit["artifact_sha256"][name] for name in
                   ("raw_features.npy", "normalized_features.npy", "metadata.npz", "improved_recipe.json", "ranks.npz", "summary.json")})
    for path, digest in hashes.items():
        assert sha256(path) == digest, path
    for name in ("biokg/RETRIEVAL_FOLLOWUPS.md", "biokg/retrieval_followups.py", "biokg/test_retrieval_followups.py",
                 "biokg/audit_retrieval_followups.py"):
        hashes[str(repo/name)] = sha256(repo/name)
    for name in ("prerun.json", "audit.json", "endpoint_audit.json"):
        hashes[str(prior/name)] = sha256(prior/name)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)

    def write(name, value):
        (out/name).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")

    def log(**event):
        event.update(lr=0.0, lr_status="not applicable: frozen inference")
        print(json.dumps(event, allow_nan=False), flush=True)
        with (out/"progress.jsonl").open("a") as handle:
            handle.write(json.dumps(event, allow_nan=False)+"\n")

    write("prerun.json", dict(protocol="RF1", hashes=hashes, prior=str(prior), checkpoint=str(checkpoint),
          channels=CHANNELS, arms=ARMS, device=torch.cuda.get_device_name(), torch=torch.__version__, numpy=np.__version__,
          model_updates=0, training_performed=False, test_loaded=False,
          started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
    log(stage="inputs_verified", baseline_report_mrr=BASELINE_MRR)
    train, valid, offset, counts = train_valid(dataset)
    n, n_ent, n_rel = len(valid["head"]), sum(counts.values()), 2*(int(train["relation"].max())+1)
    assert (n, n_ent, n_rel) == (162886, 93773, 102)
    with np.load(prior/"metadata.npz", allow_pickle=False) as saved:
        fit, relations, direction, family = (saved[k] for k in ("fit", "relation", "direction", "family"))
    np.testing.assert_array_equal(fit, paired_masks(n))
    np.testing.assert_array_equal(relations, np.tile(valid["relation"], 2))
    np.testing.assert_array_equal(direction, np.repeat([0,1], n))
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert ck["offset"] == offset and ck["n_rel"] == n_rel and ck["mode"] == "single"
    model = restore_mixed(ck, "cuda")
    assert model.n_params() == 27124129
    before = parameter_hashes(model)
    assert before == old_audit["models_after"][1]
    raw = np.load(prior/"raw_features.npy", mmap_mode="r", allow_pickle=False)
    base = np.load(prior/"normalized_features.npy", mmap_mode="r", allow_pickle=False)
    output = np.lib.format.open_memmap(out/"features_and_gates.npy", mode="w+", dtype=np.float32, shape=(4,2*n,501))
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    graph_audit = build_features(model, train, valid, offset, n_ent, n_rel, output, raw, log)
    after = parameter_hashes(model)
    assert before == after and all(not p.requires_grad and p.grad is None for p in model.parameters())
    peak = torch.cuda.max_memory_allocated()
    del model, ck
    torch.cuda.empty_cache()
    weighted = np.lib.format.open_memmap(out/"weighted_normalized.npy", mode="w+", dtype=np.float32, shape=(2,2*n,501))
    supported = np.lib.format.open_memmap(out/"supported_members.npy", mode="w+", dtype=np.float32, shape=(4,2*n,501))
    for start in range(0, 2*n, 2048):
        sl = slice(start, start+2048)
        for j in range(2):
            weighted[j,sl] = normalize(output[j,sl])
        for j, channel in enumerate((4,5,6,7)):
            supported[j,sl] = supported_feature(base[3,sl], base[channel,sl], output[2+j//2,sl])
    weighted.flush()
    supported.flush()
    views = feature_views(base, weighted, supported)
    baseline_recipe = json.loads((prior/"improved_recipe.json").read_text())
    ranks = {"baseline": apply_recipe(views["baseline"], baseline_recipe, relations, direction)}
    with np.load(prior/"ranks.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(ranks["baseline"], saved["improved_pipeline"])
        ranks["model"] = saved["improved_model"]
    assert metrics(ranks["baseline"][~fit])["mrr"] == BASELINE_MRR
    log(stage="baseline_reproduced", report=metrics(ranks["baseline"][~fit]))
    for arm in ARMS:
        log(stage="selection_start", arm=arm)
        recipe = fit_recipe(views[arm], fit, relations, family, direction,
                            log=lambda **event: log(arm=arm, **event))
        write(arm+"_recipe.json", recipe)
        ranks[arm] = apply_recipe(views[arm], recipe, relations, direction)
        ranks[arm+"_fixed_baseline_recipe"] = apply_recipe(views[arm], baseline_recipe, relations, direction)
        ranks[arm+"_uniform"] = mixture_ranks(views[arm], np.full(5,.2,np.float32), np.arange(2*n))
        for j, feature in enumerate(views[arm][1:]):
            ranks[f"{arm}_member{j+1}"] = mixture_ranks([feature], [1], np.arange(2*n))
        log(stage="arm_evaluated", arm=arm, report=metrics(ranks[arm][~fit]))
    np.savez_compressed(out/"ranks.npz", **ranks)
    summary = summarize(ranks, fit, family, direction)
    summary.update(seconds=time.monotonic()-started, peak_allocated_bytes=peak, model_unchanged=True,
                   baseline_ranks_exact=True, dataset_files_opened=sorted(opened))
    write("summary.json", summary)
    for path, digest in hashes.items():
        assert sha256(path) == digest, path
    artifacts = {name: sha256(out/name) for name in ("features_and_gates.npy", "weighted_normalized.npy", "supported_members.npy",
                "rarity_recipe.json", "support_recipe.json", "half_strength_recipe.json", "ranks.npz", "summary.json")}
    write("audit.json", dict(model_before=before, model_after=after, model_unchanged=True, gradients_absent=True,
          baseline_ranks_exact=True, input_source_hashes_unchanged=True, graph_audit=graph_audit,
          dataset_files_opened=sorted(opened), artifact_sha256=artifacts))
    log(stage="complete", primary=summary["primary"], advance_flags=summary["advance_flags"], seconds=summary["seconds"])


if __name__ == "__main__":
    main()
