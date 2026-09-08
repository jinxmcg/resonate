"""CS1 candidate-side analogy. Frozen own A, CPU, TRAIN/VALID only."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from resonate import cnorm
from biokg.compare_feature_pipeline import graph_for, metrics
from biokg.confirm_feature_blend import (
    N_TRIPLES, SEEDS, insert_report, load_features, split_masks, summarize as bc_summary,
    verify_hashes, views, write_json,
)
from biokg.gradient_diagnostic import sha256
from biokg.mixed_operator import restore_mixed
from biokg.relation_analogy import normalize, rank_rows
from biokg.train_biokg_comp import globalize
from biokg.train_candidate_focus import check_data_path, train_valid
from biokg.train_joint_operator import parameter_hashes


BETAS = (0., .025, .05, .1, .2)
BASELINE_MRR = .8555176434837662
DATASET = Path("/mnt/geocore/geocore/data_ogb/ogbl_biokg")


def install_guard(opened):
    def guard(event, arguments):
        if event == "open" and isinstance(arguments[0], (str, bytes)):
            value = arguments[0].decode() if isinstance(arguments[0], bytes) else arguments[0]
            relative = check_data_path(value, DATASET)
            if relative:
                opened.add(relative)
    sys.addaudithook(guard)


@torch.inference_mode()
def cosine_cache(table, log=lambda **x: None, chunk=512):
    if table.device.type != "cpu":
        raise ValueError("CS1 leaves the GPU untouched")
    size = len(table)
    result = np.empty((size, size), np.float32)
    right = table.conj().T.resolve_conj()
    started, last = time.monotonic(), time.monotonic()
    for start in range(0, size, chunk):
        result[start:start+chunk] = (table[start:start+chunk] @ right).real.numpy()
        if time.monotonic() - last >= 35:
            log(stage="cosine_cache", rows=min(start+chunk, size), total=size,
                stage_seconds=time.monotonic()-started)
            last = time.monotonic()
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite embedding cosine")
    return result


def pool_cached(cosine, neighbors, candidates, candidate_chunk=512, neighbor_chunk=512):
    """Candidate-local scores; neither labels nor candidate pool statistics enter."""
    neighbors = np.unique(np.asarray(neighbors, np.int64))
    candidates = np.asarray(candidates, np.int64)
    result = np.full((2, len(candidates)), -1, np.float32)
    for start in range(0, len(candidates), candidate_chunk):
        cs = candidates[start:start+candidate_chunk]
        top = np.full((len(cs), 3), -np.inf, np.float32)
        for ns in range(0, len(neighbors), neighbor_chunk):
            nb = neighbors[ns:ns+neighbor_chunk]
            values = cosine[cs[:, None], nb[None, :]].copy()
            values[cs[:, None] == nb[None, :]] = -np.inf
            merged = np.concatenate([top, values], axis=1)
            top = np.partition(merged, -3, axis=1)[:, -3:]
        top = np.sort(top, axis=1)[:, ::-1].copy()
        good = np.isfinite(top)
        count = good.sum(1)
        maximum = np.where(count > 0, top[:, 0], -1)
        mean = np.where(count > 0, np.where(good, top, 0).sum(1) / np.maximum(count, 1), -1).astype(np.float32)
        result[0, start:start+len(cs)] = maximum ** 3
        result[1, start:start+len(cs)] = mean ** 3
    return result


def direct_features(table, neighbors, candidates):
    """Independent tiny-query reference: direct dots, sorted holders, no cache."""
    result = np.full((2, len(candidates)), -1, np.float32)
    for i, candidate in enumerate(candidates):
        nb = np.unique(neighbors)
        nb = nb[nb != candidate]
        if len(nb):
            sims = (table[nb] * table[candidate].conj()).sum(1).real.astype(np.float32)
            top = np.sort(sims)[-3:][::-1].copy()
            result[:, i] = np.array([top[0] ** 3, top.mean() ** 3], np.float32)
    return result


def candidates_for(valid, indices, direction, offset, positive):
    key = "tail" if direction == 0 else "head"
    types = np.asarray(valid[key+"_type"])
    offsets = np.array([offset[name] for name in types[indices]])
    return np.concatenate([positive[indices, None], valid[key+"_neg"][indices] + offsets[:, None]], axis=1)


@torch.inference_mode()
def build_features(model, train, valid, offset, counts, output, log=lambda **x: None):
    h, r, t = globalize(valid, offset)
    th, tr, tt = globalize(train, offset)
    n, n_ent = len(r), sum(counts.values())
    table = cnorm(model.E).cpu()
    table_np = table.numpy()
    seen = np.zeros(2*n, bool)
    processed, audits, last = 0, [], time.monotonic()
    for target_type in sorted(counts):
        tasks = []
        for d in (0, 1):
            types = np.asarray(valid["tail_type" if d == 0 else "head_type"])
            for rel in np.unique(r):
                rows = np.flatnonzero(r == rel)
                if len(np.unique(types[rows])) != 1:
                    raise ValueError("Mixed target types within directed relation")
                if types[rows[0]] == target_type:
                    tasks.append((d, int(rel), rows))
        if not tasks:
            continue
        off, count = offset[target_type], counts[target_type]
        log(stage="type_start", target_type=target_type, entities=count,
            cosine_cache_bytes=count*count*4, queries=processed, total=2*n)
        cosine = cosine_cache(table[off:off+count], log=log)
        for d, rel, rows in tasks:
            ts, td = (th, tt) if d == 0 else (tt, th)
            qs, qt = (h, t) if d == 0 else (t, h)
            mask = tr == rel
            sources, graph, duplicates = graph_for(ts[mask], td[mask], qs[rows], n_ent)
            order = np.argsort(qs[rows], kind="stable")
            sorted_rows = rows[order]
            starts = np.r_[0, np.flatnonzero(np.diff(qs[sorted_rows]))+1, len(rows)]
            examples, zero_rows, had_nonempty = [], 0, False
            degree_sum = 0
            for begin, end in zip(starts[:-1], starts[1:]):
                indices = sorted_rows[begin:end]
                source = int(qs[indices[0]])
                local = int(np.searchsorted(sources, source))
                neighbors = graph.indices[graph.indptr[local]:graph.indptr[local+1]]
                if len(neighbors) and ((neighbors < off).any() or (neighbors >= off+count).any()):
                    raise ValueError("TRAIN target type mismatch")
                candidates = candidates_for(valid, indices, d, offset, qt)
                if (candidates < off).any() or (candidates >= off+count).any():
                    raise ValueError("Candidate type mismatch")
                unique, inverse = np.unique(candidates, return_inverse=True)
                values = pool_cached(cosine, neighbors-off, unique-off)
                out_rows = d*n + indices
                assert not seen[out_rows].any()
                output[:, out_rows] = values[:, inverse.ravel()].reshape(2, len(indices), candidates.shape[1])
                seen[out_rows] = True
                zero_rows += len(indices) if not len(neighbors) else 0
                degree_sum += len(neighbors)*len(indices)
                if not examples or (len(neighbors) and not had_nonempty):
                    columns = np.unique(np.linspace(0, candidates.shape[1]-1, 17, dtype=np.int64))
                    sample = candidates[0, columns]
                    actual = output[:, out_rows[0], columns]
                    expected = direct_features(table_np, neighbors, sample)
                    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
                    reversed_values = pool_cached(cosine, neighbors-off, (sample-off)[::-1], 7, 11)
                    np.testing.assert_array_equal(reversed_values[:, ::-1], actual)
                    examples.append(dict(row=int(out_rows[0]), source=source, columns=columns.tolist(),
                        candidates=sample.tolist(), neighbors=neighbors.tolist(),
                        values=actual.tolist(), direct_max_error=float(np.max(np.abs(actual-expected))),
                        candidate_permutation_and_chunk_exact=True))
                    had_nonempty = had_nonempty or bool(len(neighbors))
                processed += len(indices)
                if time.monotonic()-last >= 35:
                    log(stage="features", target_type=target_type, relation=rel, direction=d,
                        queries=processed, total=2*n)
                    last = time.monotonic()
            audits.append(dict(relation=rel, direction=d, target_type=target_type, queries=len(rows),
                query_sources=len(starts)-1, unique_train_edges=int(graph.nnz),
                duplicate_edges_removed=int(duplicates), zero_neighbor_queries=zero_rows,
                mean_train_neighbors=degree_sum/len(rows), examples=examples))
            log(stage="relation_complete", relation=rel, direction=d, queries=processed, total=2*n)
        del cosine
    assert seen.all()
    if hasattr(output, "flush"):
        output.flush()
    return audits


def baseline_scores(features, recipe, relation, direction):
    result = np.empty(features[0].shape, np.float32)
    for rel in np.unique(relation):
        for d in (0, 1):
            rows = np.flatnonzero((relation == rel) & (direction == d))
            if not len(rows):
                continue
            weights = recipe["groups"][f"{rel}/{d}"]["weights"]
            for start in range(0, len(rows), 2048):
                ids = rows[start:start+2048]
                scores = np.zeros((len(ids), result.shape[1]), np.float32)
                for feature, weight in zip(features, np.asarray(weights, np.float32)):
                    if weight:
                        scores += np.asarray(feature[ids], np.float32) * weight
                result[ids] = scores
    return result


def beta_ranks(base, extra, rows, beta):
    rows = np.asarray(rows, np.int64)
    beta = np.float32(beta)
    result = np.empty(len(rows), np.float64)
    for start in range(0, len(rows), 2048):
        ids = rows[start:start+2048]
        scores = (np.float32(1)-beta)*base[ids] + beta*extra[ids]
        result[start:start+len(ids)] = rank_rows(scores)
    return result


def fit_beta(base, extra, fit, relation, family, direction, min_rows=2000):
    ids = np.flatnonzero(fit)
    if not len(ids):
        raise ValueError("Empty fit selection")
    rr = np.full((len(BETAS), len(fit)), np.nan)
    for i, beta in enumerate(BETAS):
        rr[i, ids] = 1/beta_ranks(base, extra, ids, beta)

    def choose(rows):
        means = rr[:, rows].mean(1)
        index = int(np.argmax(means))
        return dict(beta=BETAS[index], fit_mrr=float(means[index]), fit_queries=len(rows))

    global_choice = choose(ids)
    families, groups = {}, {}
    for name in np.unique(family):
        for d in (0, 1):
            rows = np.flatnonzero(fit & (family == name) & (direction == d))
            if len(rows) >= min_rows:
                families[f"{name}/{d}"] = choose(rows)
    for rel in np.unique(relation):
        for d in (0, 1):
            mask = (relation == rel) & (direction == d)
            if not mask.any():
                continue
            names = np.unique(family[mask])
            assert len(names) == 1
            key = f"{names[0]}/{d}"
            rows = np.flatnonzero(fit & mask)
            if len(rows) >= min_rows:
                chosen, level = choose(rows), "relation"
            else:
                chosen = families.get(key, global_choice)
                level = "family" if key in families else "global"
            groups[f"{rel}/{d}"] = dict(**chosen, level=level, group_fit_queries=len(rows))
    return dict(betas=list(BETAS), global_choice=global_choice, families=families,
                groups=groups, min_rows=min_rows)


def apply_beta(base, extra, recipe, relation, direction, report):
    ids = np.flatnonzero(report)
    result = np.full(len(ids), np.nan)
    for rel in np.unique(relation):
        for d in (0, 1):
            positions = np.flatnonzero((relation[ids] == rel) & (direction[ids] == d))
            if len(positions):
                beta = recipe["groups"][f"{rel}/{d}"]["beta"]
                result[positions] = beta_ranks(base, extra, ids[positions], beta)
    assert np.isfinite(result).all()
    return result


def summarize(ranks, replicates=2000):
    result = bc_summary(dict(baseline=ranks["baseline"], half_strength=ranks["candidate_side"]),
                        bootstrap_seed=3641, replicates=replicates)

    def rename(value):
        if isinstance(value, dict):
            return {("candidate_side" if key == "half_strength" else key): rename(val) for key, val in value.items()}
        if isinstance(value, list):
            return [rename(item) for item in value]
        return value

    result = rename(result)
    result.update(protocol="CS1", dataset_splits_loaded=True, device="cpu", model_unchanged=True)
    return result


def input_receipt(repo):
    bc = repo / "biokg/results/blend_confirmation/s0"
    old = json.loads((bc / "prerun.json").read_text())
    audited = json.loads((bc / "audit.json").read_text())
    assert json.loads((bc / "endpoint_audit.json").read_text())["audit_passed"]
    hashes = dict(old["hashes"])
    rf = Path(old["prior_rf"])
    rf_receipt = json.loads((rf / "prerun.json").read_text())
    for name in ("raw/num-node-dict.csv.gz", "split/random/train.pt", "split/random/valid.pt"):
        path = str(DATASET / name)
        hashes[path] = rf_receipt["hashes"][path]
    names = ["ranks.npz", "summary.json"] + [f"seed{s}_fold{f}_half_strength_recipe.json"
                                               for s in SEEDS for f in (0, 1)]
    for name in names:
        hashes[str(bc / name)] = audited["artifact_sha256"][name]
    for name in ("prerun.json", "audit.json", "endpoint_audit.json"):
        hashes[str(bc / name)] = sha256(bc / name)
    for name in ("CANDIDATE_RETRIEVAL.md", "candidate_retrieval.py", "test_candidate_retrieval.py",
                 "audit_candidate_retrieval.py"):
        hashes[str(repo / "biokg" / name)] = sha256(repo / "biokg" / name)
    verify_hashes(hashes)
    return dict(protocol="CS1", hashes=hashes, prior_bc=str(bc), prior_fc=old["prior_fc"],
                prior_rf=old["prior_rf"], checkpoint=old["checkpoint"], device="cpu", threads=4,
                partition_seeds=list(SEEDS), betas=list(BETAS), min_fit_queries=2000,
                bootstrap_seed=3641, bootstrap_replicates=2000, numpy=np.__version__, torch=torch.__version__,
                training_performed=False, model_updates=0, lr=0.0, test_loaded=False,
                started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="biokg/results/candidate_retrieval/s0")
    args = parser.parse_args()
    torch.set_num_threads(4)
    repo, out = Path(__file__).resolve().parents[1], Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    opened = set()
    install_guard(opened)
    started = time.monotonic()

    def log(**event):
        event.update(seconds=time.monotonic()-started, lr=0.0,
                     lr_status="not applicable: frozen CPU retrieval", baseline_mrr=BASELINE_MRR)
        print(json.dumps(event, allow_nan=False), flush=True)
        with (out / "progress.jsonl").open("a") as handle:
            handle.write(json.dumps(event, allow_nan=False) + "\n")

    log(stage="verifying_inputs")
    receipt = input_receipt(repo)
    write_json(out / "prerun.json", receipt)
    train, valid, offset, counts = train_valid(DATASET)
    assert len(valid["head"]) == N_TRIPLES and sum(counts.values()) == 93773
    features, meta = load_features(receipt)
    relation, direction, family = (meta[k] for k in ("relation", "direction", "family"))
    np.testing.assert_array_equal(relation, np.tile(valid["relation"], 2))
    ck = torch.load(receipt["checkpoint"], map_location="cpu", weights_only=False)
    assert ck["offset"] == offset and ck["mode"] == "single" and ck["n_rel"] == 102
    model = restore_mixed(ck, "cpu")
    assert model.n_params() == 27124129
    before = parameter_hashes(model)
    prior_model = json.loads((Path(receipt["prior_rf"]) / "audit.json").read_text())["model_after"]
    assert before == prior_model
    raw = np.lib.format.open_memmap(out / "raw_features.npy", mode="w+", dtype=np.float32,
                                   shape=(2, 2*N_TRIPLES, 501))
    graph_audit = build_features(model, train, valid, offset, counts, raw, log)
    after = parameter_hashes(model)
    assert before == after and all(not p.requires_grad and p.grad is None for p in model.parameters())
    feature_seconds = time.monotonic()-started
    write_json(out / "graph_audit.json", graph_audit)
    del model, ck, train, valid
    normalized = np.lib.format.open_memmap(out / "normalized_features.npy", mode="w+", dtype=np.float32,
                                          shape=raw.shape)
    pair = np.lib.format.open_memmap(out / "pair_scores.npy", mode="w+", dtype=np.float32,
                                    shape=raw.shape[1:])
    standalone = {"candidate_max": np.empty(2*N_TRIPLES), "candidate_top3": np.empty(2*N_TRIPLES),
                  "candidate_pair": np.empty(2*N_TRIPLES)}
    for start in range(0, 2*N_TRIPLES, 2048):
        sl = slice(start, start+2048)
        for j, key in enumerate(("candidate_max", "candidate_top3")):
            normalized[j, sl] = normalize(raw[j, sl])
            standalone[key][sl] = rank_rows(normalized[j, sl])
        pair[sl] = (normalized[0, sl] + normalized[1, sl])*np.float32(.5)
        standalone["candidate_pair"][sl] = rank_rows(pair[sl])
    normalized.flush()
    pair.flush()
    log(stage="features_complete", queries=2*N_TRIPLES,
        standalone_full_valid={key: metrics(value) for key, value in standalone.items()},
        warning="standalone new features, not combined pipeline MRR")
    ranks = {a: np.full((3, 2*N_TRIPLES), np.nan) for a in ("baseline", "candidate_side")}
    coverage = {a: np.zeros((3, 2*N_TRIPLES), np.uint8) for a in ranks}
    bc = Path(receipt["prior_bc"])
    with np.load(bc / "ranks.npz", allow_pickle=False) as saved:
        old_ranks = saved["half_strength"]
    folds, artifacts = [], ["prerun.json", "graph_audit.json", "raw_features.npy",
                            "normalized_features.npy", "pair_scores.npy"]
    for i, seed in enumerate(SEEDS):
        for fold, fit in enumerate(split_masks(N_TRIPLES, seed)):
            log(stage="fold_start", seed=seed, fold=fold)
            baseline_recipe = json.loads((bc / f"seed{seed}_fold{fold}_half_strength_recipe.json").read_text())
            base = baseline_scores(features["half_strength"], baseline_recipe, relation, direction)
            report_ids = np.flatnonzero(~fit)
            control = beta_ranks(base, pair, report_ids, 0)
            np.testing.assert_array_equal(control, old_ranks[i, ~fit])
            recipe = fit_beta(base, pair, fit, relation, family, direction)
            filename = f"seed{seed}_fold{fold}_beta_recipe.json"
            write_json(out / filename, recipe)
            artifacts.append(filename)
            treatment = apply_beta(base, pair, recipe, relation, direction, ~fit)
            insert_report(ranks["baseline"][i], coverage["baseline"][i], fit, control)
            insert_report(ranks["candidate_side"][i], coverage["candidate_side"][i], fit, treatment)
            beta_counts = {str(beta): sum(group["beta"] == beta for group in recipe["groups"].values()) for beta in BETAS}
            result = dict(seed=seed, fold=fold, fit_triples=int(fit[:N_TRIPLES].sum()),
                report_triples=int((~fit[:N_TRIPLES]).sum()), global_beta=recipe["global_choice"]["beta"],
                directed_group_beta_counts=beta_counts,
                metrics=dict(baseline=metrics(control), candidate_side=metrics(treatment)),
                delta_mrr=float((1/treatment-1/control).mean()))
            folds.append(result)
            write_json(out / "folds.json", folds)
            log(stage="fold_complete", **result)
            del base
        assert all((coverage[a][i] == 1).all() for a in ranks)
    np.testing.assert_array_equal(ranks["baseline"], old_ranks)
    np.savez_compressed(out / "ranks.npz", **ranks, **standalone,
                        **{a+"_coverage": coverage[a] for a in ranks})
    summary = summarize(ranks)
    assert summary["mean_metrics"]["baseline"]["mrr"] == BASELINE_MRR
    summary.update(standalone_full_valid={key: metrics(value) for key, value in standalone.items()},
                   seconds_through_features=feature_seconds, seconds_through_summary=time.monotonic()-started,
                   dataset_files_opened=sorted(opened))
    write_json(out / "summary.json", summary)
    artifacts += ["folds.json", "ranks.npz", "summary.json"]
    verify_hashes(receipt["hashes"])
    write_json(out / "audit.json", dict(source_input_hashes_unchanged=True, model_before=before, model_after=after,
        model_unchanged=True, gradients_absent=True, baseline_ranks_exact=True, coverage_exact=True,
        dataset_files_opened=sorted(opened), artifact_sha256={name: sha256(out / name) for name in artifacts}))
    log(stage="complete", mean_metrics=summary["mean_metrics"], primary=summary["primary"],
        consistent_positive=summary["consistent_positive"], useful_gain_flag=summary["useful_gain_flag"])


if __name__ == "__main__":
    main()
