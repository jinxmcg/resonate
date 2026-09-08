"""CS1 saved-record audit, including direct TRAIN-neighbor feature checks."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from resonate import cnorm
from biokg.candidate_retrieval import (
    BETAS, DATASET, apply_beta, baseline_scores, beta_ranks, candidates_for,
    direct_features, fit_beta, install_guard, summarize,
)
from biokg.compare_feature_pipeline import apply_recipe, metrics
from biokg.confirm_feature_blend import N_TRIPLES, SEEDS, load_features, split_masks, verify_hashes, write_json
from biokg.mixed_operator import restore_mixed
from biokg.relation_analogy import normalize, rank_rows
from biokg.train_biokg_comp import globalize
from biokg.train_candidate_focus import train_valid
from biokg.train_joint_operator import parameter_hashes


def audit(directory):
    torch.set_num_threads(4)
    opened = set()
    install_guard(opened)
    out = Path(directory)
    receipt = json.loads((out / "prerun.json").read_text())
    recorded = json.loads((out / "audit.json").read_text())
    summary = json.loads((out / "summary.json").read_text())
    folds = json.loads((out / "folds.json").read_text())
    graph_audit = json.loads((out / "graph_audit.json").read_text())
    assert receipt["protocol"] == summary["protocol"] == "CS1"
    assert receipt["betas"] == list(BETAS) and receipt["partition_seeds"] == list(SEEDS)
    assert receipt["device"] == "cpu" and receipt["model_updates"] == 0
    assert not receipt["training_performed"] and not receipt["test_loaded"]
    assert recorded["model_before"] == recorded["model_after"]
    assert all(recorded[k] for k in ("model_unchanged", "gradients_absent", "baseline_ranks_exact",
                                     "source_input_hashes_unchanged", "coverage_exact"))
    verify_hashes(receipt["hashes"])
    verify_hashes({str(out / name): digest for name, digest in recorded["artifact_sha256"].items()})
    train, valid, offset, counts = train_valid(DATASET)
    h, r, t = globalize(valid, offset)
    th, tr, tt = globalize(train, offset)
    ck = torch.load(receipt["checkpoint"], map_location="cpu", weights_only=False)
    assert ck["offset"] == offset and ck["mode"] == "single"
    model = restore_mixed(ck, "cpu")
    assert parameter_hashes(model) == recorded["model_before"]
    with torch.inference_mode():
        table = cnorm(model.E).numpy()
    raw = np.load(out / "raw_features.npy", mmap_mode="r", allow_pickle=False)
    normalized = np.load(out / "normalized_features.npy", mmap_mode="r", allow_pickle=False)
    pair = np.load(out / "pair_scores.npy", mmap_mode="r", allow_pickle=False)
    assert raw.shape == normalized.shape == (2, 2*N_TRIPLES, 501)
    assert pair.shape == (2*N_TRIPLES, 501)
    assert raw.dtype == normalized.dtype == pair.dtype == np.float32
    assert len(graph_audit) == 102 and sum(a["queries"] for a in graph_audit) == 2*N_TRIPLES
    assert len({(a["relation"], a["direction"]) for a in graph_audit}) == 102
    maximum_error, examples_checked = 0., 0
    for record in graph_audit:
        rel, d = record["relation"], record["direction"]
        ts, td = (th, tt) if d == 0 else (tt, th)
        qs, qt = (h, t) if d == 0 else (t, h)
        rel_mask = tr == rel
        edges = ts[rel_mask]*sum(counts.values()) + td[rel_mask]
        unique_edges = len(np.unique(edges))
        assert unique_edges == record["unique_train_edges"]
        assert len(edges)-unique_edges == record["duplicate_edges_removed"]
        assert record["queries"] == int((r == rel).sum())
        assert record["examples"]
        for example in record["examples"]:
            row = example["row"]
            index = row % N_TRIPLES
            assert row // N_TRIPLES == d and int(r[index]) == rel
            source = int(qs[index])
            assert source == example["source"]
            neighbors = np.unique(td[rel_mask & (ts == source)])
            np.testing.assert_array_equal(neighbors, example["neighbors"])
            columns = np.asarray(example["columns"])
            candidates = candidates_for(valid, np.array([index]), d, offset, qt)[0, columns]
            np.testing.assert_array_equal(candidates, example["candidates"])
            actual = raw[:, row, columns]
            np.testing.assert_array_equal(actual, np.asarray(example["values"], np.float32))
            reference = direct_features(table, neighbors, candidates)
            np.testing.assert_allclose(actual, reference, atol=1e-5, rtol=1e-5)
            maximum_error = max(maximum_error, float(np.max(np.abs(actual-reference))))
            assert example["candidate_permutation_and_chunk_exact"]
            examples_checked += 1
    assert parameter_hashes(model) == recorded["model_after"]
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    del model, ck, table, train, valid
    print(json.dumps(dict(stage="audit_graph_examples", queries=examples_checked,
                          maximum_error=maximum_error, passed=True)), flush=True)
    with np.load(out / "ranks.npz", allow_pickle=False) as saved:
        ranks = {a: saved[a] for a in ("baseline", "candidate_side")}
        standalone = {a: saved[a] for a in ("candidate_max", "candidate_top3", "candidate_pair")}
        for arm in ranks:
            assert ranks[arm].shape == (3, 2*N_TRIPLES)
            np.testing.assert_array_equal(saved[arm+"_coverage"], np.ones((3, 2*N_TRIPLES)))
    for start in range(0, 2*N_TRIPLES, 2048):
        sl = slice(start, start+2048)
        assert np.isfinite(raw[:, sl]).all()
        for j, key in enumerate(("candidate_max", "candidate_top3")):
            np.testing.assert_array_equal(normalized[j, sl], normalize(raw[j, sl]))
            np.testing.assert_array_equal(standalone[key][sl], rank_rows(normalized[j, sl]))
        np.testing.assert_array_equal(pair[sl], (normalized[0, sl]+normalized[1, sl])*np.float32(.5))
        np.testing.assert_array_equal(standalone["candidate_pair"][sl], rank_rows(pair[sl]))
    print(json.dumps(dict(stage="audit_normalization", passed=True)), flush=True)
    features, meta = load_features(receipt)
    relation, direction, family = (meta[k] for k in ("relation", "direction", "family"))
    bc = Path(receipt["prior_bc"])
    with np.load(bc / "ranks.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(ranks["baseline"], saved["half_strength"])
    expected_folds = []
    for i, seed in enumerate(SEEDS):
        coverage = np.zeros(2*N_TRIPLES, np.uint8)
        for fold, fit in enumerate(split_masks(N_TRIPLES, seed)):
            baseline_recipe = json.loads((bc / f"seed{seed}_fold{fold}_half_strength_recipe.json").read_text())
            base = baseline_scores(features["half_strength"], baseline_recipe, relation, direction)
            np.testing.assert_array_equal(rank_rows(base), apply_recipe(features["half_strength"], baseline_recipe, relation, direction))
            saved_recipe = json.loads((out / f"seed{seed}_fold{fold}_beta_recipe.json").read_text())
            recipe = fit_beta(base, pair, fit, relation, family, direction)
            assert recipe == saved_recipe
            treatment = apply_beta(base, pair, recipe, relation, direction, ~fit)
            control = beta_ranks(base, pair, np.flatnonzero(~fit), 0)
            np.testing.assert_array_equal(control, ranks["baseline"][i, ~fit])
            np.testing.assert_array_equal(treatment, ranks["candidate_side"][i, ~fit])
            coverage[~fit] += 1
            beta_counts = {str(beta): sum(group["beta"] == beta for group in recipe["groups"].values()) for beta in BETAS}
            expected_folds.append(dict(seed=seed, fold=fold, fit_triples=int(fit[:N_TRIPLES].sum()),
                report_triples=int((~fit[:N_TRIPLES]).sum()), global_beta=recipe["global_choice"]["beta"],
                directed_group_beta_counts=beta_counts,
                metrics=dict(baseline=metrics(control), candidate_side=metrics(treatment)),
                delta_mrr=float((1/treatment-1/control).mean())))
            del base
            print(json.dumps(dict(stage="audit_fold", seed=seed, fold=fold, passed=True)), flush=True)
        np.testing.assert_array_equal(coverage, np.ones(2*N_TRIPLES))
    assert expected_folds == folds
    for key, value in summarize(ranks).items():
        assert summary[key] == value, key
    assert summary["standalone_full_valid"] == {key: metrics(value) for key, value in standalone.items()}
    expected_opened = ["raw/num-node-dict.csv.gz", "split/random/train.pt", "split/random/valid.pt"]
    assert sorted(opened) == recorded["dataset_files_opened"] == summary["dataset_files_opened"] == expected_opened
    verify_hashes(receipt["hashes"])
    result = dict(audit_passed=True, source_input_artifact_hashes_verified=True, model_unchanged=True,
        sampled_train_neighbor_features_reproduced=True, examples_checked=examples_checked,
        sampled_feature_maximum_error=maximum_error, normalization_and_pair_exact=True,
        all_six_fit_only_beta_recipes_reproduced=True, all_baseline_and_treatment_report_ranks_exact=True,
        coverage_metrics_and_clustered_interval_reproduced=True, test_loaded=False, device="cpu")
    write_json(out / "endpoint_audit.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    audit(parser.parse_args().directory)
