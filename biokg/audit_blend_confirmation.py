"""BC1 saved-record audit: independently refit recipes and regenerate rankings."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from biokg.compare_feature_pipeline import apply_recipe, fit_recipe, metrics
from biokg.confirm_feature_blend import (
    ARMS, N_TRIPLES, SEEDS, install_guard, load_features, split_masks,
    summarize, verify_hashes, write_json,
)


def audit(directory):
    torch.set_num_threads(4)
    install_guard()
    out = Path(directory)
    receipt = json.loads((out / "prerun.json").read_text())
    recorded = json.loads((out / "audit.json").read_text())
    summary = json.loads((out / "summary.json").read_text())
    folds = json.loads((out / "folds.json").read_text())
    assert receipt["protocol"] == summary["protocol"] == "BC1"
    assert receipt["partition_seeds"] == list(SEEDS) and receipt["interpolation"] == .5
    assert not receipt["checkpoint_deserialized"] and not receipt["dataset_splits_loaded"]
    assert not receipt["training_performed"] and not receipt["test_loaded"] and receipt["model_updates"] == 0
    assert recorded["source_input_hashes_unchanged"] and recorded["original_fold_exact"]
    assert recorded["dataset_files_opened"] == [] and not recorded["model_deserialized"]
    verify_hashes(receipt["hashes"])
    verify_hashes({str(out / name): digest for name, digest in recorded["artifact_sha256"].items()})
    features, meta = load_features(receipt)
    relation, direction, family = (meta[k] for k in ("relation", "direction", "family"))
    with np.load(out / "ranks.npz", allow_pickle=False) as saved:
        ranks = {a: saved[a] for a in ARMS}
        for arm in ARMS:
            assert ranks[arm].shape == (3, 2*N_TRIPLES)
            np.testing.assert_array_equal(saved[arm+"_coverage"], np.ones((3, 2*N_TRIPLES)))
    expected_folds = []
    for i, seed in enumerate(SEEDS):
        coverage = np.zeros(2*N_TRIPLES, np.uint8)
        for fold, fit in enumerate(split_masks(N_TRIPLES, seed)):
            coverage[~fit] += 1
            result = dict(seed=seed, fold=fold, fit_triples=int(fit[:N_TRIPLES].sum()),
                          report_triples=int((~fit[:N_TRIPLES]).sum()), metrics={})
            for arm in ARMS:
                recipe = json.loads((out / f"seed{seed}_fold{fold}_{arm}_recipe.json").read_text())
                regenerated = fit_recipe(features[arm], fit, relation, family, direction,
                    log=lambda **event: print(json.dumps(dict(seed=seed, fold=fold, arm=arm, **event)), flush=True))
                assert recipe == regenerated, (seed, fold, arm)
                # Use the old independent full-row router, not BC1's report-only router.
                rebuilt = apply_recipe(features[arm], regenerated, relation, direction)[~fit]
                np.testing.assert_array_equal(rebuilt, ranks[arm][i, ~fit])
                result["metrics"][arm] = metrics(rebuilt)
                if seed == 0 and fold == 0:
                    prior = Path(receipt["prior_fc"] if arm == "baseline" else receipt["prior_rf"])
                    name = "improved_recipe.json" if arm == "baseline" else "half_strength_recipe.json"
                    assert recipe == json.loads((prior / name).read_text())
                    with np.load(prior / "ranks.npz", allow_pickle=False) as saved:
                        key = "improved_pipeline" if arm == "baseline" else "half_strength"
                        np.testing.assert_array_equal(rebuilt, saved[key][~fit])
            result["delta_mrr"] = float((1/ranks["half_strength"][i, ~fit] - 1/ranks["baseline"][i, ~fit]).mean())
            expected_folds.append(result)
            print(json.dumps(dict(stage="audit_fold", seed=seed, fold=fold, passed=True)), flush=True)
        np.testing.assert_array_equal(coverage, np.ones(2*N_TRIPLES))
    assert folds == expected_folds
    regenerated = summarize(ranks)
    for key, value in regenerated.items():
        assert summary[key] == value, key
    # Independent grouping check: six reciprocal-rank differences per original triple.
    rr_delta = (1/ranks["half_strength"] - 1/ranks["baseline"]).reshape(3, 2, N_TRIPLES)
    clustered = rr_delta.sum(axis=(0, 1)) / 6
    np.testing.assert_allclose(clustered.mean(), summary["primary"]["delta_mrr"], atol=1e-16, rtol=0)
    verify_hashes(receipt["hashes"])
    result = dict(audit_passed=True, source_input_artifact_hashes_verified=True,
                  all_twelve_fit_only_recipes_reproduced=True, all_report_ranks_exact=True,
                  original_fc1_rf1_fold_exact=True, exactly_once_coverage_per_partition=True,
                  all_metrics_and_clustered_interval_reproduced=True,
                  dataset_splits_loaded=False, models_deserialized=False, test_loaded=False)
    write_json(out / "endpoint_audit.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    audit(parser.parse_args().directory)
