"""Reproduce FC1 recipes and metrics from saved VALID features; no split loading."""

import argparse
import json
from pathlib import Path

import numpy as np

from biokg.compare_feature_pipeline import (
    CHANNELS, VIEWS, apply_recipe, fit_recipe, mixture_ranks, paired_masks, summarize,
)
from biokg.gradient_diagnostic import sha256
from biokg.relation_analogy import normalize, rank_rows


def audit(directory):
    out = Path(directory)
    receipt = json.loads((out / "prerun.json").read_text())
    recorded = json.loads((out / "audit.json").read_text())
    summary = json.loads((out / "summary.json").read_text())
    assert receipt["protocol"] == summary["protocol"] == "FC1"
    for path, digest in receipt["hashes"].items():
        if "/data_ogb/" not in path:
            assert sha256(path) == digest, path
    for name, digest in recorded["artifact_sha256"].items():
        assert sha256(out / name) == digest, name
    assert recorded["models_before"] == recorded["models_after"]
    assert recorded["models_unchanged"] and recorded["gradients_absent"]
    assert recorded["dataset_files_opened"] == ["raw/num-node-dict.csv.gz", "split/random/train.pt", "split/random/valid.pt"]
    assert not receipt["test_loaded"] and not receipt["training_performed"]
    with np.load(out / "metadata.npz", allow_pickle=False) as saved:
        fit, relation, direction, family = (saved[k] for k in ("fit", "relation", "direction", "family"))
    with np.load(out / "ranks.npz", allow_pickle=False) as saved:
        ranks = dict(saved)
    n = len(fit) // 2
    assert n == 162886
    np.testing.assert_array_equal(fit, paired_masks(n))
    np.testing.assert_array_equal(relation[:n], relation[n:])
    np.testing.assert_array_equal(direction, np.repeat([0, 1], n))
    z = np.load(out / "normalized_features.npy", mmap_mode="r", allow_pickle=False)
    raw = np.load(out / "raw_features.npy", mmap_mode="r", allow_pickle=False)
    assert raw.shape == z.shape == (8, 2*n, 501) and raw.dtype == z.dtype == np.float32
    for j, name in enumerate(CHANNELS):
        for start in range(0, 2*n, 2048):
            sl = slice(start, start + 2048)
            np.testing.assert_array_equal(z[j, sl], normalize(raw[j, sl]))
            np.testing.assert_array_equal(ranks["feature_" + name][sl], rank_rows(raw[j, sl]))
        print(json.dumps(dict(stage="audit_feature", feature=name)), flush=True)
    recipes = {}
    for arm in ("original", "improved"):
        features = [z[j] for j in VIEWS[arm]]
        recipes[arm] = json.loads((out / f"{arm}_recipe.json").read_text())
        regenerated = fit_recipe(features, fit, relation, family, direction, min_rows=receipt["min_rows"],
                                  log=lambda **event: print(json.dumps(dict(arm=arm, **event)), flush=True))
        assert recipes[arm] == regenerated
        np.testing.assert_array_equal(ranks[arm + "_pipeline"], apply_recipe(features, recipes[arm], relation, direction))
        np.testing.assert_array_equal(ranks[arm + "_uniform"],
                                       mixture_ranks(features, np.full(5, .2, np.float32), np.arange(2*n)))
        np.testing.assert_array_equal(ranks[arm + "_model"], ranks["feature_" + CHANNELS[VIEWS[arm][0]]])
    for key, view in (("score_only_swap", "score_only_swap"), ("analogy_only_swap", "analogy_only_swap"),
                      ("both_swap_original_weights", "improved")):
        actual = apply_recipe([z[j] for j in VIEWS[view]], recipes["original"], relation, direction)
        np.testing.assert_array_equal(actual, ranks[key])
    for key, value in summarize(ranks, fit, family, direction).items():
        assert summary[key] == value, key
    result = dict(audit_passed=True, source_and_artifact_hashes_verified=True, models_unchanged=True,
                  normalization_and_feature_ranks_reproduced=True, fit_only_recipes_reproduced=True,
                  pipeline_and_crossed_diagnostic_ranks_reproduced=True, metrics_and_intervals_reproduced=True,
                  dataset_splits_loaded=False)
    (out / "endpoint_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    audit(parser.parse_args().directory)
