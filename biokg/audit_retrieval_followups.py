"""RF1 independent saved-record audit; no dataset split or model deserialization."""

import argparse
import json
from pathlib import Path

import numpy as np

from biokg.compare_feature_pipeline import apply_recipe, fit_recipe, mixture_ranks, paired_masks
from biokg.gradient_diagnostic import sha256
from biokg.relation_analogy import normalize
from biokg.retrieval_followups import ARMS, feature_views, summarize, supported_feature


def audit(directory):
    out = Path(directory)
    receipt = json.loads((out/"prerun.json").read_text())
    recorded = json.loads((out/"audit.json").read_text())
    summary = json.loads((out/"summary.json").read_text())
    prior = Path(receipt["prior"])
    assert receipt["protocol"] == summary["protocol"] == "RF1"
    for path,digest in receipt["hashes"].items():
        if "/data_ogb/" not in path:
            assert sha256(path) == digest,path
    for name,digest in recorded["artifact_sha256"].items():
        assert sha256(out/name) == digest,name
    assert recorded["model_before"] == recorded["model_after"]
    assert recorded["model_unchanged"] and recorded["gradients_absent"]
    assert recorded["baseline_ranks_exact"] and recorded["input_source_hashes_unchanged"]
    assert recorded["dataset_files_opened"] == ["raw/num-node-dict.csv.gz","split/random/train.pt","split/random/valid.pt"]
    assert not receipt["test_loaded"] and not receipt["training_performed"] and receipt["model_updates"] == 0
    assert len(recorded["graph_audit"]) == 102
    assert sum(row["queries"] for row in recorded["graph_audit"]) == 325772
    assert all(row["candidate_permutation_passed"] for row in recorded["graph_audit"])
    with np.load(prior/"metadata.npz",allow_pickle=False) as saved:
        fit,relation,direction,family = (saved[k] for k in ("fit","relation","direction","family"))
    np.testing.assert_array_equal(fit,paired_masks(162886))
    with np.load(out/"ranks.npz",allow_pickle=False) as saved:
        ranks = dict(saved)
    with np.load(prior/"ranks.npz",allow_pickle=False) as saved:
        np.testing.assert_array_equal(ranks["baseline"],saved["improved_pipeline"])
        np.testing.assert_array_equal(ranks["model"],saved["improved_model"])
    base = np.load(prior/"normalized_features.npy",mmap_mode="r",allow_pickle=False)
    raw = np.load(out/"features_and_gates.npy",mmap_mode="r",allow_pickle=False)
    weighted = np.load(out/"weighted_normalized.npy",mmap_mode="r",allow_pickle=False)
    supported = np.load(out/"supported_members.npy",mmap_mode="r",allow_pickle=False)
    assert raw.shape == supported.shape == (4,325772,501) and weighted.shape == (2,325772,501)
    assert raw.dtype == weighted.dtype == supported.dtype == base.dtype == np.float32
    for start in range(0,325772,2048):
        sl = slice(start,start+2048)
        assert np.isfinite(raw[:,sl]).all()
        assert ((raw[2:,sl]>=0)&(raw[2:,sl]<=1)).all()
        for j in range(2):
            np.testing.assert_array_equal(weighted[j,sl],normalize(raw[j,sl]))
        for j,channel in enumerate((4,5,6,7)):
            expected = supported_feature(base[3,sl],base[channel,sl],raw[2+j//2,sl])
            np.testing.assert_array_equal(supported[j,sl],expected)
    print(json.dumps(dict(stage="audit_transforms",passed=True)),flush=True)
    views = feature_views(base,weighted,supported)
    baseline_recipe = json.loads((prior/"improved_recipe.json").read_text())
    np.testing.assert_array_equal(ranks["baseline"],apply_recipe(views["baseline"],baseline_recipe,relation,direction))
    for arm in ARMS:
        recipe = json.loads((out/(arm+"_recipe.json")).read_text())
        regenerated = fit_recipe(views[arm],fit,relation,family,direction,
                     log=lambda **event:print(json.dumps(dict(arm=arm,**event)),flush=True))
        assert recipe == regenerated,arm
        np.testing.assert_array_equal(ranks[arm],apply_recipe(views[arm],recipe,relation,direction))
        np.testing.assert_array_equal(ranks[arm+"_fixed_baseline_recipe"],apply_recipe(views[arm],baseline_recipe,relation,direction))
        np.testing.assert_array_equal(ranks[arm+"_uniform"],
             mixture_ranks(views[arm],np.full(5,.2,np.float32),np.arange(325772)))
        for j,feature in enumerate(views[arm][1:]):
            np.testing.assert_array_equal(ranks[f"{arm}_member{j+1}"],mixture_ranks([feature],[1],np.arange(325772)))
        print(json.dumps(dict(stage="audit_arm",arm=arm,passed=True)),flush=True)
    for key,value in summarize(ranks,fit,family,direction).items():
        assert summary[key] == value,key
    result = dict(audit_passed=True,source_and_artifact_hashes_verified=True,model_unchanged=True,
                  baseline_ranks_exact=True,normalization_and_support_transforms_reproduced=True,
                  fit_only_recipes_reproduced=True,all_saved_ranks_reproduced=True,
                  metrics_and_adjusted_intervals_reproduced=True,dataset_splits_loaded=False)
    (out/"endpoint_audit.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2),flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    audit(parser.parse_args().directory)
