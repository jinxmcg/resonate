"""BC1: fixed half-strength selection, repeated paired VALID partitions, CPU only."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from biokg.compare_feature_pipeline import fit_recipe, metrics, mixture_ranks
from biokg.gradient_diagnostic import sha256
from biokg.retrieval_followups import HalfFeature


SEEDS = (0, 1, 2)
ARMS = ("baseline", "half_strength")
N_TRIPLES = 162886
BOOTSTRAP_SEED = 3631
DATASET = Path("/mnt/geocore/geocore/data_ogb/ogbl_biokg")
CHECKPOINT_SHA = "df7c783e7b8a6110f126bca1c60d89209735accbaa5ffc0b362f2cab52d9d2f1"


def reject_dataset_path(path, dataset=DATASET):
    if Path(path).resolve().is_relative_to(Path(dataset).resolve()):
        raise PermissionError("BC1 uses audited VALID caches only; dataset opens forbidden")


def install_guard():
    def guard(event, arguments):
        if event == "open" and isinstance(arguments[0], (str, bytes)):
            value = arguments[0].decode() if isinstance(arguments[0], bytes) else arguments[0]
            reject_dataset_path(value)
    sys.addaudithook(guard)


def split_masks(n, seed):
    mask = np.random.default_rng(seed).random(n) < .5
    if not mask.any() or mask.all():
        raise ValueError("Need two nonempty halves")
    paired = np.tile(mask, 2)
    return paired, ~paired


def views(base):
    ordinary = [base[j] for j in (3, 4, 5, 6, 7)]
    return dict(baseline=ordinary,
                half_strength=[ordinary[0]] + [HalfFeature(ordinary[0], r) for r in ordinary[1:]])


def report_ranks(features, recipe, relation, direction, report):
    """Only evaluate the report rows; return them in original global-row order."""
    ids = np.flatnonzero(report)
    result = np.full(len(ids), np.nan)
    for rel in np.unique(relation):
        for d in (0, 1):
            local = np.flatnonzero((relation[ids] == rel) & (direction[ids] == d))
            if len(local):
                weights = recipe["groups"][f"{rel}/{d}"]["weights"]
                result[local] = mixture_ranks(features, weights, ids[local])
    if not np.isfinite(result).all():
        raise ValueError("Incomplete report ranks")
    return result


def insert_report(destination, coverage, fit, ranks):
    n = len(fit) // 2
    if len(fit) % 2 or not np.array_equal(fit[:n], fit[n:]):
        raise ValueError("Unpaired fit mask")
    report = ~fit
    if coverage[report].any() or len(ranks) != int(report.sum()):
        raise ValueError("Repeated or incomplete report assignment")
    if not np.isfinite(ranks).all() or (ranks < 1).any() or (ranks > 501).any():
        raise ValueError("Invalid report rank")
    destination[report] = ranks
    coverage[report] += 1


def paired_triple_deltas(treatment, baseline):
    treatment, baseline = np.asarray(treatment), np.asarray(baseline)
    if treatment.ndim != 2 or treatment.shape != baseline.shape or treatment.shape[1] % 2:
        raise ValueError("Expected matching (partitions, 2*triples) arrays")
    if not treatment.size or not np.isfinite(treatment).all() or not np.isfinite(baseline).all():
        raise ValueError("Empty or incomplete ranks")
    if (treatment < 1).any() or (baseline < 1).any() or (treatment > 501).any() or (baseline > 501).any():
        raise ValueError("Invalid ranks")
    delta = 1 / treatment.astype(np.float64) - 1 / baseline.astype(np.float64)
    n = delta.shape[1] // 2
    return ((delta[:, :n] + delta[:, n:]) / 2).mean(axis=0)


def bootstrap(values, seed=BOOTSTRAP_SEED, replicates=2000):
    rng = np.random.default_rng(seed)
    means = [rng.choice(values, len(values), replace=True).mean() for _ in range(replicates)]
    return np.quantile(means, [.025, .975]).tolist()


def summarize(ranks, seeds=SEEDS, bootstrap_seed=BOOTSTRAP_SEED, replicates=2000):
    paired = paired_triple_deltas(ranks["half_strength"], ranks["baseline"])
    if len(seeds) != ranks["baseline"].shape[0]:
        raise ValueError("Partition count mismatch")
    partitions = []
    for i, seed in enumerate(seeds):
        control, treatment = ranks["baseline"][i], ranks["half_strength"][i]
        partitions.append(dict(seed=seed, metrics={a: metrics(ranks[a][i]) for a in ARMS},
            delta_mrr=float((1/treatment - 1/control).mean()),
            recovered_top1=int(((treatment == 1) & (control > 1)).sum()),
            lost_top1=int(((treatment > 1) & (control == 1)).sum())))
    interval = bootstrap(paired, bootstrap_seed, replicates)
    delta = float(paired.mean())
    consistent = interval[0] > 0 and all(p["delta_mrr"] > 0 for p in partitions)
    mean_metrics = {a: {key: float(np.mean([p["metrics"][a][key] for p in partitions]))
                        for key in ("mrr", "hits1", "hits10")} for a in ARMS}
    n = len(paired)
    directions = {name: {a: float((1/ranks[a][:, sl]).mean()) for a in ARMS}
                  for name, sl in (("tail", slice(0, n)), ("head", slice(n, 2*n)))}
    return dict(protocol="BC1", unique_triples=n, unique_directed_queries=2*n,
                partition_seeds=list(seeds), partitions=partitions, mean_metrics=mean_metrics,
                direction_mrr=directions, primary=dict(delta_mrr=delta, bootstrap_95=interval,
                    bootstrap_seed=bootstrap_seed, replicates=replicates, resampling_unit="original triple",
                    repeated_partitions_averaged_within_triple=True, descriptive_interval=True),
                consistent_positive=bool(consistent), useful_gain_flag=bool(consistent and delta >= .001),
                validation_reused=True, model_seed_robustness_tested=False, inference_ensemble=False,
                training_performed=False, model_updates=0, lr=0.0, dataset_splits_loaded=False,
                test_loaded=False, full_valid_refit=False, submission_changed=False)


def verify_hashes(hashes):
    for path, expected in hashes.items():
        if sha256(path) != expected:
            raise ValueError(f"Hash changed: {path}")


def input_receipt(repo):
    fc = repo / "biokg/results/feature_compare/s0_retry1"
    rf = repo / "biokg/results/retrieval_followups/s0"
    old = json.loads((rf / "prerun.json").read_text())
    fc_audit = json.loads((fc / "audit.json").read_text())
    rf_audit = json.loads((rf / "audit.json").read_text())
    for directory in (fc, rf):
        assert json.loads((directory / "endpoint_audit.json").read_text())["audit_passed"]
    checkpoint = repo / "biokg/results/h35f/campaign_s0/single.pt"
    assert old["hashes"][str(checkpoint)] == CHECKPOINT_SHA
    # Check pinned source dependencies but do not reopen dataset or unused models.
    hashes = {path: digest for path, digest in old["hashes"].items()
              if Path(path).suffix in (".py", ".md")}
    hashes[str(checkpoint)] = CHECKPOINT_SHA  # Bytes hashed; never deserialized.
    for directory, audit, names in (
        (fc, fc_audit, ("normalized_features.npy", "metadata.npz", "improved_recipe.json", "ranks.npz")),
        (rf, rf_audit, ("half_strength_recipe.json", "ranks.npz", "summary.json")),
    ):
        for name in names:
            if directory == fc:
                assert audit["artifact_sha256"][name] == old["hashes"][str(directory / name)]
            hashes[str(directory / name)] = audit["artifact_sha256"][name]
        for name in ("prerun.json", "audit.json", "endpoint_audit.json"):
            hashes[str(directory / name)] = (old["hashes"][str(directory / name)]
                                            if directory == fc else sha256(directory / name))
    for name in ("BLEND_CONFIRMATION.md", "confirm_feature_blend.py", "test_blend_confirmation.py",
                 "audit_blend_confirmation.py"):
        hashes[str(repo / "biokg" / name)] = sha256(repo / "biokg" / name)
    verify_hashes(hashes)
    return dict(protocol="BC1", hashes=hashes, prior_fc=str(fc), prior_rf=str(rf),
                checkpoint=str(checkpoint), checkpoint_deserialized=False, device="cpu",
                partition_seeds=list(SEEDS), arms=list(ARMS), interpolation=.5, min_fit_queries=2000,
                bootstrap_seed=BOOTSTRAP_SEED, bootstrap_replicates=2000,
                torch=torch.__version__, numpy=np.__version__, lr=0.0, model_updates=0,
                training_performed=False, test_loaded=False, dataset_splits_loaded=False,
                started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_features(receipt):
    fc = Path(receipt["prior_fc"])
    with np.load(fc / "metadata.npz", allow_pickle=False) as saved:
        metadata = {k: saved[k] for k in ("fit", "relation", "direction", "family")}
    np.testing.assert_array_equal(metadata["fit"], split_masks(N_TRIPLES, 0)[0])
    np.testing.assert_array_equal(metadata["direction"], np.repeat([0, 1], N_TRIPLES))
    np.testing.assert_array_equal(metadata["relation"][:N_TRIPLES], metadata["relation"][N_TRIPLES:])
    np.testing.assert_array_equal(metadata["family"][:N_TRIPLES], metadata["family"][N_TRIPLES:])
    base = np.load(fc / "normalized_features.npy", mmap_mode="r", allow_pickle=False)
    assert base.shape == (8, 2*N_TRIPLES, 501) and base.dtype == np.float32
    for start in range(0, 2*N_TRIPLES, 2048):
        assert np.isfinite(base[3:, start:start+2048]).all()
    return views(base), metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="biokg/results/blend_confirmation/s0")
    args = parser.parse_args()
    torch.set_num_threads(4)
    install_guard()
    repo = Path(__file__).resolve().parents[1]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()

    def log(**event):
        event.update(seconds=time.monotonic()-started, lr=0.0, lr_status="not applicable: cached frozen scores")
        print(json.dumps(event, allow_nan=False), flush=True)
        with (out / "progress.jsonl").open("a") as handle:
            handle.write(json.dumps(event, allow_nan=False) + "\n")

    log(stage="verifying_inputs")
    receipt = input_receipt(repo)
    write_json(out / "prerun.json", receipt)
    features, meta = load_features(receipt)
    relation, direction, family = (meta[k] for k in ("relation", "direction", "family"))
    ranks = {a: np.full((len(SEEDS), 2*N_TRIPLES), np.nan) for a in ARMS}
    coverage = {a: np.zeros((len(SEEDS), 2*N_TRIPLES), np.uint8) for a in ARMS}
    folds, artifacts = [], ["prerun.json"]
    log(stage="inputs_verified", folds_total=6)
    for i, seed in enumerate(SEEDS):
        for fold, fit in enumerate(split_masks(N_TRIPLES, seed)):
            result = dict(seed=seed, fold=fold, fit_triples=int(fit[:N_TRIPLES].sum()),
                          report_triples=int((~fit[:N_TRIPLES]).sum()), metrics={})
            for arm in ARMS:
                log(stage="selection_start", seed=seed, fold=fold, arm=arm)
                recipe = fit_recipe(features[arm], fit, relation, family, direction,
                    log=lambda **event: log(seed=seed, fold=fold, arm=arm, **event))
                filename = f"seed{seed}_fold{fold}_{arm}_recipe.json"
                write_json(out / filename, recipe)
                artifacts.append(filename)
                report = report_ranks(features[arm], recipe, relation, direction, ~fit)
                insert_report(ranks[arm][i], coverage[arm][i], fit, report)
                if seed == 0 and fold == 0:
                    prior = Path(receipt["prior_fc"] if arm == "baseline" else receipt["prior_rf"])
                    name = "improved_recipe.json" if arm == "baseline" else "half_strength_recipe.json"
                    assert recipe == json.loads((prior / name).read_text()), arm
                    with np.load(prior / "ranks.npz", allow_pickle=False) as saved:
                        key = "improved_pipeline" if arm == "baseline" else "half_strength"
                        np.testing.assert_array_equal(report, saved[key][~fit])
                result["metrics"][arm] = metrics(report)
                log(stage="arm_report", seed=seed, fold=fold, arm=arm, report=result["metrics"][arm])
            result["delta_mrr"] = float((1/ranks["half_strength"][i, ~fit] - 1/ranks["baseline"][i, ~fit]).mean())
            folds.append(result)
            write_json(out / "folds.json", folds)
            log(stage="fold_complete", **result)
        assert all((coverage[a][i] == 1).all() for a in ARMS)
    np.savez_compressed(out / "ranks.npz", **ranks, **{a+"_coverage": coverage[a] for a in ARMS})
    summary = summarize(ranks)
    summary["seconds_through_summary"] = time.monotonic()-started
    write_json(out / "summary.json", summary)
    artifacts += ["folds.json", "ranks.npz", "summary.json"]
    verify_hashes(receipt["hashes"])
    write_json(out / "audit.json", dict(source_input_hashes_unchanged=True, original_fold_exact=True,
        coverage_exact=True, model_deserialized=False, dataset_files_opened=[],
        artifact_sha256={name: sha256(out / name) for name in artifacts}))
    log(stage="complete", mean_metrics=summary["mean_metrics"], primary=summary["primary"],
        consistent_positive=summary["consistent_positive"], useful_gain_flag=summary["useful_gain_flag"])


if __name__ == "__main__":
    main()
