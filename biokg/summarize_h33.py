"""H33 immutable provenance, per-pair receipts and all-seed aggregation."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import statistics


SEEDS = (0, 1, 2)
ARMS = ("baseline", "random")
SOURCES = ("resonate.py", "resonate_wiki.py", "resonate_comp.py", "rowadagrad.py", "biokg/train_biokg_comp.py",
           "biokg/train_positive_filter.py", "biokg/hard_negative.py", "biokg/direction_sampling.py",
           "biokg/analyze_errors.py", "biokg/test_direction_sampling.py", "biokg/test_hard_negative.py",
           "biokg/summarize_h33.py", "biokg/test_h33_report.py", "biokg/scripts/run_h33.sh", "biokg/H33.md",
           "biokg/test_training_progress.py", "biokg/H33_VAST.md", "biokg/run_h33_vast.py")


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def aggregate_results(rows):
    expected = {f"{arm}_s{seed}" for arm in ARMS for seed in SEEDS}
    if set(rows) != expected:
        raise ValueError("All and only the six pre-stated runs must be present")
    scores = {arm: [float(rows[f"{arm}_s{s}"]["overall"]["mrr"]) for s in SEEDS] for arm in ARMS}
    if any(not math.isfinite(v) or not 0 <= v <= 1 for values in scores.values() for v in values):
        raise ValueError("Invalid MRR")
    gains = [a - b for a, b in zip(scores["random"], scores["baseline"])]
    means = {arm: dict(seeds=list(SEEDS), mrr=values, mean=statistics.mean(values),
                       sample_std=statistics.stdev(values)) for arm, values in scores.items()}
    mean_gain = statistics.mean(gains)
    return dict(arms=means, paired_gains=gains, mean_paired_gain=mean_gain,
                sample_std_paired_gain=statistics.stdev(gains),
                followup_gate=mean_gain >= .002 and all(x > 0 for x in gains),
                caveat="Three-seed compute-allocation gate; not a significance test or distilled-model result")


def common_args(args):
    return {k: v for k, v in args.items() if k not in ("save", "seed", "mining_mode")}


def collect_pair(run, diag_prefix, seed):
    import torch
    torch.set_num_threads(4)
    rows, reference_args, reference_keys = {}, None, None
    fixed = dict(steps=50000, batch=2048, neg=4096, k=12, block_size=4,
                 shell="sparse", table_dtype="fp32", table_lr=.3, lr=.005, sched="cosine", lam=.1,
                 direction_sampling="uniform", mining_count=64, mining_weight=.1,
                 mining_warmup=2500, mining_ramp=1250, eval="valid", probe_every=5000, probe_size=5000,
                 filter_train_positives=False, low_rank=0, distill=[], peers=1,
                 compose=0., comp=0, aux_rp=0., n3=0.)
    for arm in ARMS:
        name = f"{arm}_s{seed}"
        path = run / name / "model.pt"
        ck = torch.load(path, map_location="cpu", weights_only=False)
        ca = ck["args"]
        expected_mining = "random" if arm == "random" else "none"
        if any(ca.get(k) != v for k, v in fixed.items()) or ca["seed"] != seed or ca["mining_mode"] != expected_mining:
            raise RuntimeError(f"Non-matching preregistered recipe: {name}")
        if ck["direction_sampling"]["mode"] != "uniform" or ck["train_positive_filter"]["enabled"]:
            raise RuntimeError(f"Unexpected skew or full mask: {name}")
        if ck["negative_mining"]["mode"] != expected_mining:
            raise RuntimeError(f"Incorrect auxiliary metadata: {name}")
        count = sum(v.numel() * (2 if v.is_complex() else 1) for v in ck["model"].values())
        if count != 27124129 or not all(torch.isfinite(v).all() for v in ck["model"].values()):
            raise RuntimeError(f"Nonfinite or unexpected model: {name}")
        if reference_args is None:
            reference_args, reference_keys = common_args(ca), list(ck["model"])
        if common_args(ca) != reference_args or list(ck["model"]) != reference_keys:
            raise RuntimeError(f"Unmatched pair: {name}")
        diagnostic_path = Path(f"{diag_prefix}_{name}") / "summary.json"
        diag = json.loads(diagnostic_path.read_text())
        checkpoint_hash = sha256(path)
        if diag["checkpoint_sha256"] != checkpoint_hash or diag["checkpoint_args"] != ca:
            raise RuntimeError(f"Diagnostic provenance mismatch: {name}")
        if diag["overall"]["queries"] != 325772 or diag["negative_candidates_per_query"] != 500:
            raise RuntimeError(f"Unexpected evaluation scope: {name}")
        log = (run / name / "train.log").read_text()
        elapsed = re.search(r"step 50000/50000\s+loss .*?\((\d+)s\)", log)
        if not elapsed or "[test]" in log:
            raise RuntimeError(f"Incomplete or test-evaluated log: {name}")
        mining = ck["negative_mining"]
        if arm == "random" and (mining["candidate_slots"] != 6225920000 or
                not 0 <= mining["selected_count"] <= mining["candidate_slots"] or not mining["train_sha256"]):
            raise RuntimeError(f"Invalid auxiliary exposure: {name}")
        rows[name] = dict(checkpoint=str(path.resolve()), checkpoint_sha256=checkpoint_hash,
                          args=ca, model_keys=reference_keys, parameter_count=count,
                          diagnostic=str(diagnostic_path.resolve()), overall=diag["overall"], groups=diag["groups"],
                          checks=diag["checks"], training_seconds=int(elapsed.group(1)),
                          negative_mining=mining, direction_sampling=ck["direction_sampling"])
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("initialize", "check", "pair", "aggregate"), required=True)
    p.add_argument("--seed", type=int, choices=SEEDS)
    p.add_argument("--run-root", required=True)
    p.add_argument("--report-root", required=True)
    p.add_argument("--diag-prefix", required=True)
    args = p.parse_args()
    run, report = Path(args.run_root), Path(args.report_root)
    hashes = {path: sha256(path) for path in SOURCES}
    paths = dict(run_root=str(run.resolve()), diag_prefix=str(Path(args.diag_prefix).resolve()))
    if args.phase == "initialize":
        import torch
        if run.exists() or report.exists():
            p.error("Run/report root already exists")
        if shutil.disk_usage(Path.cwd()).free < 1100 * 2**20:
            p.error("Less than 1100 MiB free; refusing to start six-run campaign")
        report.mkdir(parents=True, exist_ok=False)
        write_json(report / "prerun.json", dict(
            started_utc=datetime.now(timezone.utc).isoformat(), source_sha256=hashes, **paths,
            torch_version=torch.__version__, cuda_version=torch.version.cuda,
            protocol="Fresh train-only updates; uniform directions; final validation only; no test"))
        return
    prereg = json.loads((report / "prerun.json").read_text())
    if prereg["source_sha256"] != hashes or any(prereg[k] != v for k, v in paths.items()):
        raise RuntimeError("Frozen source/preregistration or paths changed")
    if args.phase == "check":
        if shutil.disk_usage(run).free < 450 * 2**20:
            raise RuntimeError("Less than 450 MiB free; stopping before the next training/diagnostic operation")
        return
    if args.phase == "pair":
        if args.seed is None:
            p.error("Pair phase requires --seed")
        destination = report / f"seed_{args.seed}.json"
        if destination.exists() or any((report / f"{a}_s{args.seed}.log").exists() for a in ARMS):
            p.error("Pair receipt/logs already exist")
        rows = collect_pair(run, args.diag_prefix, args.seed)
        gain = rows[f"random_s{args.seed}"]["overall"]["mrr"] - rows[f"baseline_s{args.seed}"]["overall"]["mrr"]
        write_json(destination, dict(completed_utc=datetime.now(timezone.utc).isoformat(), seed=args.seed,
                                    paired_gain=gain, runs=rows))
        for arm in ARMS:
            name = f"{arm}_s{args.seed}"
            shutil.copyfile(run / name / "train.log", report / f"{name}.log")
        print(json.dumps(dict(seed=args.seed, mrr={k: v["overall"]["mrr"] for k, v in rows.items()}, paired_gain=gain)))
        return
    rows = {}
    for seed in SEEDS:
        pair = json.loads((report / f"seed_{seed}.json").read_text())
        if pair["seed"] != seed or set(pair["runs"]) != {f"{a}_s{seed}" for a in ARMS}:
            raise RuntimeError("Invalid pair receipt")
        rows.update(pair["runs"])
    ref = rows["baseline_s0"]
    for name, row in rows.items():
        if common_args(row["args"]) != common_args(ref["args"]) or row["model_keys"] != ref["model_keys"]:
            raise RuntimeError(f"Cross-seed configuration mismatch: {name}")
        if sha256(row["checkpoint"]) != row["checkpoint_sha256"]:
            raise RuntimeError(f"Checkpoint changed after pair collection: {name}")
    graph_hashes = {rows[f"random_s{s}"]["negative_mining"]["train_sha256"] for s in SEEDS}
    if len(graph_hashes) != 1:
        raise RuntimeError("Different training graphs across seeds")
    summary = aggregate_results(rows)
    write_json(report / "confirmation.json", dict(completed_utc=datetime.now(timezone.utc).isoformat(),
                                                  prerun=prereg, summary=summary, runs=rows))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
