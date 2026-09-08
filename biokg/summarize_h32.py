"""Freeze H32 inputs, then collect all four results without selecting runs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil

import torch


ARMS = ("baseline", "random", "skew", "random_skew")
SOURCES = ("resonate.py", "resonate_wiki.py", "rowadagrad.py", "biokg/train_biokg_comp.py",
           "biokg/train_positive_filter.py", "biokg/hard_negative.py", "biokg/direction_sampling.py",
           "biokg/analyze_errors.py", "biokg/test_direction_sampling.py", "biokg/test_hard_negative.py",
           "biokg/summarize_h32.py", "biokg/scripts/run_h32.sh", "biokg/H32.md")


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--initialize", action="store_true")
    p.add_argument("--run-root", required=True)
    p.add_argument("--report-root", required=True)
    p.add_argument("--diag-prefix", required=True)
    args = p.parse_args()
    run, report = Path(args.run_root), Path(args.report_root)
    hashes = {path: sha256(path) for path in SOURCES}
    if args.initialize:
        if run.exists():
            p.error("Run root already exists")
        # Preserve room for four models, a rolling checkpoint and diagnostics.
        if shutil.disk_usage(Path.cwd()).free < 1024 ** 3:
            p.error("Less than 1 GiB free; refusing to start H32")
        report.mkdir(parents=True, exist_ok=False)
        write_json(report / "prerun.json", dict(
            started_utc=datetime.now(timezone.utc).isoformat(), source_sha256=hashes,
            run_root=str(run.resolve()), diag_prefix=str(Path(args.diag_prefix).resolve()),
            torch_version=torch.__version__, cuda_version=torch.version.cuda,
            protocol="train-only updates and direction policy; validation evaluation; no test"))
        return
    prereg = json.loads((report / "prerun.json").read_text())
    if prereg["source_sha256"] != hashes:
        raise RuntimeError("H32 source/preregistration changed after initialization")
    if prereg["run_root"] != str(run.resolve()) or prereg["diag_prefix"] != str(Path(args.diag_prefix).resolve()):
        raise RuntimeError("Result paths differ from preregistration")
    rows, base_args, state_keys = {}, None, None
    for arm in ARMS:
        ckpath = run / arm / "model.pt"
        ck = torch.load(ckpath, map_location="cpu", weights_only=False)
        ca = ck["args"]
        expected_mining = "random" if arm in ("random", "random_skew") else "none"
        expected_direction = "train-fanout" if arm in ("skew", "random_skew") else "uniform"
        if ca["mining_mode"] != expected_mining or ca["direction_sampling"] != expected_direction:
            raise RuntimeError(f"Wrong arm configuration: {arm}")
        if ca["eval"] != "valid" or ca["steps"] != 12500 or ca["seed"] != 0 or ck["train_positive_filter"]["enabled"]:
            raise RuntimeError(f"Wrong protocol: {arm}")
        common = {k: v for k, v in ca.items() if k not in ("save", "mining_mode", "direction_sampling")}
        if base_args is None:
            base_args, state_keys = common, list(ck["model"])
        if common != base_args or list(ck["model"]) != state_keys:
            raise RuntimeError(f"Unmatched common configuration/model keys: {arm}")
        if not all(torch.isfinite(v).all() for v in ck["model"].values()):
            raise RuntimeError(f"Nonfinite checkpoint: {arm}")
        summary_path = Path(f"{args.diag_prefix}_{arm}_s0") / "summary.json"
        diag = json.loads(summary_path.read_text())
        checkpoint_hash = sha256(ckpath)
        if diag["checkpoint_sha256"] != checkpoint_hash:
            raise RuntimeError(f"Diagnostic/checkpoint hash mismatch: {arm}")
        log = (run / arm / "train.log").read_text()
        elapsed = re.search(r"step 12500/12500\s+loss .*?\((\d+)s\)", log)
        if elapsed is None:
            raise RuntimeError(f"Incomplete training log: {arm}")
        control_match = None
        oldpath = Path("biokg/runs/h31") / arm / "model.pt"
        if arm in ("baseline", "random") and oldpath.exists():
            old = torch.load(oldpath, map_location="cpu", weights_only=False)["model"]
            control_match = old.keys() == ck["model"].keys() and all(torch.equal(old[k], ck["model"][k]) for k in old)
            del old
        rows[arm] = dict(checkpoint=str(ckpath.resolve()), checkpoint_sha256=checkpoint_hash,
                         diagnostic=str(summary_path.resolve()), overall=diag["overall"],
                         directions=diag["groups"]["directions"], families=diag["groups"]["families"],
                         relation_directions=diag["groups"]["relation_directions"], checks=diag["checks"],
                         training_seconds=int(elapsed.group(1)), h31_model_bitwise_equal=control_match,
                         parameter_count=sum(v.numel() * (2 if v.is_complex() else 1) for v in ck["model"].values()),
                         negative_mining=ck["negative_mining"], direction_sampling=ck["direction_sampling"])
        shutil.copyfile(run / arm / "train.log", report / f"{arm}_s0.log")
        del ck
    m = {arm: row["overall"]["mrr"] for arm, row in rows.items()}
    effects = dict(auxiliary_alone=m["random"] - m["baseline"], skew_alone=m["skew"] - m["baseline"],
                   skew_on_auxiliary=m["random_skew"] - m["random"])
    effects["interaction"] = effects["skew_on_auxiliary"] - effects["skew_alone"]
    result = dict(completed_utc=datetime.now(timezone.utc).isoformat(), prerun=prereg,
                  common_args=base_args, arms=rows, effects=effects,
                  screen_gates=dict(combined=effects["skew_on_auxiliary"] >= .003 and effects["skew_alone"] > 0,
                                    skew_alone=effects["skew_alone"] >= .003),
                  caveat="Single-seed early screen, not full-training or distilled-model confirmation")
    write_json(report / "screen_s0.json", result)
    print(json.dumps(dict(mrr=m, effects=effects, screen_gates=result["screen_gates"]), indent=2))


if __name__ == "__main__":
    main()
