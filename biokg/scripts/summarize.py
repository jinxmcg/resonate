"""Leaderboard statistics from the run receipts, for either biokg ladder.
    python scripts/summarize.py             # results/sparse  (the submitted ladder)
    python scripts/summarize.py dense       # results/dense   (the first ladder, dense Adam shell)
    python scripts/summarize.py runs        # any directory you produced
Prints, per row, test MRR mean +/- unbiased std (torch.std convention, ddof=1)
over the ten seeds, the matching validation number(s) and the per-seed values.
File names per ladder:
  dense   final_seed{s}.log | committed_single_s{s}.json/.log, pure_s{s}.log
          | committed_dist_s{s}.json/.log, pure_dist_s{s}.log, dist27_s{s}.log
  sparse  h24_sparse_s{s}.log | committed_single_s{s}.json/.log, pure_single_s{s}.log
          | committed_dist_s{s}.json/.log, pure_dist_s{s}.log, h25_dist_s{s}.log   (T=1 students)
          | committed_distT2_s{s}.json/.log, pure_distT2_s{s}.log, h26_dist_T2_s{s}.log (T=2, submitted)
"""
import glob
import json
import os
import re
import sys
import numpy as np

SEEDS = range(10)
HERE = os.path.dirname(os.path.abspath(__file__))
LADDERS = {
    "dense": dict(final="final_seed{s}.log", pure_single="pure_s{s}.log",
                  rows=[("single", "B. 27M model + retrieval features", "pure_s{s}.log", None),
                        ("dist", "C. distilled 27M model (T=1) + retrieval features", "pure_dist_s{s}.log", "dist27_s{s}.log")]),
    "sparse": dict(final="h24_sparse_s{s}.log",
                   rows=[("single", "B. 27M model + retrieval features", "pure_single_s{s}.log", None),
                         ("dist", "C(T=1). distilled 27M model + retrieval features [ablation]", "pure_dist_s{s}.log", "h25_dist_s{s}.log"),
                         ("distT2", "C. distilled 27M model (T=2) + retrieval features [submitted]", "pure_distT2_s{s}.log", "h26_dist_T2_s{s}.log")]),
}


def grab(path, pattern):
    with open(path) as f:
        txt = f.read()
    m = re.findall(pattern, txt)
    if not m:
        raise SystemExit(f"{path}: no match for {pattern!r}")
    return float(m[-1])


def stat(vals):
    v = np.array(vals, dtype=np.float64)
    return f"{v.mean():.4f} +/- {v.std(ddof=1):.4f}"


def row(name, test, valid, hits, extra=None):
    print(f"\n{name}")
    print(f"  test  MRR  {stat(test)}")
    print(f"  valid MRR  {stat(valid)}")
    if extra:
        for lab, vals in extra.items():
            print(f"  {lab:<10} {stat(vals)}")
    print("  test hits@1/3/10  " + "  ".join(f"{np.mean(hits[k]):.4f}" for k in (1, 3, 10)))
    print("  per seed: " + " ".join(f"{x:.4f}" for x in test))


def hits_at(paths, fmt):
    return {k: [grab(p, fmt.format(k=k)) for p in paths] for k in (1, 3, 10)}


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "sparse"
    if arg in LADDERS:
        d, cfg = os.path.join(HERE, "..", "results", arg), LADDERS[arg]
    else:
        d = arg
        cfg = LADDERS["sparse"] if glob.glob(f"{d}/h24_sparse_s*.log") else LADDERS["dense"]
    print(f"receipts: {os.path.abspath(d)}")
    final = cfg["final"]
    test = [grab(f"{d}/{final.format(s=s)}", r"\[test\] MRR ([0-9.]+)") for s in SEEDS]
    valid = [grab(f"{d}/{final.format(s=s)}", r"\[valid\] MRR ([0-9.]+)") for s in SEEDS]
    hits = hits_at([f"{d}/{final.format(s=s)}" for s in SEEDS], r"\[test\] MRR [0-9.]+.*?hits@{k} ([0-9.]+)")
    row("A. single 27M model (10 seeds)", test, valid, hits)
    for tag, label, pure, alone in cfg["rows"]:
        have = glob.glob(f"{d}/committed_{tag}_s*.json")
        if not have:
            continue
        if len(have) < len(SEEDS):
            print(f"\n{label}: {len(have)}/{len(SEEDS)} seeds present, skipped")
            continue
        test = [json.load(open(f"{d}/committed_{tag}_s{s}.json"))["test_mrr"] for s in SEEDS]
        valid = [grab(f"{d}/committed_{tag}_s{s}.log", r"in-sample\): ([0-9.]+)") for s in SEEDS]
        held = [grab(f"{d}/{pure.format(s=s)}", r"HELD-OUT ([0-9.]+)") for s in SEEDS]
        extra = {"held-out": held}
        if alone and glob.glob(f"{d}/{alone.format(s='*')}"):
            extra["alone(val)"] = [grab(f"{d}/{alone.format(s=s)}", r"\[valid\] MRR ([0-9.]+)") for s in SEEDS]
        hits = hits_at([f"{d}/committed_{tag}_s{s}.log" for s in SEEDS], r"hits@{k}: ([0-9.]+)")
        row(label, test, valid, hits, extra)
        gap = np.array(held) - np.array(test)
        print(f"  held-out - test: mean {gap.mean():+.4f}, max |gap| {np.abs(gap).max():.4f}")


if __name__ == "__main__":
    main()
