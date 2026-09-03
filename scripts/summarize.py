"""Leaderboard statistics from the run receipts.

    python scripts/summarize.py            # reads results/ (the shipped receipts)
    python scripts/summarize.py runs       # or a directory you produced

Prints, per row, test MRR mean +/- unbiased std (torch.std convention,
ddof=1) over the ten seeds, the matching validation number, and the
per-seed values, read from:
  final_seed{s}.log         row A  [valid]/[test] MRR lines (train_ogb.py)
  committed_single_s{s}.json  row B  test_mrr (freeze_test.py)
  committed_single_s{s}.log         'frozen weights on full valid (in-sample)'
  pure_s{s}.log                     'HELD-OUT' (ensemble_weights.py)
  committed_dist_s{s}.json / .log, pure_dist_s{s}.log, dist27_s{s}.log  row C
"""

import glob
import json
import os
import re
import sys

import numpy as np

SEEDS = range(10)


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


def row(name, test, valid, extra=None):
    print(f"\n{name}")
    print(f"  test  MRR  {stat(test)}")
    print(f"  valid MRR  {stat(valid)}")
    if extra:
        for lab, vals in extra.items():
            print(f"  {lab:<10} {stat(vals)}")
    print("  per seed: " + " ".join(f"{x:.4f}" for x in test))


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(os.path.dirname(__file__), "..", "results")

    # row A: single model, both numbers from the trainer's final eval
    test = [grab(f"{d}/final_seed{s}.log", r"\[test\] MRR ([0-9.]+)")
            for s in SEEDS]
    valid = [grab(f"{d}/final_seed{s}.log", r"\[valid\] MRR ([0-9.]+)")
             for s in SEEDS]
    row("A. single 27M model (10 seeds)", test, valid)

    for tag, label, alone in (
            ("single", "B. 27M model + retrieval features", None),
            ("dist", "C. distilled 27M model + retrieval features",
             "dist27_s{s}.log")):
        if not glob.glob(f"{d}/committed_{tag}_s*.json"):
            continue
        test = [json.load(open(f"{d}/committed_{tag}_s{s}.json"))["test_mrr"]
                for s in SEEDS]
        valid = [grab(f"{d}/committed_{tag}_s{s}.log",
                      r"in-sample\): ([0-9.]+)") for s in SEEDS]
        pure = "pure_s{s}.log" if tag == "single" else "pure_dist_s{s}.log"
        held = [grab(f"{d}/{pure.format(s=s)}", r"HELD-OUT ([0-9.]+)")
                for s in SEEDS]
        extra = {"held-out": held}
        if alone:
            extra["alone(val)"] = [grab(f"{d}/{alone.format(s=s)}",
                                        r"\[valid\] MRR ([0-9.]+)")
                                   for s in SEEDS]
        row(label, test, valid, extra)
        # a blend fit on full valid must not beat its held-out estimate
        # by more than noise; print the gap so drift is visible
        gap = np.array(held) - np.array(test)
        print(f"  held-out - test: mean {gap.mean():+.4f}, "
              f"max |gap| {np.abs(gap).max():.4f}")


if __name__ == "__main__":
    main()
