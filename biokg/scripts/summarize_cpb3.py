"""Entry-2 statistics from the CP-B3 receipts (COMPACT_B.md).

    python scripts/summarize_cpb3.py            # results/cpb3
    python scripts/summarize_cpb3.py runs       # a directory you produced

The compact distilled row: the released T=2 student's entity table replaced by
degree-tiered coefficients over per-tier subspace banks (k-means + per-cluster
PCA, no training), blended with the same four label-free retrieval members.
Prints test MRR mean +/- unbiased std (ddof=1) over the seeds present, the
matching validation numbers, and the comparison against row C's committed
receipts (results/sparse) at the same seeds.
"""
import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TIERED_PARAMS = 9_555_497
WIDE_PARAMS = 27_124_129


def grab(path, pattern, cast=float):
    with open(path) as f:
        m = re.findall(pattern, f.read())
    return cast(m[-1]) if m else None


def stat(v):
    v = np.asarray(v, dtype=np.float64)
    if len(v) < 2:
        return f"{v.mean():.4f} (n=1)"
    return f"{v.mean():.4f} +/- {v.std(ddof=1):.4f}"


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "results", "cpb3")
    print(f"receipts: {os.path.abspath(d)}")
    seeds = sorted(int(re.search(r"_s(\d+)\.json$", f).group(1))
                   for f in glob.glob(f"{d}/committed_cpb3_s*.json"))
    if not seeds:
        raise SystemExit(f"no committed_cpb3_s*.json in {d}")
    if len(seeds) < 10:
        print(f"NOTE: {len(seeds)}/10 seeds present ({seeds}) - not a final number")

    test = [json.load(open(f"{d}/committed_cpb3_s{s}.json"))["test_mrr"] for s in seeds]
    held = [grab(f"{d}/pure_cpb3_s{s}.log", r"HELD-OUT ([0-9.]+)") for s in seeds]
    insample = [grab(f"{d}/committed_cpb3_s{s}.log", r"in-sample\): ([0-9.]+)") for s in seeds]
    alone = [grab(f"{d}/pure_cpb3_s{s}.log", r"model-only \(z-mean\).*held-out ([0-9.]+)") for s in seeds]
    hits = {k: [grab(f"{d}/committed_cpb3_s{s}.log", rf"hits@{k}: ([0-9.]+)") for s in seeds]
            for k in (1, 3, 10)}

    print(f"\nentry 2. compact distilled (tiered table) + retrieval features  "
          f"[{TIERED_PARAMS:,} params]")
    print(f"  test  MRR  {stat(test)}")
    print(f"  valid MRR  {stat(insample)} in-sample / {stat(held)} held-out")
    print(f"  model alone (held-out, no members)  {stat(alone)}")
    print("  test hits@1/3/10  " + "  ".join(f"{np.mean(hits[k]):.4f}" for k in (1, 3, 10)))
    print("  per seed: " + " ".join(f"{x:.4f}" for x in test))
    gap = np.array(held) - np.array(test)
    print(f"  held-out - test: mean {gap.mean():+.4f}, max |gap| {np.abs(gap).max():.4f}")

    # the same seeds of row C, for the compression cost
    ref = os.path.join(HERE, "..", "results", "sparse")
    have = [s for s in seeds if os.path.exists(f"{ref}/committed_distT2_s{s}.json")]
    if have:
        ct = [json.load(open(f"{ref}/committed_distT2_s{s}.json"))["test_mrr"] for s in have]
        ch = [grab(f"{ref}/pure_distT2_s{s}.log", r"HELD-OUT ([0-9.]+)") for s in have]
        ca = [grab(f"{ref}/pure_distT2_s{s}.log", r"model-only \(z-mean\).*held-out ([0-9.]+)") for s in have]
        mine = [t for s, t in zip(seeds, test) if s in have]
        mh = [h for s, h in zip(seeds, held) if s in have]
        ma = [a for s, a in zip(seeds, alone) if s in have]
        print(f"\nrow C. distilled + retrieval features  [{WIDE_PARAMS:,} params], same {len(have)} seeds")
        print(f"  test  MRR  {stat(ct)}")
        print(f"  valid MRR  {stat(ch)} held-out")
        print(f"  model alone (held-out)  {stat(ca)}")
        d_alone = np.mean(ca) - np.mean(ma)
        d_blend = np.mean(ct) - np.mean(mine)
        print(f"\ncompression cost at {TIERED_PARAMS/WIDE_PARAMS*100:.1f}% of the parameters "
              f"({WIDE_PARAMS/TIERED_PARAMS:.2f}x smaller):")
        print(f"  model alone   {d_alone:+.4f}")
        print(f"  blended test  {d_blend:+.4f}   (held-out {np.mean(ch)-np.mean(mh):+.4f})")
        if d_alone > 1e-9:
            print(f"  the members absorb {100*(1-d_blend/d_alone):.0f}% of the loss")


if __name__ == "__main__":
    main()
