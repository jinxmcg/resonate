"""HC1: pair-grouped TRAIN holdout for ogbl-wikikg2 (build + audit only; no model).

Reserves the unordered entity pairs whose SplitMix64 bucket (seed 36840, the
TH1 hash from biokg/typed_path_pilot.py) is >= --cutoff of 100, i.e. ~3% of
pairs at the default 97. Every TRAIN row of a reserved pair is held out
together: duplicates, reverse copies and other relations on the same pair.
The held-out rows become a query split with 500 fixed uniform negatives per
direction (OGB's protocol), the rest is fit-TRAIN. A relation whose rows
would all be held out keeps every row of those pairs in fit (the trainer's
relation count must not change); the receipt lists them.

Writes holdout_wiki.npz (row indices into official TRAIN, the negatives, the
TRAIN digest) and holdout_wiki.json (the audit). train_wiki.load() applies
the file when WIKI_HOLDOUT points at it. Run this WITHOUT WIKI_HOLDOUT set.

Usage: python holdout_wiki.py --out holdout_wiki.npz
"""

import argparse
import hashlib
import json
import os
import time

import numpy as np

from train_wiki import load

SEED, CUTOFF, NNEG = 36840, 97, 500


def pair_keys(h, t, n_entities):
    return np.minimum(h, t).astype(np.int64) * n_entities + np.maximum(h, t)


def buckets(keys, seed):
    # SplitMix64; the uint64 overflow is deliberate (biokg/typed_path_pilot.py)
    x = np.asarray(keys, np.uint64) + np.uint64(seed) + np.uint64(0x9e3779b97f4a7c15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xbf58476d1ce4e5b9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94d049bb133111eb)
    return ((x ^ (x >> np.uint64(31))) % np.uint64(100)).astype(np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="holdout_wiki.npz")
    p.add_argument("--data-root", default="data_ogb")
    p.add_argument("--cutoff", type=int, default=CUTOFF)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--grouping", choices=["pair", "row"], default="pair",
                   help="pair: every TRAIN row of a reserved pair is held out together (HC1); "
                        "row: rows are bucketed independently, other rows on the same pair stay in fit (HC2: "
                        "validation and test pairs are already linked in TRAIN 9-12%% of the time; a pair holdout is 0%%)")
    args = p.parse_args()
    if os.environ.get("WIKI_HOLDOUT"):
        raise SystemExit("unset WIKI_HOLDOUT: this script reads the official TRAIN")
    t0 = time.time()
    split, n_ent = load(args.data_root)
    tr = split["train"]
    h, r, t = (np.asarray(tr[k]).astype(np.int64) for k in ("head", "relation", "tail"))
    n = len(h)
    digest = hashlib.sha256(np.concatenate([h, r, t]).tobytes()).hexdigest()
    assert n_ent * n_ent < np.iinfo(np.int64).max
    keys = pair_keys(h, t, n_ent)
    if args.grouping == "pair":
        b = buckets(keys, args.seed)
    else:   # row-level: hash the row index, so the same pair can sit on both sides
        b = buckets(np.arange(n, dtype=np.int64), args.seed)
    hold = b >= args.cutoff
    R = int(r.max()) + 1
    n_all = np.bincount(r, minlength=R)
    n_hold = np.bincount(r[hold], minlength=R)
    vanish = (n_all > 0) & (n_hold == n_all)
    kept_back = []
    if vanish.any():
        bad = np.unique(keys[vanish[r]])
        moved = hold & np.isin(keys, bad)
        kept_back = [dict(relation=int(x), rows=int(n_all[x])) for x in np.flatnonzero(vanish)]
        hold &= ~moved
        print(f"{int(vanish.sum())} relations would vanish from fit; {int(moved.sum())} rows of "
              f"{len(bad)} pairs kept in fit", flush=True)
    fit_rows = np.flatnonzero(~hold)
    hold_rows = np.flatnonzero(hold)
    # audit
    assert len(fit_rows) + len(hold_rows) == n
    pair_disjoint = not np.intersect1d(np.unique(keys[fit_rows]), np.unique(keys[hold_rows])).size
    assert pair_disjoint or args.grouping == "row"
    assert n_all[R - 1] > 0 and (r[fit_rows] == R - 1).any(), "max relation must stay in fit"
    N = len(hold_rows)
    rng = np.random.default_rng(args.seed)
    head_neg = rng.integers(0, n_ent, (N, NNEG), dtype=np.int32)
    tail_neg = rng.integers(0, n_ent, (N, NNEG), dtype=np.int32)
    hh, tt = h[hold_rows], t[hold_rows]
    # a negative equal to the answer would tie with it: shift it by one
    head_neg = np.where(head_neg == hh[:, None], (head_neg + 1) % n_ent, head_neg).astype(np.int32)
    tail_neg = np.where(tail_neg == tt[:, None], (tail_neg + 1) % n_ent, tail_neg).astype(np.int32)
    fit_nodes = np.zeros(n_ent, bool)
    fit_nodes[h[fit_rows]] = True
    fit_nodes[t[fit_rows]] = True
    cold = ~fit_nodes[hh] | ~fit_nodes[tt]
    n_fit_rel = np.bincount(r[fit_rows], minlength=R)
    summary = dict(
        dataset="ogbl-wikikg2", train_rows=n, entities=n_ent, relations=R,
        train_sha256=digest, seed=args.seed, cutoff=args.cutoff, negatives=NNEG,
        fit=dict(rows=len(fit_rows), pairs=int(len(np.unique(keys[fit_rows]))),
                 row_fraction=len(fit_rows) / n, relations_present=int((n_fit_rel > 0).sum())),
        holdout=dict(rows=N, pairs=int(len(np.unique(keys[hold_rows]))), row_fraction=N / n,
                     relations_present=int((n_hold > 0).sum()),
                     rows_with_endpoint_absent_from_fit=int(cold.sum()),
                     self_link_rows=int((hh == tt).sum())),
        relations_kept_entirely_in_fit=kept_back, grouping=args.grouping,
        pair_disjoint=bool(pair_disjoint), rows_partitioned=True,
        holdout_rows_whose_pair_is_linked_in_fit=int(np.isin(keys[hold_rows], np.unique(keys[fit_rows])).sum()),
        per_relation={str(i): dict(fit=int(n_fit_rel[i]), holdout=int(n_hold[i] if not vanish[i] else 0))
                      for i in range(R)},
        built=time.strftime("%Y-%m-%dT%H:%M:%S"), seconds=round(time.time() - t0, 1))
    np.savez(args.out, fit_rows=fit_rows, holdout_rows=hold_rows, head_neg=head_neg, tail_neg=tail_neg,
             train_sha256=np.array(digest), seed=np.array(args.seed), cutoff=np.array(args.cutoff),
             n_train=np.array(n), grouping=np.array(args.grouping))
    with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
        json.dump(summary, f, indent=1)
    s = dict(summary); s.pop("per_relation")
    print(json.dumps(s, indent=1))
    print(f"HOLDOUT_BUILT {args.out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
