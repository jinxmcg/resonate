"""Per-relation-direction ensemble weight search (RelEns-style) on
cached validation scores. Valid-only: the committed test shot runs
separately once the config is frozen.

Discipline (the seed-null lesson): weights are fit per (relation,
direction) group ONLY where the group is big enough; small groups fall
back to the globally-fitted weights. Honest estimate = fit on a random
half of triples (both directions of a triple stay together), report on
the untouched half. Candidate weight family per group (evaluated on
the fit half; global-uniform always included so a group can never
choose worse than baseline on the fit half):
  - uniform mean over all members
  - uniform mean over the top-m members by fit-half group MRR
  - softmax(eta * member group MRR) for a small eta grid

Usage: python ensemble_weights.py --members model_final_seed0 analogy_s0 \
           analogy_s0_t3 jaccard jaccard_t3 [--min-rows 2000]
"""

import argparse
import csv
import gzip
import glob
import os

import numpy as np


def normalize(x, norm="z"):
    """Per-row member normalisation. 'z': z-score over the 501
    candidates. 'rank': minus the competition rank (1 + #strictly
    better candidates) scaled to [-1, 0) -- RelEns combines ranks, which
    neutralises heavy-tailed member scores; ties share a rank."""
    if norm == "z":
        mu = x.mean(1, keepdims=True)
        sd = x.std(1, keepdims=True) + 1e-6
        return ((x - mu) / sd).astype(np.float16)
    o = np.argsort(-x, axis=1, kind="stable")
    rows = np.arange(x.shape[0])[:, None]
    xs = x[rows, o]
    new = np.ones_like(xs, dtype=bool)
    new[:, 1:] = xs[:, 1:] != xs[:, :-1]
    first = np.where(new, np.arange(x.shape[1])[None, :], 0)
    first = np.maximum.accumulate(first, axis=1)
    r = np.empty(x.shape, np.float32)
    r[rows, o] = first + 1
    return (-r / x.shape[1]).astype(np.float16)


def load_members(cache_dir="ens_cache", split="valid", exclude=(), norm="z",
                 members=None):
    """members: explicit tag list (in order); None = every cache in
    cache_dir minus `exclude`."""
    if members:
        files = [os.path.join(cache_dir, f"{m}.{split}.npz") for m in members]
    else:
        files = sorted(glob.glob(os.path.join(cache_dir, f"*.{split}.npz")))
    names, Z = [], []
    rel = None
    for f in files:
        tag = os.path.basename(f).split(".")[0]
        if tag in exclude:
            continue
        d = np.load(f)
        sp, sn = d["sp"].astype(np.float32), d["sn"].astype(np.float32)
        x = np.concatenate([sp[:, None], sn], axis=1)  # (2N, 501)
        Z.append(normalize(x, norm))
        names.append(tag)
        if rel is None:
            rel = d["rel"]
    return names, Z, rel


def mrr_rows(s):  # s: (n, 501) f32, col 0 = positive
    rk = (s[:, 1:] >= s[:, :1]).sum(1) + 1
    return 1.0 / rk


def families(rel, data_root="data_ogb"):
    """Group id per row: relation name up to the first '_' (drug-drug_*
    and protein-protein_* pool their ~2k-row relations)."""
    with gzip.open(os.path.join(data_root, "ogbl_biokg", "mapping",
                                "relidx2relname.csv.gz"), "rt") as f:
        rd = csv.reader(f)
        next(rd)
        nm = {int(i): n for i, n in rd}
    fam_of = {i: n.split("_")[0] for i, n in nm.items()}
    fams = sorted(set(fam_of.values()))
    idx = {f: k for k, f in enumerate(fams)}
    return np.array([idx[fam_of[int(x)]] for x in rel]), fams


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--min-rows", type=int, default=2000,
                   help="groups with fewer fit-half rows use the "
                        "global weights (seed-null guard)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exclude", nargs="*", default=[],
                   help="member tags to leave out (ablation)")
    p.add_argument("--norm", choices=("z", "rank"), default="z",
                   help="per-row member normalisation (see normalize)")
    p.add_argument("--members", nargs="*", default=None,
                   help="explicit member tags (default: every cache)")
    p.add_argument("--cache-dir", default="ens_cache")
    p.add_argument("--data-root", default="data_ogb")
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)

    names, Z, rel = load_members(args.cache_dir, exclude=set(args.exclude),
                                 norm=args.norm, members=args.members)
    if args.exclude:
        print(f"excluded: {args.exclude}")
    print(f"norm: {args.norm}")
    M, R2 = len(Z), len(rel)
    n_tri = R2 // 2
    print(f"members ({M}): {', '.join(names)}")

    # triple-paired half split: a triple's tail and head rows together
    half_tri = rng.random(n_tri) < 0.5
    fitmask = np.concatenate([half_tri, half_tri])
    dirs = np.concatenate([np.zeros(n_tri, bool), np.ones(n_tri, bool)])

    def combined(w, rows):
        s = np.zeros((rows.sum(), 501), dtype=np.float32)
        for m in range(M):
            if w[m]:
                s += w[m] * Z[m][rows].astype(np.float32)
        return s

    # baselines: uniform z-mean over all members, and the model members
    # alone (every tag that is not a retrieval feature)
    uni = np.full(M, 1.0 / M)
    is_model = np.array([not n.startswith(("analogy", "jaccard"))
                         for n in names])
    model_only = is_model / max(is_model.sum(), 1)
    for lab, w in (("uniform (z-mean)", uni),
                   ("model-only (z-mean)", model_only)):
        a = mrr_rows(combined(w, fitmask)).mean()
        b = mrr_rows(combined(w, ~fitmask)).mean()
        print(f"{lab}: fit-half {a:.4f}  held-out {b:.4f}")

    # global weight candidates fit on the fit half
    per_member_fit = np.array(
        [mrr_rows(Z[m][fitmask].astype(np.float32)).mean()
         for m in range(M)])
    order = np.argsort(-per_member_fit)
    cands = {"uniform": uni}
    for mtop in (3, 5, 9, 13):
        w = np.zeros(M)
        w[order[:mtop]] = 1.0 / mtop
        cands[f"top{mtop}"] = w
    for eta in (20, 50, 100):
        e = np.exp(eta * (per_member_fit - per_member_fit.max()))
        cands[f"soft{eta}"] = e / e.sum()
    g_lab, g_w, g_val = None, None, -1
    for lab, w in cands.items():
        v = mrr_rows(combined(w, fitmask)).mean()
        if v > g_val:
            g_lab, g_w, g_val = lab, w, v
    print(f"global best on fit half: {g_lab} ({g_val:.4f})")

    def fit_group(gf, extra):
        per_m = np.array(
            [mrr_rows(Z[m][gf].astype(np.float32)).mean()
             for m in range(M)])
        o = np.argsort(-per_m)
        local = {"uniform": uni, f"global:{g_lab}": g_w}
        local.update(extra)
        for mtop in (1, 3, 5, 9):
            w = np.zeros(M)
            w[o[:mtop]] = 1.0 / mtop
            local[f"top{mtop}"] = w
        for eta in (20, 50, 100):
            e = np.exp(eta * (per_m - per_m.max()))
            local[f"soft{eta}"] = e / e.sum()
        lab_best, w_best, v_best = None, None, -1
        for lab, w in local.items():
            v = mrr_rows(combined(w, gf)).mean()
            if v > v_best:
                lab_best, w_best, v_best = lab, w, v
        return lab_best, w_best

    # family level (drug-drug_*, protein-protein_* pooled) between
    # global and local: small relations fall back here, not to global
    fam, fam_names = families(rel, args.data_root)
    fam_w = {}
    for fi in np.unique(fam):
        for d in (0, 1):
            gf = (fam == fi) & (dirs == bool(d)) & fitmask
            if gf.sum() >= args.min_rows:
                lab, w = fit_group(gf, {})
                fam_w[(int(fi), d)] = (f"family:{lab}", w)
    print(f"family weights fitted for {len(fam_w)} (family,dir) groups")

    # per-(relation, direction) choice
    rr_fit = np.zeros(R2, dtype=np.float32)   # chosen recipe, fit half
    rr_out = np.zeros(R2, dtype=np.float32)   # applied to held-out
    chosen = {}
    for r in np.unique(rel):
        for d in (0, 1):
            grp = (rel == r) & (dirs == bool(d))
            gf, gh = grp & fitmask, grp & ~fitmask
            fkey = (int(fam[grp][0]), d)
            fb = fam_w.get(fkey, (f"global:{g_lab}", g_w))
            if gf.sum() < args.min_rows:
                lab_best, w_best = fb
            else:
                lab_best, w_best = fit_group(gf, {fb[0]: fb[1]})
            chosen[(int(r), d)] = (lab_best, w_best)
            if gf.sum():
                rr_fit[gf] = mrr_rows(combined(w_best, gf))
            if gh.sum():
                rr_out[gh] = mrr_rows(combined(w_best, gh))

    print(f"\nper-relation recipe: fit-half {rr_fit[fitmask].mean():.4f}"
          f"  HELD-OUT {rr_out[~fitmask].mean():.4f}")
    base_out = mrr_rows(combined(uni, ~fitmask)).mean()
    print(f"held-out delta vs uniform: "
          f"{rr_out[~fitmask].mean() - base_out:+.4f}")
    n_loc = sum(1 for v in chosen.values()
                if not v[0].startswith(("global", "family")))
    n_fam = sum(1 for v in chosen.values() if v[0].startswith("family"))
    print(f"groups with local weights: {n_loc}/{len(chosen)}; "
          f"family weights: {n_fam}")
    lines = [f"{k}: {v[0]}" for k, v in sorted(chosen.items())
             if not v[0].startswith("global")]
    print("local choices:", "; ".join(lines[:20]),
          "..." if len(lines) > 20 else "")


if __name__ == "__main__":
    main()
