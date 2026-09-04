"""Blend selection for ogbl-wikikg2 members (RelEns-style, biokg row B).

Same discipline as ensemble_weights.py / freeze_test.py on biokg,
minus the relation-family level (wikikg2 has no families): weights
are a SELECTION among a small candidate set — uniform, top-m by
member MRR, softmax(eta * MRR) — chosen per (relation, direction)
group where the group is large enough, else the global choice.
Members are z-normalised per row over the 501 candidates.

  search  : fit on a random half of triples, report on the other half
            (the honest estimate; both directions of a triple stay
            together)
  freeze  : fit on the FULL valid split, save weights; with --test,
            apply ONCE to the test caches and print the official
            Evaluator MRR (the committed shot)

Usage:
  python blend_wiki.py search --members lever_x analogy_s0 analogy_s0_t3 holders
  python blend_wiki.py freeze --members ... [--test]
"""

import argparse
import json
import os

import numpy as np


def normalize(x):
    mu = x.mean(1, keepdims=True)
    sd = x.std(1, keepdims=True) + 1e-6
    return ((x - mu) / sd).astype(np.float16)


def load_split(members, split, cache_dir):
    Z, rel = [], None
    for tag in members:
        d = np.load(os.path.join(cache_dir, f"{tag}.{split}.npz"))
        sp, sn = d["sp"].astype(np.float32), d["sn"].astype(np.float32)
        Z.append(normalize(np.concatenate([sp[:, None], sn], axis=1)))
        if rel is None:
            rel = d["rel"]
    return Z, rel


def mrr_rows(s):
    return 1.0 / ((s[:, 1:] >= s[:, :1]).sum(1) + 1)


def combined(Z, w, rows):
    s = np.zeros((int(rows.sum()), 501), dtype=np.float32)
    for m, zm in enumerate(Z):
        if w[m]:
            s += w[m] * zm[rows].astype(np.float32)
    return s


def candidates(per_m, M, tops=(1, 3, 5, 9)):
    cands = {"uniform": np.full(M, 1.0 / M)}
    order = np.argsort(-per_m)
    for mtop in tops:
        if mtop <= M:
            w = np.zeros(M)
            w[order[:mtop]] = 1.0 / mtop
            cands[f"top{mtop}"] = w
    for eta in (20, 50, 100):
        e = np.exp(eta * (per_m - per_m.max()))
        cands[f"soft{eta}"] = e / e.sum()
    return cands


def fit(Z, grp, extra, M):
    per_m = np.array([mrr_rows(z[grp].astype(np.float32)).mean() for z in Z])
    local = dict(extra)
    local.update(candidates(per_m, M))
    best = (None, None, -1.0)
    for lab, w in local.items():
        v = mrr_rows(combined(Z, w, grp)).mean()
        if v > best[2]:
            best = (lab, w, v)
    return best


def choose(Z, rel, fitmask, min_rows, M):
    """Global choice on fitmask rows, then per (relation, direction)."""
    R2 = len(rel)
    dirs = np.concatenate([np.zeros(R2 // 2, bool), np.ones(R2 // 2, bool)])
    g_lab, g_w, g_val = fit(Z, fitmask, {}, M)
    print(f"global choice on fit rows: {g_lab} ({g_val:.4f})", flush=True)
    # direction level: tail queries and head queries get their own
    # fallback (a member can be good in one direction only)
    dir_w = {}
    for d in (0, 1):
        lab, w, v = fit(Z, fitmask & (dirs == bool(d)),
                        {"global:" + g_lab: g_w}, M)
        dir_w[d] = ("dir:" + lab, w)
        print(f"direction {'head' if d else 'tail'} choice: {lab} ({v:.4f})",
              flush=True)
    chosen = {}
    for rr in np.unique(rel):
        for d in (0, 1):
            grp = (rel == rr) & (dirs == bool(d))
            gf = grp & fitmask
            if gf.sum() < min_rows:
                chosen[(int(rr), d)] = dir_w[d]
            else:
                lab, w, _ = fit(Z, gf, {dir_w[d][0]: dir_w[d][1]}, M)
                chosen[(int(rr), d)] = (lab, w)
    n_loc = sum(1 for v in chosen.values()
                if not v[0].startswith(("global", "dir")))
    print(f"groups with local weights: {n_loc}/{len(chosen)}", flush=True)
    return chosen, dirs, (g_lab, g_w, {d: dir_w[d][1] for d in (0, 1)})


def apply(Z, rel, dirs, chosen, fallback=None):
    """fallback: {d: weights} for (relation, direction) groups that never
    occurred in the fitting split (test has relations valid lacks)."""
    R2 = len(rel)
    out = np.zeros((R2, 501), np.float32)
    done = np.zeros(R2, bool)
    for (rr, d), (lab, w) in chosen.items():
        grp = (rel == rr) & (dirs == bool(d))
        if grp.sum():
            out[grp] = combined(Z, w, grp)
            done |= grp
    if fallback is not None and not done.all():
        for d in (0, 1):
            grp = (~done) & (dirs == bool(d))
            if grp.sum():
                out[grp] = combined(Z, fallback[d], grp)
        print(f"  fallback weights applied to {int((~done).sum()):,} rows "
              f"of unseen (relation, direction) groups", flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["search", "freeze"])
    p.add_argument("--members", nargs="+", required=True)
    p.add_argument("--min-rows", type=int, default=None,
                   help="group size guard (default 250 search / 500 freeze; "
                        "swept on wikikg2 validation 2026-09-03, held-out "
                        "MRR rises monotonically down to 250 on the half)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--test", action="store_true",
                   help="freeze only: apply the frozen weights to the "
                        "test caches ONCE and report the Evaluator MRR")
    p.add_argument("--cache-dir", default="ens_cache")
    p.add_argument("--out", default="frozen_weights_wiki.npz")
    p.add_argument("--result", default="committed_test_wiki.json")
    args = p.parse_args()
    M = len(args.members)
    min_rows = args.min_rows or (250 if args.mode == "search" else 500)
    print(f"members ({M}): {', '.join(args.members)}  min_rows={min_rows}",
          flush=True)
    Zv, relv = load_split(args.members, "valid", args.cache_dir)
    R2 = len(relv)
    n_tri = R2 // 2
    dv = np.concatenate([np.zeros(n_tri, bool), np.ones(n_tri, bool)])
    for m, tag in enumerate(args.members):
        rm = mrr_rows(Zv[m].astype(np.float32))
        print(f"  {tag}: valid MRR alone {rm.mean():.4f}  (tail {rm[~dv].mean():.4f} "
              f"head {rm[dv].mean():.4f})", flush=True)
    is_model = np.array([not n.startswith(("analogy", "holders", "jaccard",
                                           "cn", "linked"))
                         for n in args.members])
    uni = np.full(M, 1.0 / M)
    model_only = is_model / max(is_model.sum(), 1)

    if args.mode == "search":
        half = np.random.default_rng(args.seed).random(n_tri) < 0.5
        fitmask = np.concatenate([half, half])
        for lab, w in (("uniform z-mean", uni), ("model-only", model_only)):
            a = mrr_rows(combined(Zv, w, fitmask)).mean()
            b = mrr_rows(combined(Zv, w, ~fitmask)).mean()
            print(f"{lab}: fit-half {a:.4f}  held-out {b:.4f}", flush=True)
        chosen, dirs, _ = choose(Zv, relv, fitmask, min_rows, M)
        s = apply(Zv, relv, dirs, chosen)
        rr = mrr_rows(s)
        print(f"\nper-relation recipe: fit-half {rr[fitmask].mean():.4f}  "
              f"HELD-OUT {rr[~fitmask].mean():.4f}")
        base_rows = mrr_rows(combined(Zv, model_only, ~fitmask))
        print(f"held-out delta vs model-only: {rr[~fitmask].mean() - base_rows.mean():+.4f}")
        hm = ~fitmask
        for lab, dm in (("tail (h,r,?)", ~dirs), ("head (?,r,t)", dirs)):
            print(f"  {lab}: model-only {mrr_rows(combined(Zv, model_only, hm & dm)).mean():.4f}"
                  f" -> recipe {rr[hm & dm].mean():.4f}")
        return

    # freeze on FULL valid
    chosen, dirs, (g_lab, g_w, dir_fb) = choose(Zv, relv, np.ones(R2, bool),
                                                min_rows, M)
    s = apply(Zv, relv, dirs, chosen)
    ins = mrr_rows(s).mean()
    print(f"frozen weights on full valid (in-sample): {ins:.4f}")
    np.savez(args.out, members=np.array(args.members),
             keys=np.array([f"{k[0]},{k[1]}" for k in chosen]),
             labels=np.array([v[0] for v in chosen.values()]),
             weights=np.stack([v[1] for v in chosen.values()]),
             fallback=np.stack([dir_fb[0], dir_fb[1]]))
    print(f"weights frozen -> {args.out}")
    del Zv
    if not args.test:
        print("test caches not loaded (pass --test for the committed shot)")
        return
    import torch
    from ogb.linkproppred import Evaluator
    Zt, relt = load_split(args.members, "test", args.cache_dir)
    dirst = np.concatenate([np.zeros(len(relt) // 2, bool),
                            np.ones(len(relt) // 2, bool)])
    st = apply(Zt, relt, dirst, chosen, fallback=dir_fb)
    res = Evaluator(name="ogbl-wikikg2").eval(
        {"y_pred_pos": torch.from_numpy(st[:, 0].copy()),
         "y_pred_neg": torch.from_numpy(st[:, 1:].copy())})
    mrr = float(res["mrr_list"].mean())
    print("\n================= COMMITTED TEST RESULT (wikikg2) ==============")
    print(f"official Evaluator test MRR: {mrr:.4f}")
    rt = mrr_rows(st)
    print(f"  by direction: tail {rt[~dirst].mean():.4f}  head {rt[dirst].mean():.4f}")
    for k in ("hits@1_list", "hits@3_list", "hits@10_list"):
        if k in res:
            print(f"{k.replace('_list', '')}: {float(res[k].mean()):.4f}")
    with open(args.result, "w") as f:
        json.dump({"members": args.members, "valid_in_sample": ins,
                   "test_mrr": mrr, "global": g_lab}, f, indent=2)


if __name__ == "__main__":
    main()
