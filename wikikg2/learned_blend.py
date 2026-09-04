"""Learned per-relation combiner (fail-fast, validation only).

Instead of selecting one of nine fixed weight patterns per (relation,
direction) group, fit a small logistic regression per group over the
members' z-scores: for each query row, the positive candidate against
its 500 negatives (softmax cross-entropy over the 501 candidates, i.e.
a listwise fit whose optimum ranks the positive first). Groups below
--min-rows fall back to a direction-level model; the direction-level
model falls back to a global one. Cross-fitted: fit on one half of the
validation triples, report on the other half, exactly like the search
mode of blend_wiki.py, so the two are directly comparable.

Usage: python learned_blend.py --members model_wiki_s0 analogy_s0_t3 holders cn_aa linked cn3_aa typed
"""

import argparse
import io
import contextlib

import numpy as np
import torch

from blend_wiki import load_split, mrr_rows, combined, choose, apply
from train_wiki import load


def fit_weights(Z, rows, dev, steps=300, lr=0.05, l2=1e-3, init=None):
    """Z: list of (R2, 501) float16 members; rows: bool mask. Returns
    weight vector (M,) + bias-free listwise logistic fit."""
    X = torch.stack([torch.from_numpy(z[rows].astype(np.float32)) for z in Z], -1).to(dev)  # (n, 501, M)
    M = X.shape[-1]
    w = torch.zeros(M, device=dev) if init is None else torch.tensor(init, device=dev, dtype=torch.float32)
    w.requires_grad_(True)
    opt = torch.optim.Adam([w], lr=lr)
    y = torch.zeros(X.shape[0], dtype=torch.long, device=dev)
    for _ in range(steps):
        opt.zero_grad()
        s = X @ w
        loss = torch.nn.functional.cross_entropy(s, y) + l2 * (w * w).sum()
        loss.backward()
        opt.step()
    return w.detach().cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--members", nargs="+", required=True)
    p.add_argument("--min-rows", type=int, default=250)
    p.add_argument("--cache-dir", default="ens_cache")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--freeze", action="store_true",
                   help="fit on FULL validation and, with --test, apply once to test")
    p.add_argument("--test", action="store_true")
    p.add_argument("--out", default="results/learned.npz")
    p.add_argument("--result", default="results/learned.json")
    p.add_argument("--dataset", default="ogbl-wikikg2",
                   help="evaluator name; for ogbl-biokg the test-mix proxy is skipped")
    args = p.parse_args()
    if args.freeze:
        return freeze(args)
    dev = torch.device(args.device)
    Z, rel = load_split(args.members, "valid", args.cache_dir)
    R2 = len(rel); n = R2 // 2; M = len(args.members)
    if args.dataset == "ogbl-wikikg2":
        split, _ = load("data_ogb")
        rv = np.asarray(split["valid"]["relation"]); rt = np.asarray(split["test"]["relation"])
        R = int(max(rv.max(), rt.max())) + 1
        cv = np.bincount(rv, minlength=R) / len(rv); ct = np.bincount(rt, minlength=R) / len(rt)
        wq = np.where(cv[rv] > 0, ct[rv] / np.maximum(cv[rv], 1e-12), 0); W = np.concatenate([wq, wq])
    else:
        W = np.ones(R2)
    half = np.random.default_rng(args.seed).random(n) < 0.5
    fit = np.concatenate([half, half]); dirs = np.concatenate([np.zeros(n, bool), np.ones(n, bool)])
    ho = ~fit
    rew = lambda x, m: (x[m] * W[m]).sum() / W[m].sum()
    watch = {"P1412": 111, "P106": 24, "P27": 260, "P641": 404} if args.dataset == "ogbl-wikikg2" \
        else {f"rel{r}": r for r in np.unique(rel)[:4]}

    # reference: the selection blend on the same split
    with contextlib.redirect_stdout(io.StringIO()):
        chosen, d_, (gl, gw, fb) = choose(Z, rel, fit, args.min_rows, M)
        s_sel = apply(Z, rel, d_, chosen, fallback=fb)
    r_sel = mrr_rows(s_sel)
    print(f"{'blend':>10} {'held-out':>9} {'test-mix':>9} {'head':>6} | " + " ".join(f"{k:>6}" for k in watch))
    print(f"{'selection':>10} {r_sel[ho].mean():9.4f} {rew(r_sel, ho):9.4f} {r_sel[ho & dirs].mean():6.3f} | "
          + " ".join(f"{r_sel[ho & dirs & (rel == r)].mean():6.3f}" for r in watch.values()))

    # learned: global -> direction -> relation, fit on the fit half only
    # subsample rows for the global/direction fits (speed)
    rng = np.random.default_rng(1)
    def sub(mask, k=60000):
        idx = np.nonzero(mask)[0]
        return np.isin(np.arange(R2), rng.choice(idx, size=min(k, len(idx)), replace=False))
    w_g = fit_weights(Z, sub(fit), dev)
    w_d = {d: fit_weights(Z, sub(fit & (dirs == bool(d))), dev, init=w_g) for d in (0, 1)}
    score = np.zeros((R2, 501), np.float32)
    n_loc = 0
    for r in np.unique(rel):
        for d in (0, 1):
            grp = (rel == r) & (dirs == bool(d))
            gf = grp & fit
            if gf.sum() >= args.min_rows:
                w = fit_weights(Z, gf, dev, steps=200, init=w_d[d]); n_loc += 1
            else:
                w = w_d[d]
            score[grp] = combined(Z, w, grp)
    r_l = mrr_rows(score)
    print(f"{'learned':>10} {r_l[ho].mean():9.4f} {rew(r_l, ho):9.4f} {r_l[ho & dirs].mean():6.3f} | "
          + " ".join(f"{r_l[ho & dirs & (rel == r)].mean():6.3f}" for r in watch.values())
          + f"   ({n_loc} local groups)")
    print("global weights:", dict(zip(args.members, np.round(w_g, 3))))
    print("head-direction weights:", dict(zip(args.members, np.round(w_d[1], 3))))


def fit_all(Z, rel, rows_mask, dirs, min_rows, dev):
    """global -> direction -> relation fits on rows_mask; returns
    (chosen {(r,d): w}, dir weights {d: w})."""
    rng = np.random.default_rng(1)
    R2 = len(rel)
    def sub(mask, k=60000):
        idx = np.nonzero(mask)[0]
        return np.isin(np.arange(R2), rng.choice(idx, size=min(k, len(idx)), replace=False))
    w_g = fit_weights(Z, sub(rows_mask), dev)
    w_d = {d: fit_weights(Z, sub(rows_mask & (dirs == bool(d))), dev, init=w_g) for d in (0, 1)}
    chosen = {}
    for r in np.unique(rel):
        for d in (0, 1):
            gf = (rel == r) & (dirs == bool(d)) & rows_mask
            chosen[(int(r), d)] = fit_weights(Z, gf, dev, steps=200, init=w_d[d]) if gf.sum() >= min_rows else w_d[d]
    return chosen, w_d


def freeze(args):
    import json, os, torch
    from ogb.linkproppred import Evaluator
    dev = torch.device(args.device)
    Zv, relv = load_split(args.members, "valid", args.cache_dir)
    R2 = len(relv); dirs = np.concatenate([np.zeros(R2 // 2, bool), np.ones(R2 // 2, bool)])
    chosen, w_d = fit_all(Zv, relv, np.ones(R2, bool), dirs, max(args.min_rows, 500), dev)
    n_loc = sum(1 for (r, d), w in chosen.items() if not np.array_equal(w, w_d[d]))
    sv = np.zeros((R2, 501), np.float32)
    for (r, d), w in chosen.items():
        g = (relv == r) & (dirs == bool(d)); sv[g] = combined(Zv, w, g)
    ins = mrr_rows(sv).mean()
    print(f"learned freeze: {n_loc} local groups; in-sample valid {ins:.4f}", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, members=np.array(args.members), keys=np.array([f"{k[0]},{k[1]}" for k in chosen]),
             weights=np.stack(list(chosen.values())), dir0=w_d[0], dir1=w_d[1])
    if not args.test:
        return
    Zt, relt = load_split(args.members, "test", args.cache_dir)
    R2t = len(relt); dt = np.concatenate([np.zeros(R2t // 2, bool), np.ones(R2t // 2, bool)])
    st = np.zeros((R2t, 501), np.float32); done = np.zeros(R2t, bool)
    for (r, d), w in chosen.items():
        g = (relt == r) & (dt == bool(d))
        if g.any(): st[g] = combined(Zt, w, g); done |= g
    for d in (0, 1):
        g = (~done) & (dt == bool(d))
        if g.any(): st[g] = combined(Zt, w_d[d], g)
    print(f"  fallback weights on {int((~done).sum()):,} rows of unseen groups")
    res = Evaluator(name=args.dataset).eval({"y_pred_pos": torch.from_numpy(st[:, 0].copy()),
                                               "y_pred_neg": torch.from_numpy(st[:, 1:].copy())})
    mrr = float(res["mrr_list"].mean()); rt_ = mrr_rows(st)
    print("\n================= COMMITTED TEST RESULT (learned combiner) ==============")
    print(f"official Evaluator test MRR: {mrr:.4f}")
    print(f"  by direction: tail {rt_[~dt].mean():.4f}  head {rt_[dt].mean():.4f}")
    for k in ("hits@1_list", "hits@3_list", "hits@10_list"):
        print(f"{k.replace('_list', '')}: {float(res[k].mean()):.4f}")
    json.dump({"members": args.members, "valid_in_sample": ins, "test_mrr": mrr, "local_groups": n_loc},
              open(args.result, "w"), indent=1)


if __name__ == "__main__":
    main()
