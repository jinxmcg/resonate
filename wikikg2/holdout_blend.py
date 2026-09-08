"""HC1: the learned per-relation combiner fit on the TRAIN holdout, applied
unchanged to validation.

Fit world: the members' 'holdout' caches (ens_cache_holdout/), produced with
WIKI_HOLDOUT set, i.e. a model trained on fit-TRAIN and members that read
fit-TRAIN, scored on the reserved TRAIN rows. Apply world: the members'
'valid' caches of the deployed system (models trained on all of TRAIN,
members reading all of TRAIN). Members are matched by position; the model
and analogy members differ in name between the two worlds, the shared
members do not.

Nothing is fit on validation. The group-size guard is chosen inside the
holdout by cross-fitting on halves (both folds) over --guards, unless
--min-rows is given. Validation is read to REPORT: the official Evaluator on
the full split, plus, on one held-out half (seed 0, the split learned_blend.py
search uses), the references fit on the other half: the selection blend and
the validation-fit learned combiner. The pre-registered gate is in
HOLDOUT_COMBINER.md.

Usage:
  python holdout_blend.py --fit-members model_fit97_s0 analogy_f0_t3 holders cn_aa linked cn3_aa typed \
      --apply-members model_dist_s1 analogy_d1_t3 holders cn_aa linked cn3_aa typed \
      --fit-cache-dir ens_cache_holdout --cache-dir ens_cache --label dist_s1
"""

import argparse
import contextlib
import io
import json
import os

import numpy as np
import torch

from blend_wiki import load_split, mrr_rows, combined, choose, apply
from learned_blend import fit_all


def apply_w(Z, rel, dirs, chosen, w_d):
    R2 = len(rel)
    out = np.zeros((R2, 501), np.float32)
    done = np.zeros(R2, bool)
    for (r, d), w in chosen.items():
        g = (rel == r) & (dirs == bool(d))
        if g.any():
            out[g] = combined(Z, w, g)
            done |= g
    for d in (0, 1):
        g = (~done) & (dirs == bool(d))
        if g.any():
            out[g] = combined(Z, w_d[d], g)
    return out, int((~done).sum())


def halves(n_tri, seed):
    half = np.random.default_rng(seed).random(n_tri) < 0.5
    return np.concatenate([half, half])


def dirs_of(R2):
    return np.concatenate([np.zeros(R2 // 2, bool), np.ones(R2 // 2, bool)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fit-members", nargs="+", required=True)
    p.add_argument("--apply-members", nargs="+", required=True)
    p.add_argument("--fit-cache-dir", default="ens_cache_holdout")
    p.add_argument("--cache-dir", default="ens_cache")
    p.add_argument("--min-rows", type=int, default=None)
    p.add_argument("--guards", type=int, nargs="+", default=[250, 500, 1000, 2000])
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--label", default="hc1")
    p.add_argument("--out-dir", default="results/hc1")
    p.add_argument("--row-weights", default=None,
                   help="HC3: .npy of one importance weight per holdout triple (applied to both directions)")
    args = p.parse_args()
    assert len(args.fit_members) == len(args.apply_members)
    M = len(args.fit_members)
    dev = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    rec = {"fit_members": args.fit_members, "apply_members": args.apply_members, "seed": args.seed}

    # ---- fit world -------------------------------------------------------
    Zh, relh = load_split(args.fit_members, "holdout", args.fit_cache_dir)
    R2h = len(relh); dh = dirs_of(R2h)
    W = None
    if args.row_weights:
        w1 = np.load(args.row_weights).astype(np.float32)
        assert len(w1) == R2h // 2, (len(w1), R2h)
        W = np.concatenate([w1, w1]); rec["row_weights"] = args.row_weights
        print(f"row weights: {args.row_weights}, ESS {W.sum()**2/(W*W).sum()/2:,.0f} of {R2h//2:,} triples", flush=True)
    wm = (lambda r, m: float((r[m] * W[m]).sum() / W[m].sum())) if W is not None else (lambda r, m: float(r[m].mean()))
    print(f"holdout: {R2h//2:,} rows x 2 directions; members ({M}):", flush=True)
    rec["holdout_alone"] = {}
    for m, tag in enumerate(args.fit_members):
        rm = mrr_rows(Zh[m].astype(np.float32))
        rec["holdout_alone"][tag] = float(rm.mean())
        print(f"  {tag:>22}: holdout MRR alone {rm.mean():.4f} (tail {rm[~dh].mean():.4f} head {rm[dh].mean():.4f})", flush=True)
    if args.min_rows is None:
        fm = halves(R2h // 2, args.seed)
        rec["guard_sweep"] = {}
        for mr in args.guards:
            vals = []
            for fold in (fm, ~fm):
                ch, wd = fit_all(Zh, relh, fold, dh, mr, dev, w=W)
                s, _ = apply_w(Zh, relh, dh, ch, wd)
                vals.append(wm(mrr_rows(s), ~fold))
            rec["guard_sweep"][str(mr)] = vals
            print(f"guard {mr:>5}: holdout cross-fit {np.mean(vals):.4f} (folds {vals[0]:.4f} {vals[1]:.4f})", flush=True)
        best = max(rec["guard_sweep"].values(), key=np.mean)
        min_rows = max(int(k) for k, v in rec["guard_sweep"].items() if np.mean(v) == np.mean(best))
    else:
        min_rows = args.min_rows
    rec["min_rows"] = min_rows
    chosen, w_d = fit_all(Zh, relh, np.ones(R2h, bool), dh, min_rows, dev, w=W)
    n_loc = sum(1 for (r, d), w in chosen.items() if not np.array_equal(w, w_d[d]))
    s_in, _ = apply_w(Zh, relh, dh, chosen, w_d)
    rec["holdout_in_sample"] = float(mrr_rows(s_in).mean())
    if W is not None:
        rec["holdout_in_sample_weighted"] = wm(mrr_rows(s_in), np.ones(R2h, bool))
    rec["local_groups"] = n_loc
    print(f"holdout fit: guard {min_rows}, {n_loc} local groups, in-sample holdout MRR {rec['holdout_in_sample']:.4f}", flush=True)
    print("holdout global/dir weights:")
    for d in (0, 1):
        print(f"  {'head' if d else 'tail'}: " + " ".join(f"{t}={w:+.3f}" for t, w in zip(args.fit_members, w_d[d])))
    np.savez(os.path.join(args.out_dir, f"weights_{args.label}.npz"), fit_members=np.array(args.fit_members),
             apply_members=np.array(args.apply_members), keys=np.array([f"{k[0]},{k[1]}" for k in chosen]),
             weights=np.stack(list(chosen.values())), dir0=w_d[0], dir1=w_d[1], min_rows=np.array(min_rows))
    del Zh, s_in

    # ---- apply world -----------------------------------------------------
    from ogb.linkproppred import Evaluator
    Zv, relv = load_split(args.apply_members, "valid", args.cache_dir)
    R2v = len(relv); dv = dirs_of(R2v)
    sv, nfb = apply_w(Zv, relv, dv, chosen, w_d)
    res = Evaluator(name="ogbl-wikikg2").eval({"y_pred_pos": torch.from_numpy(sv[:, 0].copy()),
                                                "y_pred_neg": torch.from_numpy(sv[:, 1:].copy())})
    rv = mrr_rows(sv)
    rec["valid_full"] = {"mrr": float(res["mrr_list"].mean()), "hits@1": float(res["hits@1_list"].mean()),
                         "hits@3": float(res["hits@3_list"].mean()), "hits@10": float(res["hits@10_list"].mean()),
                         "tail": float(rv[~dv].mean()), "head": float(rv[dv].mean()), "fallback_rows": nfb}
    print(f"\nHOLDOUT-FIT combiner on FULL validation (official Evaluator): MRR {rec['valid_full']['mrr']:.4f}  "
          f"hits@1 {rec['valid_full']['hits@1']:.4f} hits@3 {rec['valid_full']['hits@3']:.4f} hits@10 {rec['valid_full']['hits@10']:.4f}  "
          f"(tail {rv[~dv].mean():.4f} head {rv[dv].mean():.4f}; {nfb:,} rows on direction fallback)", flush=True)

    # references on the held-out half of validation (fit on the other half)
    fv = halves(R2v // 2, args.seed); ho = ~fv
    rec["valid_alone"] = {}
    for m, tag in enumerate(args.apply_members):
        rm = mrr_rows(Zv[m].astype(np.float32))
        rec["valid_alone"][tag] = float(rm.mean())
    with contextlib.redirect_stdout(io.StringIO()):
        ch_s, d_, (gl, gw, fb) = choose(Zv, relv, fv, 250, M)
        r_sel = mrr_rows(apply(Zv, relv, d_, ch_s, fallback=fb))
    ch_l, wd_l = fit_all(Zv, relv, fv, dv, 250, dev)
    s_l, _ = apply_w(Zv, relv, dv, ch_l, wd_l)
    r_l = mrr_rows(s_l)
    model_alone = mrr_rows(Zv[0].astype(np.float32))
    rows = [("model alone", model_alone), ("selection blend (fit other half)", r_sel),
            ("learned combiner (fit other half, validation labels)", r_l),
            ("HOLDOUT-FIT combiner (no validation fit)", rv)]
    print(f"\non the held-out half of validation (seed {args.seed}, {int(ho.sum())//2:,} triples):")
    print(f"  {'row':<54} {'MRR':>7} {'tail':>7} {'head':>7}")
    rec["valid_half"] = {}
    for name, rr in rows:
        rec["valid_half"][name] = float(rr[ho].mean())
        print(f"  {name:<54} {rr[ho].mean():7.4f} {rr[ho & ~dv].mean():7.4f} {rr[ho & dv].mean():7.4f}")
    gain = r_l[ho].mean() - r_sel[ho].mean()
    bar = r_sel[ho].mean() + 0.5 * gain
    kept = (rv[ho].mean() - r_sel[ho].mean()) / gain if gain > 0 else float("nan")
    rec["gate"] = {"selection": float(r_sel[ho].mean()), "learned_xfit": float(r_l[ho].mean()),
                   "holdout_fit": float(rv[ho].mean()), "bar": float(bar), "fraction_of_gain_kept": float(kept),
                   "pass": bool(rv[ho].mean() >= bar)}
    print(f"\nGATE: combiner gain over selection {gain:+.4f}; holdout-fit keeps {kept*100:.0f}% of it; "
          f"bar {bar:.4f} -> {'PASS' if rec['gate']['pass'] else 'FAIL'}")
    print("validation-fit global/dir weights (for comparison, not used):")
    for d in (0, 1):
        print(f"  {'head' if d else 'tail'}: " + " ".join(f"{t}={w:+.3f}" for t, w in zip(args.apply_members, wd_l[d])))
    with open(os.path.join(args.out_dir, f"{args.label}.json"), "w") as f:
        json.dump(rec, f, indent=1)
    print(f"HC1_DONE {args.label}")


if __name__ == "__main__":
    main()
