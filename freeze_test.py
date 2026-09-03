"""THE committed test shot.

Freezes per-(relation,direction) ensemble weights on the FULL valid
split for an explicit member list, applies them ONCE to the test
caches, and reports the official OGB Evaluator MRR. Run this exactly
once per submission decision.

  python freeze_test.py --members analogy analogy_ms analogy_t3 \
      model_final_seed0 ... --min-rows 4000

min-rows 4000 mirrors the search's 2000-on-a-half guard. Weights are
saved to frozen_weights.npz for reproducibility.
"""

import argparse
import json
import os

import numpy as np
import torch
from ogb.linkproppred import Evaluator

from ensemble_weights import families, normalize


def load_split(members, split, norm="z", cache_dir="ens_cache"):
    Z, rel = [], None
    for tag in members:
        f = os.path.join(cache_dir, f"{tag}.{split}.npz")
        d = np.load(f)
        sp, sn = d["sp"].astype(np.float32), d["sn"].astype(np.float32)
        x = np.concatenate([sp[:, None], sn], axis=1)
        Z.append(normalize(x, norm))
        if rel is None:
            rel = d["rel"]
    return Z, rel


def mrr_rows(s):
    rk = (s[:, 1:] >= s[:, :1]).sum(1) + 1
    return 1.0 / rk


def combined(Z, w, rows):
    s = np.zeros((rows.sum(), 501), dtype=np.float32)
    for m, zm in enumerate(Z):
        if w[m]:
            s += w[m] * zm[rows].astype(np.float32)
    return s


def candidates(per_m, M):
    uni = np.full(M, 1.0 / M)
    cands = {"uniform": uni}
    order = np.argsort(-per_m)
    for mtop in (1, 3, 5, 9):
        if mtop <= M:
            w = np.zeros(M)
            w[order[:mtop]] = 1.0 / mtop
            cands[f"top{mtop}"] = w
    for eta in (20, 50, 100):
        e = np.exp(eta * (per_m - per_m.max()))
        cands[f"soft{eta}"] = e / e.sum()
    return cands


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--members", nargs="+", required=True)
    p.add_argument("--min-rows", type=int, default=4000)
    p.add_argument("--out", default="frozen_weights.npz")
    p.add_argument("--result", default="committed_test_result.json")
    p.add_argument("--dry", action="store_true",
                   help="fit + freeze only; never load test caches")
    p.add_argument("--norm", choices=("z", "rank"), default="z")
    p.add_argument("--cache-dir", default="ens_cache")
    p.add_argument("--data-root", default="data_ogb")
    args = p.parse_args()
    M = len(args.members)
    print(f"members ({M}): {', '.join(args.members)}  norm: {args.norm}")

    # ---- fit on FULL valid ----
    Zv, relv = load_split(args.members, "valid", args.norm, args.cache_dir)
    R2 = len(relv)
    dirs = np.concatenate([np.zeros(R2 // 2, bool), np.ones(R2 // 2, bool)])
    per_m_glob = np.array([mrr_rows(z.astype(np.float32)).mean()
                           for z in Zv])
    g_lab, g_w, g_val = None, None, -1
    for lab, w in candidates(per_m_glob, M).items():
        v = mrr_rows(combined(Zv, w, np.ones(R2, bool))).mean()
        if v > g_val:
            g_lab, g_w, g_val = lab, w, v
    print(f"global weights on full valid: {g_lab} ({g_val:.4f})")

    def fit_group(grp, extra):
        per_m = np.array([mrr_rows(z[grp].astype(np.float32)).mean()
                          for z in Zv])
        local = {"global:" + g_lab: g_w}
        local.update(extra)
        local.update(candidates(per_m, M))
        lab_b, w_b, v_b = None, None, -1
        for lab, w in local.items():
            v = mrr_rows(combined(Zv, w, grp)).mean()
            if v > v_b:
                lab_b, w_b, v_b = lab, w, v
        return lab_b, w_b

    # family level (drug-drug_*, protein-protein_* pooled) sits between
    # global and local, mirroring ensemble_weights.py
    fam, _ = families(relv, args.data_root)
    fam_w = {}
    for fi in np.unique(fam):
        for d in (0, 1):
            grp = (fam == fi) & (dirs == bool(d))
            if grp.sum() >= args.min_rows:
                lab, w = fit_group(grp, {})
                fam_w[(int(fi), d)] = ("family:" + lab, w)
    print(f"family weights fitted for {len(fam_w)} (family,dir) groups")

    chosen = {}
    for rr in np.unique(relv):
        for d in (0, 1):
            grp = (relv == rr) & (dirs == bool(d))
            fb = fam_w.get((int(fam[grp][0]), d), ("global:" + g_lab, g_w))
            if grp.sum() < args.min_rows:
                chosen[(int(rr), d)] = fb
                continue
            chosen[(int(rr), d)] = fit_group(grp, {fb[0]: fb[1]})
    n_loc = sum(1 for v in chosen.values()
                if not v[0].startswith(("global", "family")))
    n_fam = sum(1 for v in chosen.values() if v[0].startswith("family"))
    print(f"groups with local weights: {n_loc}/{len(chosen)}; "
          f"family weights: {n_fam}")
    np.savez(args.out,
             members=np.array(args.members), norm=args.norm,
             keys=np.array([f"{k[0]},{k[1]}" for k in chosen]),
             labels=np.array([v[0] for v in chosen.values()]),
             weights=np.stack([v[1] for v in chosen.values()]))
    print(f"weights frozen -> {args.out}")
    fit_all = np.zeros(R2, np.float32)
    for (rr, d), (lab, w) in chosen.items():
        grp = (relv == rr) & (dirs == bool(d))
        if grp.sum():
            fit_all[grp] = mrr_rows(combined(Zv, w, grp))
    print(f"frozen weights on full valid (in-sample): {fit_all.mean():.4f}")
    del Zv
    if args.dry:
        print("dry run: test caches not loaded")
        return

    # ---- apply ONCE to test ----
    Zt, relt = load_split(args.members, "test", args.norm, args.cache_dir)
    R2t = len(relt)
    Nt = R2t // 2
    dirst = np.concatenate([np.zeros(Nt, bool), np.ones(Nt, bool)])
    score = np.zeros((R2t, 501), dtype=np.float32)
    for (rr, d), (lab, w) in chosen.items():
        grp = (relt == rr) & (dirst == bool(d))
        if grp.sum():
            score[grp] = combined(Zt, w, grp)

    ev = Evaluator(name="ogbl-biokg")
    res = ev.eval({"y_pred_pos": torch.from_numpy(score[:, 0].copy()),
                   "y_pred_neg": torch.from_numpy(score[:, 1:].copy())})
    mrr = float(res["mrr_list"].mean())
    print("\n================= COMMITTED TEST RESULT =================")
    print(f"official Evaluator test MRR: {mrr:.4f}")
    for k in ("hits@1_list", "hits@3_list", "hits@10_list"):
        if k in res:
            print(f"{k.replace('_list', '')}: "
                  f"{float(res[k].mean()):.4f}")
    with open(args.result, "w") as f:
        json.dump({"members": args.members, "test_mrr": mrr,
                   "global": g_lab, "local_groups": n_loc}, f, indent=2)


if __name__ == "__main__":
    main()
