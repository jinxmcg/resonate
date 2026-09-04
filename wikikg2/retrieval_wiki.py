"""Label-free retrieval members for ogbl-wikikg2, vectorised on the GPU.

For a query row (src, rel, dir) and each candidate c (positive + 500
OGB negatives), holders(c) = training sources already linked to c
under (rel, dir), src itself excluded. Members written to ens_cache/
in the standard sp/sn/rel layout:

  analogy_<tag>     max over holders of cos(e_src, e_holder)^3
  analogy_<tag>_t3  mean of the top-3 holder cosines, cubed
  holders           log1p(#holders)  — model-free popularity of c
                    under (rel, dir); the "does c take this relation
                    at all" signal that uniform negatives mostly fail

Rows/candidates with no holder get -1 (as on biokg). Only training
edges are read: validation and test edges are never looked up.

Scale differences from analogy_member.py (biokg): everything is
batched — a chunk of rows gets its full cosine row against the fp16
unit-norm table (H19: fp16 serving table), holder segments come from
a sorted (rel, target) key array via searchsorted, and the per-pair
max / top-3 are segment reductions. Wikidata hubs (a target with 1e6
holders) are capped at --cap holders drawn uniformly at random
(triples are shuffled before the key sort), so the max is over a
random --cap-subset for those; the uncapped count still feeds
`holders`.

Usage: python retrieval_wiki.py --device cuda --model lever_x.pt \
           --tag s0 --split valid
"""

import argparse
import os
import time

import numpy as np
import torch

from resonate import cnorm
from train_wiki import load, load_model


def build_holders(hh, rr, tt, n_ent, d, seed=0):
    """Sorted (rel*N + target) keys -> contiguous holder segments."""
    s = hh if d == 0 else tt
    o = tt if d == 0 else hh
    perm = np.random.default_rng(seed).permutation(len(s))
    key = rr[perm].astype(np.int64) * n_ent + o[perm]
    order = np.argsort(key, kind="stable")
    key, hold = key[order], s[perm][order]
    uniq, start, cnt = np.unique(key, return_index=True, return_counts=True)
    return uniq, start, cnt, hold


@torch.no_grad()
def features(En, src, rel, cands, uniq, start, cnt, hold, n_ent, cap):
    """src (C,), rel (C,), cands (C,501) on device. Returns
    (mx, t3, nh) each (C,501) float32: max cos, top-3 mean cos, and
    the uncapped holder count (src excluded, approximately)."""
    C, K = cands.shape
    sims = En[src] @ En.t()                          # (C, N) fp16
    q = rel[:, None] * n_ent + cands                 # (C, K) keys
    qf = q.reshape(-1)
    pos = torch.searchsorted(uniq, qf).clamp(max=len(uniq) - 1)
    found = uniq[pos] == qf
    n = torch.where(found, cnt[pos], torch.zeros_like(pos))
    take = n.clamp(max=cap)
    tot = int(take.sum())
    mx = torch.full((C * K,), -1.0, device=src.device)
    t3 = torch.full((C * K,), -1.0, device=src.device)
    if tot == 0:
        return (mx.view(C, K), t3.view(C, K),
                n.view(C, K).float())
    pair = torch.repeat_interleave(torch.arange(C * K, device=src.device),
                                   take)
    off = torch.arange(tot, device=src.device) \
        - torch.repeat_interleave(take.cumsum(0) - take, take)
    hidx = hold[start[pos][pair] + off]
    row = pair // K
    val = sims[row, hidx].float()
    self_h = hidx == src[row]
    val = torch.where(self_h, torch.full_like(val, -2.0), val)
    n = n - torch.zeros_like(n).scatter_add_(
        0, pair, self_h.long())                      # src is not a holder
    mx.scatter_reduce_(0, pair, val, reduce="amax", include_self=True)
    mx = torch.where(mx < -1.5, torch.full_like(mx, -1.0), mx)
    # top-3 mean: sort by value desc, then stable by pair -> segments
    v_s, o1 = torch.sort(val, descending=True)
    p_s, o2 = torch.sort(pair[o1], stable=True)
    v_s = v_s[o2]
    new = torch.ones_like(p_s, dtype=torch.bool)
    new[1:] = p_s[1:] != p_s[:-1]
    seg_start = torch.cummax(torch.where(new, torch.arange(tot, device=src.device),
                                         torch.zeros_like(p_s)), 0)[0]
    rank = torch.arange(tot, device=src.device) - seg_start
    keep = (rank < 3) & (v_s > -1.5)
    ssum = torch.zeros(C * K, device=src.device).scatter_add_(
        0, p_s[keep], v_s[keep])
    scnt = torch.zeros(C * K, device=src.device).scatter_add_(
        0, p_s[keep], torch.ones_like(v_s[keep]))
    t3 = torch.where(scnt > 0, ssum / scnt.clamp(min=1), t3)
    return mx.view(C, K), t3.view(C, K), n.view(C, K).float()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tag", required=True, help="suffix: analogy_<tag>")
    p.add_argument("--split", default="valid", choices=["valid", "test"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--chunk", type=int, default=512)
    p.add_argument("--cap", type=int, default=128)
    p.add_argument("--out-dir", default="ens_cache")
    p.add_argument("--data-root", default="data_ogb")
    p.add_argument("--keep-frac", type=float, default=1.0,
                   help="keep this fraction of training triples (seeded) before "
                        "building the graph: simulates evidence scarcity")
    p.add_argument("--keep-seed", type=int, default=0)
    p.add_argument("--suffix", default="",
                   help="appended to member names (e.g. _aug)")
    p.add_argument("--extra-edges", default=None,
                   help="npz (h, r, t) of proposed edges added to the holder maps")
    args = p.parse_args()
    dev = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    split, n_ent = load(args.data_root)
    part = split[args.split]
    h = np.asarray(part["head"]).astype(np.int64)
    r = np.asarray(part["relation"]).astype(np.int64)
    t = np.asarray(part["tail"]).astype(np.int64)
    neg_h = np.asarray(part["head_neg"]).astype(np.int64)
    neg_t = np.asarray(part["tail_neg"]).astype(np.int64)
    N = len(h)
    tr = split["train"]
    hh = np.asarray(tr["head"]).astype(np.int64)
    rr = np.asarray(tr["relation"]).astype(np.int64)
    tt = np.asarray(tr["tail"]).astype(np.int64)
    if args.keep_frac < 1.0:
        keep = np.random.default_rng(args.keep_seed).random(len(hh)) < args.keep_frac
        hh, rr, tt = hh[keep], rr[keep], tt[keep]
        print(f"thinned training graph: keeping {keep.mean()*100:.0f}% of triples", flush=True)
    if args.extra_edges:
        ex = np.load(args.extra_edges)
        hh = np.concatenate([hh, ex["h"]]); rr = np.concatenate([rr, ex["r"]]); tt = np.concatenate([tt, ex["t"]])
        print(f"extra edges: +{len(ex['h']):,}", flush=True)

    model, _ = load_model(args.model, n_ent, dev)
    En = torch.view_as_real(cnorm(model.table())).reshape(n_ent, -1) \
        .contiguous().half()                         # cos = <real views>
    del model
    torch.cuda.empty_cache()

    sp = {k: np.full(2 * N, -1.0, np.float16) for k in ("mx", "t3", "nh")}
    sn = {k: np.full((2 * N, 500), -1.0, np.float16) for k in sp}
    t0 = time.time()
    for d in (0, 1):
        uniq, start, cnt, hold = build_holders(hh, rr, tt, n_ent, d)
        uniq, start, cnt, hold = (torch.from_numpy(x).to(dev)
                                  for x in (uniq, start, cnt, hold))
        print(f"dir {d}: {len(uniq):,} (rel,target) keys, max holders "
              f"{int(cnt.max())}, keys over cap {(cnt > args.cap).float().mean()*100:.1f}%",
              flush=True)
        src_a = h if d == 0 else t
        pos_a = t if d == 0 else h
        cand_a = neg_t if d == 0 else neg_h
        for i in range(0, N, args.chunk):
            sl = slice(i, min(i + args.chunk, N))
            src = torch.from_numpy(src_a[sl]).to(dev)
            rel = torch.from_numpy(r[sl]).to(dev)
            cands = torch.from_numpy(
                np.concatenate([pos_a[sl][:, None], cand_a[sl]], 1)).to(dev)
            mx, t3, nh = features(En, src, rel, cands, uniq, start, cnt,
                                  hold, n_ent, args.cap)
            mx = torch.sign(mx) * mx.abs().pow(3)
            t3 = torch.sign(t3) * t3.abs().pow(3)
            nh = torch.where(nh > 0, torch.log1p(nh), torch.full_like(nh, -1.0))
            rows = slice(d * N + sl.start, d * N + sl.stop)
            for k, v in (("mx", mx), ("t3", t3), ("nh", nh)):
                v = v.cpu().numpy().astype(np.float16)
                sp[k][rows] = v[:, 0]
                sn[k][rows] = v[:, 1:]
            if (i // args.chunk) % 200 == 0:
                print(f"dir {d} row {i}/{N} ({time.time()-t0:.0f}s)",
                      flush=True)

    rel_out = np.concatenate([r, r])
    names = {"mx": f"analogy_{args.tag}{args.suffix}",
             "t3": f"analogy_{args.tag}_t3{args.suffix}",
             "nh": f"holders{args.suffix}"}
    for k, name in names.items():
        out = os.path.join(args.out_dir, f"{name}.{args.split}.npz")
        np.savez(out, sp=sp[k], sn=sn[k], rel=rel_out)
        cov = (sp[k] > -1).mean()
        if args.split == "test":
            print(f"{name}: test features cached (MRR not computed; "
                  f"positive has holders on {cov*100:.0f}% of rows) -> {out}",
                  flush=True)
        else:
            mrr = (1.0 / ((sn[k] >= sp[k][:, None]).sum(1) + 1)).mean()
            print(f"{name}: valid MRR alone {mrr:.4f} (positive has "
                  f"holders on {cov*100:.0f}% of rows) -> {out}", flush=True)
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
