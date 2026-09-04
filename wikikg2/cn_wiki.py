"""Two-hop neighbourhood members for ogbl-wikikg2 (model-free, label-free).

For a query entity q and a candidate c, look at the training graph as
an undirected, relation-agnostic graph and count the length-2 paths
c - n - q (common neighbours):

  cn        log1p(#common neighbours)
  cn_aa     Adamic-Adar: sum over common n of 1/log(2 + deg n) — a
            hub like "human" (deg 1e6) contributes ~0, a shared small
            entity (a college, a town, a book) contributes ~1
  linked    1 if c and q are already joined by ANY training edge
            (of another relation), else 0

Head queries (?, r, t): q = t and c ranges over the 500 head
candidates + the positive; the signal is "is this candidate connected
to t by other paths" (Douglas Adams - Cambridge - United Kingdom), the
StarGraph-style neighbourhood evidence the spectrum alone lacks for a
rare head. Tail queries: q = h symmetrically.

Only training edges are read. Candidate-side neighbour lists are
capped at --cap (random order) so a hub candidate costs O(cap); the
query side is a membership test against the sorted edge key array.

Usage: python cn_wiki.py --device cuda --split valid
"""

import argparse
import os
import time

import numpy as np
import torch

from train_wiki import load


def build_graph(hh, tt, n_ent, seed=0):
    """Undirected dedup'd edges -> (deg, indptr, nbr) CSR with random
    neighbour order, plus the sorted key array a*N+b for both orders."""
    a = np.minimum(hh, tt)
    b = np.maximum(hh, tt)
    keep = a != b
    key = np.unique(a[keep] * n_ent + b[keep])
    a, b = key // n_ent, key % n_ent
    src = np.concatenate([a, b])
    dst = np.concatenate([b, a])
    perm = np.random.default_rng(seed).permutation(len(src))
    src, dst = src[perm], dst[perm]
    o = np.argsort(src, kind="stable")
    src, dst = src[o], dst[o]
    deg = np.bincount(src, minlength=n_ent)
    indptr = np.concatenate([[0], np.cumsum(deg)])
    keys = np.sort(src * n_ent + dst)
    return deg, indptr, dst, keys


@torch.no_grad()
def features(q, cands, deg, indptr, nbr, keys, n_ent, cap, wlog):
    C, K = cands.shape
    cf = cands.reshape(-1)
    d = deg[cf].clamp(max=cap)
    tot = int(d.sum())
    dev = q.device
    cn = torch.zeros(C * K, device=dev)
    aa = torch.zeros(C * K, device=dev)
    qk = torch.repeat_interleave(q, K)               # query per pair
    lk = cf * n_ent + qk
    pos = torch.searchsorted(keys, lk).clamp(max=len(keys) - 1)
    linked = (keys[pos] == lk).float()
    if tot:
        pair = torch.repeat_interleave(torch.arange(C * K, device=dev), d)
        off = torch.arange(tot, device=dev) \
            - torch.repeat_interleave(d.cumsum(0) - d, d)
        n = nbr[indptr[cf][pair] + off]
        qq = qk[pair]
        k2 = n * n_ent + qq
        pos = torch.searchsorted(keys, k2).clamp(max=len(keys) - 1)
        hit = ((keys[pos] == k2) & (n != qq)).float()
        cn.scatter_add_(0, pair, hit)
        aa.scatter_add_(0, pair, hit * wlog[n])
    return (torch.log1p(cn).view(C, K), aa.view(C, K), linked.view(C, K))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="valid", choices=["valid", "test"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--chunk", type=int, default=256)
    p.add_argument("--cap", type=int, default=64)
    p.add_argument("--out-dir", default="ens_cache")
    p.add_argument("--data-root", default="data_ogb")
    p.add_argument("--keep-frac", type=float, default=1.0,
                   help="keep this fraction of training triples (seeded) before "
                        "building the graph: simulates evidence scarcity")
    p.add_argument("--keep-seed", type=int, default=0)
    p.add_argument("--suffix", default="",
                   help="appended to member names (e.g. _aug)")
    p.add_argument("--extra-edges", default=None,
                   help="npz (h, r, t) of proposed edges to add to the "
                        "training graph (self-augmentation)")
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
    t0 = time.time()
    hh_, tt_ = np.asarray(tr["head"]).astype(np.int64), np.asarray(tr["tail"]).astype(np.int64)
    if args.keep_frac < 1.0:
        keep = np.random.default_rng(args.keep_seed).random(len(hh_)) < args.keep_frac
        hh_, tt_ = hh_[keep], tt_[keep]
        print(f"thinned training graph: keeping {keep.mean()*100:.0f}% of triples", flush=True)
    if args.extra_edges:
        ex = np.load(args.extra_edges)
        hh_ = np.concatenate([hh_, ex["h"]]); tt_ = np.concatenate([tt_, ex["t"]])
        print(f"extra edges: +{len(ex['h']):,}", flush=True)
    deg, indptr, nbr, keys = build_graph(hh_, tt_, n_ent)
    print(f"undirected train graph: {len(keys)//2:,} edges, mean deg "
          f"{deg.mean():.1f}, max deg {deg.max():,}, nodes over cap "
          f"{(deg > args.cap).mean()*100:.1f}% ({time.time()-t0:.0f}s)",
          flush=True)
    deg_t = torch.from_numpy(deg).to(dev)
    wlog = 1.0 / torch.log(2.0 + deg_t.float())
    indptr_t = torch.from_numpy(indptr).to(dev)
    nbr_t = torch.from_numpy(nbr).to(dev)
    keys_t = torch.from_numpy(keys).to(dev)

    names = ("cn", "cn_aa", "linked")
    sp = {k: np.zeros(2 * N, np.float16) for k in names}
    sn = {k: np.zeros((2 * N, 500), np.float16) for k in names}
    for d in (0, 1):
        q_a = h if d == 0 else t
        pos_a = t if d == 0 else h
        cand_a = neg_t if d == 0 else neg_h
        for i in range(0, N, args.chunk):
            sl = slice(i, min(i + args.chunk, N))
            q = torch.from_numpy(q_a[sl]).to(dev)
            cands = torch.from_numpy(
                np.concatenate([pos_a[sl][:, None], cand_a[sl]], 1)).to(dev)
            outs = features(q, cands, deg_t, indptr_t, nbr_t, keys_t,
                            n_ent, args.cap, wlog)
            rows = slice(d * N + sl.start, d * N + sl.stop)
            for k, v in zip(names, outs):
                v = v.cpu().numpy().astype(np.float16)
                sp[k][rows] = v[:, 0]
                sn[k][rows] = v[:, 1:]
            if (i // args.chunk) % 400 == 0:
                print(f"dir {d} row {i}/{N} ({time.time()-t0:.0f}s)", flush=True)

    rel_out = np.concatenate([r, r])
    for k in names:
        out = os.path.join(args.out_dir, f"{k}{args.suffix}.{args.split}.npz")
        np.savez(out, sp=sp[k], sn=sn[k], rel=rel_out)
        if args.split == "test":
            print(f"{k}: test features cached -> {out}", flush=True)
        else:
            for lab, sl in (("tail", slice(0, N)), ("head", slice(N, 2 * N))):
                mrr = (1.0 / ((sn[k][sl] >= sp[k][sl][:, None]).sum(1) + 1)).mean()
                nz = (sp[k][sl] > 0).mean()
                print(f"{k}: valid MRR alone {lab} {mrr:.4f} (positive "
                      f"nonzero {nz*100:.0f}%)", flush=True)
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
