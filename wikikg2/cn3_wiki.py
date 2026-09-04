"""Three-hop neighbourhood member for ogbl-wikikg2 (model-free).

For candidate c and query entity q, weighted count of length-3 paths
c - n1 - n2 - q in the undirected relation-agnostic training graph,
each path weighted Adamic-Adar style by 1/log(2+deg n1) * 1/log(2+deg n2)
so that paths through hubs ("human", "United States") contribute ~0.
Degenerate paths (n1 == q, n2 == c) are excluded. Neighbour lists are
sampled to --cap on both expansion steps, so the feature is a capped
estimate for high-degree candidates (whose hub-mediated paths are
down-weighted anyway).

Cost per query row: 501 * cap^2 key lookups; cap 16 -> 128k, all
859k rows in a few minutes on the GPU.

Writes ens_cache/cn3_aa.<split>.npz.
Usage: python cn3_wiki.py --device cuda --split valid --cap 16
"""

import argparse
import os
import time

import numpy as np
import torch

from cn_wiki import build_graph
from train_wiki import load


def expand(nodes, deg, indptr, nbr, cap):
    """nodes (P,) -> (parent index (T,), neighbour (T,)) with each node's
    list cut at cap."""
    d = deg[nodes].clamp(max=cap)
    tot = int(d.sum())
    dev = nodes.device
    if tot == 0:
        e = torch.zeros(0, dtype=torch.long, device=dev)
        return e, e
    parent = torch.repeat_interleave(torch.arange(len(nodes), device=dev), d)
    off = torch.arange(tot, device=dev) \
        - torch.repeat_interleave(d.cumsum(0) - d, d)
    return parent, nbr[indptr[nodes][parent] + off]


@torch.no_grad()
def features(q, cands, deg, indptr, nbr, keys, n_ent, cap, wlog):
    C, K = cands.shape
    dev = q.device
    cf = cands.reshape(-1)
    qk = torch.repeat_interleave(q, K)
    out = torch.zeros(C * K, device=dev)
    p1, n1 = expand(cf, deg, indptr, nbr, cap)      # c -> n1
    if len(n1) == 0:
        return out.view(C, K)
    keep = n1 != qk[p1]                               # n1 == q: 1-hop
    p1, n1 = p1[keep], n1[keep]
    p2, n2 = expand(n1, deg, indptr, nbr, cap)      # n1 -> n2
    if len(n2) == 0:
        return out.view(C, K)
    pair = p1[p2]                                     # candidate pair id
    qq = qk[pair]
    ok = (n2 != cf[pair]) & (n2 != qq)
    k2 = n2 * n_ent + qq                              # is n2 - q an edge?
    pos = torch.searchsorted(keys, k2).clamp(max=len(keys) - 1)
    hit = ((keys[pos] == k2) & ok).float()
    w = hit * wlog[n1[p2]] * wlog[n2]
    out.scatter_add_(0, pair, w)
    return out.view(C, K)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="valid", choices=["valid", "test"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--chunk", type=int, default=64)
    p.add_argument("--cap", type=int, default=16)
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
    deg_t = torch.from_numpy(deg).to(dev)
    wlog = 1.0 / torch.log(2.0 + deg_t.float())
    indptr_t = torch.from_numpy(indptr).to(dev)
    nbr_t = torch.from_numpy(nbr).to(dev)
    keys_t = torch.from_numpy(keys).to(dev)
    sp = np.zeros(2 * N, np.float16)
    sn = np.zeros((2 * N, 500), np.float16)
    for d in (0, 1):
        q_a = h if d == 0 else t
        pos_a = t if d == 0 else h
        cand_a = neg_t if d == 0 else neg_h
        for i in range(0, N, args.chunk):
            sl = slice(i, min(i + args.chunk, N))
            q = torch.from_numpy(q_a[sl]).to(dev)
            cands = torch.from_numpy(
                np.concatenate([pos_a[sl][:, None], cand_a[sl]], 1)).to(dev)
            v = features(q, cands, deg_t, indptr_t, nbr_t, keys_t, n_ent,
                         args.cap, wlog)
            v = torch.log1p(v).cpu().numpy().astype(np.float16)
            rows = slice(d * N + sl.start, d * N + sl.stop)
            sp[rows] = v[:, 0]
            sn[rows] = v[:, 1:]
            if (i // args.chunk) % 1000 == 0:
                print(f"dir {d} row {i}/{N} ({time.time()-t0:.0f}s)", flush=True)
    out = os.path.join(args.out_dir, f"cn3_aa{args.suffix}.{args.split}.npz")
    np.savez(out, sp=sp, sn=sn, rel=np.concatenate([r, r]))
    if args.split != "test":
        for lab, sl in (("tail", slice(0, N)), ("head", slice(N, 2 * N))):
            mrr = (1.0 / ((sn[sl] >= sp[sl][:, None]).sum(1) + 1)).mean()
            print(f"cn3_aa: valid MRR alone {lab} {mrr:.4f} (positive nonzero "
                  f"{(sp[sl] > 0).mean()*100:.0f}%)", flush=True)
    print(f"done in {time.time()-t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
