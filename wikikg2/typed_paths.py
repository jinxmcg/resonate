"""Typed two-hop path member (fail-fast stage 1).

A path candidate -> n -> query carries the RELATION TYPES of its two
edges (directed ops: r forward, r+R reverse). For a query relation r
the informative path types are those that connect the head and tail of
training triples of r far more often than they connect random pairs:

    lo[r, (o1, o2)] = log P(type | (h, r, t) in train) - log P(type | random pair)

estimated from a sample of training triples (h, r, t) — paths h -> n -> t
— against paths h -> n -> t' for random t'. Only training edges are read.

The member score of a candidate is the sum over its path types of
lo[r, type] * log1p(count). Head queries (?, r, t): paths run candidate
-> n -> t; tail queries (h, r, ?): paths h -> n -> candidate. Both use
lo[r, .] in head-to-tail orientation.

Usage: python typed_paths.py --device cuda --split valid --out-dir ens_cache
"""

import argparse
import os
import time

import numpy as np
import torch

from train_wiki import load


def build_typed(h, r, t, n_ent, R, seed=0):
    """Directed typed CSR (random neighbour order) + sorted edge keys
    (a*N+b)*2R+op for membership/op lookup between two nodes."""
    src = np.concatenate([h, t]); dst = np.concatenate([t, h])
    op = np.concatenate([r, r + R]).astype(np.int64)
    perm = np.random.default_rng(seed).permutation(len(src))
    src, dst, op = src[perm], dst[perm], op[perm]
    o = np.argsort(src, kind="stable")
    src, dst, op = src[o], dst[o], op[o]
    deg = np.bincount(src, minlength=n_ent)
    indptr = np.concatenate([[0], np.cumsum(deg)])
    keys = np.unique((src * n_ent + dst) * (2 * R) + op)   # dedup repeated triples
    return deg, indptr, dst, op, keys


@torch.no_grad()
def path_types(start, end, deg, indptr, nbr, nop, keys, n_ent, R, cap):
    """start (P,), end (P,) -> (pair index (T,), path type (T,)) for all
    typed 2-hop paths start -> n -> end, n != start,end; at most 2 ops
    per (n, end) pair are taken."""
    dev = start.device
    d = deg[start].clamp(max=cap)
    tot = int(d.sum())
    if tot == 0:
        e = torch.zeros(0, dtype=torch.long, device=dev)
        return e, e
    pair = torch.repeat_interleave(torch.arange(len(start), device=dev), d)
    off = torch.arange(tot, device=dev) - torch.repeat_interleave(d.cumsum(0) - d, d)
    idx = indptr[start][pair] + off
    n, o1 = nbr[idx], nop[idx]
    q = end[pair]
    ok = (n != q) & (n != start[pair])
    base = (n * n_ent + q) * (2 * R)
    lo = torch.searchsorted(keys, base)
    hi = torch.searchsorted(keys, base + 2 * R)
    cnt = (hi - lo).clamp(max=2) * ok.long()
    tot2 = int(cnt.sum())
    if tot2 == 0:
        e = torch.zeros(0, dtype=torch.long, device=dev)
        return e, e
    rep = torch.repeat_interleave(torch.arange(len(n), device=dev), cnt)
    off2 = torch.arange(tot2, device=dev) - torch.repeat_interleave(cnt.cumsum(0) - cnt, cnt)
    o2 = keys[lo[rep] + off2] % (2 * R)
    return pair[rep], o1[rep] * (2 * R) + o2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="valid", choices=["valid", "test"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--chunk", type=int, default=128)
    p.add_argument("--cap", type=int, default=48)
    p.add_argument("--n-train", type=int, default=3000000, help="training triples sampled for the table")
    p.add_argument("--min-count", type=int, default=20)
    p.add_argument("--out-dir", default="ens_cache")
    p.add_argument("--table", default="typed_lo.npz")
    p.add_argument("--data-root", default="data_ogb")
    p.add_argument("--keep-frac", type=float, default=1.0,
                   help="keep this fraction of training triples (seeded) before "
                        "building the graph: simulates evidence scarcity")
    p.add_argument("--keep-seed", type=int, default=0)
    args = p.parse_args()
    dev = torch.device(args.device)
    split, n_ent = load(args.data_root)
    tr = split["train"]
    h, r, t = (np.asarray(tr[k]).astype(np.int64) for k in ("head", "relation", "tail"))
    R = int(r.max()) + 1
    if args.keep_frac < 1.0:
        keep = np.random.default_rng(args.keep_seed).random(len(h)) < args.keep_frac
        h, r, t = h[keep], r[keep], t[keep]
        print(f"thinned training graph: keeping {keep.mean()*100:.0f}% of triples", flush=True)
    t0 = time.time()
    deg, indptr, nbr, nop, keys = build_typed(h, r, t, n_ent, R)
    G = [torch.from_numpy(x).to(dev) for x in (deg, indptr, nbr, nop, keys)]
    print(f"typed graph: {len(keys)//2:,} edges, {2*R} ops ({time.time()-t0:.0f}s)", flush=True)
    NT = (2 * R) ** 2

    if os.path.exists(args.table):
        d = np.load(args.table); lo_tab = torch.from_numpy(d["lo"]).to(dev)
        print(f"loaded {args.table}", flush=True)
    else:
        rng = np.random.default_rng(0)
        pick = rng.choice(len(h), size=min(args.n_train, len(h)), replace=False)
        pos = torch.zeros(R * NT, device=dev)      # count[r, type]
        neg = torch.zeros(NT, device=dev)          # base[type] over random pairs
        npos = torch.zeros(R, device=dev); nneg = 0
        for i in range(0, len(pick), 4096):
            sl = pick[i:i + 4096]
            hs = torch.from_numpy(h[sl]).to(dev); ts = torch.from_numpy(t[sl]).to(dev)
            rs = torch.from_numpy(r[sl]).to(dev)
            pr, ty = path_types(hs, ts, *G, n_ent, R, args.cap)
            # once per (pair, type)
            u = torch.unique(pr * NT + ty)
            pos.index_add_(0, rs[u // NT] * NT + u % NT, torch.ones(len(u), device=dev))
            npos.index_add_(0, rs, torch.ones(len(sl), device=dev))
            tr_ = torch.randint(0, n_ent, (len(sl),), device=dev)
            pr, ty = path_types(hs, tr_, *G, n_ent, R, args.cap)
            u = torch.unique(pr * NT + ty)
            neg.index_add_(0, u % NT, torch.ones(len(u), device=dev))
            nneg += len(sl)
            if (i // 4096) % 100 == 0:
                print(f"table: {i}/{len(pick)} ({time.time()-t0:.0f}s)", flush=True)
        pos = pos.view(R, NT)
        lo_tab = torch.log((pos + 1) / (npos[:, None] + 10)) - torch.log((neg + 1) / (nneg + 10))[None, :]
        lo_tab = torch.where(pos >= args.min_count, lo_tab, torch.zeros_like(lo_tab))
        np.savez(args.table, lo=lo_tab.cpu().numpy())
        nz = int((lo_tab != 0).sum())
        print(f"table: {nz:,} (relation, path type) entries with >= {args.min_count} support; "
              f"mean log-odds of kept entries {lo_tab[lo_tab != 0].mean():.2f} ({time.time()-t0:.0f}s)", flush=True)

    part = split[args.split]
    vh, vr, vt = (np.asarray(part[k]).astype(np.int64) for k in ("head", "relation", "tail"))
    neg_h = np.asarray(part["head_neg"]).astype(np.int64)
    neg_t = np.asarray(part["tail_neg"]).astype(np.int64)
    N = len(vh)
    sp = np.zeros(2 * N, np.float16); sn = np.zeros((2 * N, 500), np.float16)
    for d_ in (0, 1):
        for i in range(0, N, args.chunk):
            sl = slice(i, min(i + args.chunk, N))
            C = sl.stop - sl.start
            rel = torch.from_numpy(vr[sl]).to(dev)
            if d_ == 0:   # tail query: paths h -> n -> candidate
                cands = torch.from_numpy(np.concatenate([vt[sl][:, None], neg_t[sl]], 1)).to(dev)
                start = torch.from_numpy(vh[sl]).to(dev).repeat_interleave(501); end = cands.reshape(-1)
            else:         # head query: paths candidate -> n -> t
                cands = torch.from_numpy(np.concatenate([vh[sl][:, None], neg_h[sl]], 1)).to(dev)
                start = cands.reshape(-1); end = torch.from_numpy(vt[sl]).to(dev).repeat_interleave(501)
            pr, ty = path_types(start, end, *G, n_ent, R, args.cap)
            score = torch.zeros(C * 501, device=dev)
            if len(pr):
                key = pr * NT + ty
                u, cnt = torch.unique(key, return_counts=True)
                pu, tu = u // NT, u % NT
                w = lo_tab[rel.repeat_interleave(501)[pu], tu]
                score.index_add_(0, pu, w * torch.log1p(cnt.float()))
            v = score.view(C, 501).cpu().numpy().astype(np.float16)
            rows = slice(d_ * N + sl.start, d_ * N + sl.stop)
            sp[rows] = v[:, 0]; sn[rows] = v[:, 1:]
            if (i // args.chunk) % 500 == 0:
                print(f"dir {d_} row {i}/{N} ({time.time()-t0:.0f}s)", flush=True)
    out = os.path.join(args.out_dir, f"typed.{args.split}.npz")
    np.savez(out, sp=sp, sn=sn, rel=np.concatenate([vr, vr]))
    if args.split == "valid":
        for lab, sl in (("tail", slice(0, N)), ("head", slice(N, 2 * N))):
            mrr = (1.0 / ((sn[sl] >= sp[sl][:, None]).sum(1) + 1)).mean()
            print(f"typed: valid MRR alone {lab} {mrr:.4f} (positive nonzero {(sp[sl] != 0).mean()*100:.0f}%)", flush=True)
    print(f"done in {time.time()-t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
