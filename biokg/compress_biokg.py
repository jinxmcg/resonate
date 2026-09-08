"""CP-B1: init-only sweep of the degree-tiered table on ogbl-biokg
(COMPACT_K.md CP1/CP3b, ported to the biokg model class).

Entities are grouped by training degree; inside each tier the trained rows
(real view of the complex row) are k-means clustered and each cluster is
replaced by its top principal directions, i.e. a narrow coefficient vector per
entity plus K shared projections and offsets per tier
(`resonate_tiered_biokg.TieredTableResonatE.init_from_wide`). Operators and
temperature are the wide model's, copied and frozen. NO TRAINING: this maps the
size-accuracy frontier before any refit. Every configuration is evaluated on
validation with the official OGB Evaluator (500 type-matched negatives per
direction). No test read.

Usage:
  PYTHONPATH=.. python compress_biokg.py --model checkpoints/sparse_s0.pt \
      --cutoffs 5 8 32 128 1024 \
      --configs 144,144,144,144,144,144 4,8,16,48,144,144:256,256,256,16,1,1 \
      --device cuda --data-root ../../geocore/data_ogb
"""
import argparse
import time

import numpy as np
import torch

from resonate import ResonatE
from resonate_tiered_biokg import TieredTableResonatE
from train_ogb import load, globalize, eval_split


def parse_cfg(tok, T):
    """'w0,..,wT-1[:K0,..,KT-1]' -> (widths, Ks)."""
    w, _, ks = tok.partition(":")
    widths = [int(x) for x in w.split(",")]
    assert len(widths) == T, f"{tok}: {len(widths)} widths for {T} tiers"
    K = [int(x) for x in ks.split(",")] if ks else [1] * T
    assert len(K) == T, f"{tok}: {len(K)} subspace counts for {T} tiers"
    return widths, K


def load_wide(path, n_ent, n_rel, dev):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    ca = ck.get("args", {})
    m = ResonatE(n_entities=n_ent, n_relations=n_rel, k=ca.get("k", 12), block=True,
                 block_size=ca.get("block_size", 4),
                 tied_reverse=ca.get("tied_reverse", False),
                 ent_bias=ca.get("ent_bias", False),
                 rel_gain=ca.get("rel_gain", False)).to(dev)
    m.load_state_dict(ck["model"])
    m.eval()
    for q in m.parameters():
        q.requires_grad_(False)
    return m, ck


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="checkpoints/sparse_s0.pt")
    p.add_argument("--cutoffs", type=int, nargs="+", default=[5, 8, 32, 128, 1024])
    p.add_argument("--configs", nargs="+", required=True,
                   help="widths per tier in complex dims, optionally ':' K per tier")
    p.add_argument("--device", default="cuda")
    p.add_argument("--data-root", default="data_ogb")
    p.add_argument("--chunk", type=int, default=512)
    p.add_argument("--no-wide-eval", action="store_true")
    p.add_argument("--save-prefix", default=None,
                   help="if set, save each configuration as <prefix>_<i>.pt")
    a = p.parse_args()
    dev = torch.device(a.device)

    t0 = time.time()
    split, offset, n_ent, types, num_nodes = load(a.data_root)
    tr, va = split["train"], split["valid"]
    h, r, t = globalize(tr, offset)
    n_rel = 2 * (int(r.max()) + 1)
    deg = np.bincount(np.concatenate([h, t]), minlength=n_ent)
    cutoffs = np.array(a.cutoffs)
    tier = np.searchsorted(cutoffs, deg, side="right")
    T = len(cutoffs) + 1
    vh, _, vt = globalize(va, offset)
    ans = np.concatenate([vt, vh])                    # validation answers, both directions
    print(f"ogbl-biokg loaded in {time.time()-t0:.0f}s: {n_ent:,} entities, "
          f"{n_rel//2} relations, {len(h):,} train triples", flush=True)
    bounds = [0] + list(cutoffs) + [deg.max() + 1]
    print(f"tiers by training degree {cutoffs.tolist()}:")
    for ti in range(T):
        m_ = tier == ti
        print(f"  tier {ti} [{bounds[ti]}, {bounds[ti+1]-1}]: {int(m_.sum()):>6,} entities "
              f"({100*m_.mean():5.1f}%), {100*m_[ans].mean():5.1f}% of validation answers")

    wide, ck = load_wide(a.model, n_ent, n_rel, dev)
    M2 = 2 * wide.m
    E0 = torch.view_as_real(wide.E.detach()).reshape(n_ent, M2).float().contiguous()
    full = n_ent * M2
    ops = wide.n_params() - full
    print(f"wide model {a.model}: k={wide.k} b={wide.block_size} M={wide.m} complex; "
          f"table {full:,} + operators {ops:,} = {wide.n_params():,}", flush=True)
    if not a.no_wide_eval:
        eval_split(wide, va, offset, n_rel, dev, chunk=a.chunk, label="valid wide (sanity)")
    del wide
    torch.cuda.empty_cache()

    for ci, tok in enumerate(a.configs):
        widths, K = parse_cfg(tok, T)
        t1 = time.time()
        m = TieredTableResonatE(n_ent, n_rel, tier, widths, k=ck["args"].get("k", 12),
                                block_size=ck["args"].get("block_size", 4), device=dev,
                                rel_gain=ck["args"].get("rel_gain", False), subspaces=K)
        m.load_state_dict({key: v for key, v in ck["model"].items() if key != "E"}, strict=False)
        kept = m.init_from_wide(E0)
        m.eval()
        for q in m.parameters():
            q.requires_grad_(False)
        m.build_eval_table()
        tp = m.table_n_params()
        lbl = f"valid ({tok})"
        mrr = eval_split(m, va, offset, n_rel, dev, chunk=a.chunk, label=lbl)
        print(f"CP-B1 cfg {tok}: widths {widths} K {m.K} -> table {tp:,} "
              f"({100*tp/full:.1f}% of the wide table), total {m.n_params():,}  "
              f"valid MRR {mrr:.4f}  variance kept per tier "
              f"{[round(x, 3) for x in kept]}  ({time.time()-t1:.0f}s)", flush=True)
        if a.save_prefix:
            torch.save({"model": m.state_dict(), "offset": offset, "n_rel": n_rel,
                        "args": {**ck["args"], "tiers": ",".join(map(str, a.cutoffs)),
                                 "widths": ",".join(map(str, widths)),
                                 "subspaces": ",".join(map(str, m.K)),
                                 "tiered_from": a.model}},
                       f"{a.save_prefix}_{ci}.pt")
        del m
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
