"""CP1: post-hoc tiered compression of a trained table (COMPACT_K.md, CP1).

Entities are grouped by training degree; in each tier the trained rows (real
view of the complex row, centred) are projected onto their top principal
directions and reconstructed. Parameters counted = narrow coefficients per
entity + one shared projection per tier. Evaluates each configuration on
validation with the official Evaluator. No training, no test.

Usage: python compress_wiki.py --model teachers/model_dist_s1.bf16.pt --cutoffs 8 64 1024 \
           --configs 64,64,64,64 8,16,36,64 4,8,16,64 --device cuda
"""
import argparse
import time

import numpy as np
import torch

from train_wiki import load, load_model, eval_split


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--cutoffs", type=int, nargs="+", default=[8, 64, 1024])
    p.add_argument("--configs", nargs="+", default=["64,64,64,64", "8,16,36,64", "4,8,16,64", "16,32,64,64", "8,8,16,64"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--data-root", default="data_ogb")
    p.add_argument("--table-dtype", default=None, choices=["fp32", "fp16", "bf16"])
    a = p.parse_args()
    dev = torch.device(a.device)
    split, n_ent = load(a.data_root)
    tr = split["train"]
    deg = np.bincount(np.concatenate([np.asarray(tr["head"]), np.asarray(tr["tail"])]), minlength=n_ent)
    tier = np.searchsorted(np.array(a.cutoffs), deg, side="right")      # 0..len(cutoffs)
    T = len(a.cutoffs) + 1
    counts = [int((tier == t).sum()) for t in range(T)]
    m, ck = load_model(a.model, n_ent, dev, table_dtype=a.table_dtype)
    E0 = m.E_real.detach().float().clone()                                # (N, 2M)
    M2 = E0.shape[1]; Mc = M2 // 2
    print(f"table {n_ent:,} x {Mc} complex; tiers by degree {a.cutoffs}: counts {counts} "
          f"(full-width params {n_ent * M2:,})", flush=True)
    # per-tier PCA basis once (covariance in fp32 on the GPU)
    bases = {}
    for t in range(T):
        idx = torch.from_numpy(np.flatnonzero(tier == t)).to(dev)
        X = E0[idx]
        mu = X.mean(0, keepdim=True)
        C = (X - mu).t() @ (X - mu) / len(idx)
        evals, evecs = torch.linalg.eigh(C)                               # ascending
        bases[t] = (idx, mu, evecs.flip(1), evals.flip(0))
        del X, C
    for cfg in a.configs:
        widths = [int(x) for x in cfg.split(",")]
        assert len(widths) == T
        E = E0.clone()
        params = 0; kept = []
        for t, w in enumerate(widths):
            idx, mu, V, ev = bases[t]
            d = min(2 * w, M2)
            P = V[:, :d]                                                  # (2M, d)
            X = E0[idx] - mu
            E[idx] = mu + (X @ P) @ P.t()
            params += len(idx) * d + M2 * d + M2
            kept.append(float(ev[:d].sum() / ev.sum()))
        with torch.no_grad():
            m.E_real.copy_(E.to(m.E_real.dtype))
        t0 = time.time()
        mrr = eval_split(m, split["valid"], ck["n_rel"], dev, label=f"valid cfg {cfg}")
        print(f"CP1 cfg ({cfg}): params {params:,} ({params/(n_ent*M2)*100:.1f}% of full)  valid MRR {mrr:.4f}  "
              f"variance kept per tier {[round(k, 3) for k in kept]}  ({time.time()-t0:.0f}s)", flush=True)
    with torch.no_grad():
        m.E_real.copy_(E0.to(m.E_real.dtype))


if __name__ == "__main__":
    main()
