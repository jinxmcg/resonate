"""RV1: opposite-direction operator members (REVERSE_MEMBER.md).

For each question (known entity q, operator op, candidates x_0..x_500 with
x_0 the answer) score every candidate with the OPPOSITE operator:
z_x = out(hop(embed(x), op'), op'), S[x, y] = Re<z_x, row(y)> * tau over
targets y in {q} u {x_1..x_500}. rev_raw = S[x, q]; rev_nov = S[x, q] -
logsumexp_y S[x, y]. Writes ens_cache/rev_raw_<tag>.<split>.npz and
rev_nov_<tag>.<split>.npz (tail block then head block, like cache_wiki.py).

Usage: python reverse_wiki.py --device cuda --model teachers/model_dist_s1.bf16.pt --tag d1 --split valid
"""
import argparse
import os
import time

import numpy as np
import torch

from train_wiki import load, load_model


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--split", default="valid", choices=["valid", "test", "holdout"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--chunk", type=int, default=96)
    p.add_argument("--out-dir", default="ens_cache")
    p.add_argument("--data-root", default="data_ogb")
    p.add_argument("--table-dtype", default=None, choices=["fp32", "fp16", "bf16"])
    p.add_argument("--limit", type=int, default=0, help="debug: only the first N questions")
    a = p.parse_args()
    dev = torch.device(a.device)
    os.makedirs(a.out_dir, exist_ok=True)
    split, n_ent = load(a.data_root)
    part = split[a.split]
    h = np.asarray(part["head"]).astype(np.int64); r = np.asarray(part["relation"]).astype(np.int64)
    t = np.asarray(part["tail"]).astype(np.int64)
    neg_h = np.asarray(part["head_neg"]).astype(np.int64); neg_t = np.asarray(part["tail_neg"]).astype(np.int64)
    if a.limit:
        h, r, t, neg_h, neg_t = h[:a.limit], r[:a.limit], t[:a.limit], neg_h[:a.limit], neg_t[:a.limit]
    N = len(h)
    m, ck = load_model(a.model, n_ent, dev, table_dtype=a.table_dtype)
    R = ck["n_rel"] // 2
    tau = m.log_tau.exp()
    raw = np.zeros((2 * N, 501), np.float32); nov = np.zeros((2 * N, 501), np.float32)
    t0 = time.time()
    # d=0: tail question (h, r, ?): candidates are tails, known q = h, opposite operator = reverse r + R
    # d=1: head question (?, r, t): candidates are heads, known q = t, opposite operator = forward r
    for d in (0, 1):
        q_a = h if d == 0 else t
        pos_a = t if d == 0 else h
        cand_a = neg_t if d == 0 else neg_h
        op_a = r + R if d == 0 else r
        for i in range(0, N, a.chunk):
            sl = slice(i, min(i + a.chunk, N)); B = sl.stop - sl.start
            cands = torch.from_numpy(np.concatenate([pos_a[sl][:, None], cand_a[sl]], 1)).to(dev)   # (B, 501)
            q = torch.from_numpy(q_a[sl]).to(dev)
            op = torch.from_numpy(op_a[sl]).to(dev).repeat_interleave(501)
            z = m.out(m.hop(m.embed(cands.reshape(-1)), op), op).reshape(B, 501, -1)             # (B, 501, M)
            targets = torch.cat([q[:, None], cands[:, 1:]], 1)                                    # (B, 501): q then decoys
            rows = m.rows(targets)                                                                # (B, 501, M)
            S = torch.real(torch.einsum("bxm,bym->bxy", z, rows.conj())) * tau                    # (B, 501, 501)
            sq = S[:, :, 0]
            block = slice(d * N + sl.start, d * N + sl.stop)
            raw[block] = sq.float().cpu().numpy()
            nov[block] = (sq - torch.logsumexp(S, dim=2)).float().cpu().numpy()
            if (i // a.chunk) % 500 == 0:
                print(f"dir {d} row {i}/{N} ({time.time()-t0:.0f}s)", flush=True)
    rel_out = np.concatenate([r, r])
    for name, v in (("rev_raw", raw), ("rev_nov", nov)):
        out = os.path.join(a.out_dir, f"{name}_{a.tag}.{a.split}.npz")
        np.savez(out, sp=v[:, 0].astype(np.float16), sn=v[:, 1:].astype(np.float16), rel=rel_out)
        if a.split != "test":
            mrr = (1.0 / ((v[:, 1:] >= v[:, :1]).sum(1) + 1))
            print(f"{name}_{a.tag}: {a.split} MRR alone {mrr.mean():.4f} (tail {mrr[:N].mean():.4f} head {mrr[N:].mean():.4f}) -> {out}", flush=True)
        else:
            print(f"{name}_{a.tag}: test cached (MRR not computed) -> {out}", flush=True)
    # self-check: head-block rev_raw at the true head == the model's own tail-direction score of the triple
    if a.split != "test":
        from train_wiki import score_split
        k = min(N, 512)
        sub = {kk: np.asarray(part[kk])[:k] for kk in ("head", "relation", "tail", "head_neg", "tail_neg")}
        sp, _, _ = score_split(m, sub, ck["n_rel"], dev)
        diff = np.abs(sp[:k].numpy() - raw[N:N + k, 0]).max()
        print(f"self-check: |tail-direction score - rev_raw(head block, true head)| max {diff:.2e} on {k} questions", flush=True)
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
