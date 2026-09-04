"""Jaccard (pure-graph) analogy member — no embeddings at all.

For a row (src, rel, dir) and candidate c: holders(c) = train sources
already linked to c under (rel, dir). Similarity between src and a
holder h' = Jaccard overlap of their training target sets under the
same (rel, dir). Feature = max / top3-mean over holders. This is
case-based reasoning on raw edge overlap: immune to every failure
mode the spectra share, including Species C hubs and Species D
bimodal rows.

Writes jaccard.{split}.npz (max) and jaccard_t3.{split}.npz.

Usage: python jaccard_member.py --split valid
"""

import argparse
import os
from collections import defaultdict

import numpy as np
from scipy import sparse

from train_ogb import load, globalize


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="valid", choices=["valid", "test"])
    p.add_argument("--out-dir", default="ens_cache")
    p.add_argument("--data-root", default="data_ogb")
    args = p.parse_args()

    split, offset, n_ent, types, num_nodes = load(args.data_root)
    part = split[args.split]
    h, r, t = globalize(part, offset)
    N = len(h)
    off_h = np.array([offset[x] for x in part["head_type"]])
    off_t = np.array([offset[x] for x in part["tail_type"]])
    neg_t = part["tail_neg"] + off_t[:, None]
    neg_h = part["head_neg"] + off_h[:, None]

    hh, rr, tt = globalize(split["train"], offset)
    n_rel = int(rr.max()) + 1

    # per (rel, dir): sparse source->target matrix + holder lists
    A = {}
    hold = {}
    for d in (0, 1):
        s = hh if d == 0 else tt
        o = tt if d == 0 else hh
        for rel_id in range(n_rel):
            m = rr == rel_id
            A[(rel_id, d)] = sparse.csr_matrix(
                (np.ones(m.sum(), np.float32), (s[m], o[m])),
                shape=(n_ent, n_ent))
    for d in (0, 1):
        s = hh if d == 0 else tt
        o = tt if d == 0 else hh
        tmp = defaultdict(list)
        for a, b, c in zip(s, rr, o):
            tmp[(b, c)].append(a)
        hold[d] = {k: np.array(v, np.int64) for k, v in tmp.items()}
    deg = {k: np.asarray(m.sum(1)).ravel() for k, m in A.items()}
    print("adjacency + holder maps built", flush=True)

    sp_mx = np.full(2 * N, -1.0, np.float16)
    sn_mx = np.full((2 * N, 500), -1.0, np.float16)
    sp_t3 = np.full(2 * N, -1.0, np.float16)
    sn_t3 = np.full((2 * N, 500), -1.0, np.float16)

    for d in (0, 1):
        src_a = h if d == 0 else t
        pos_a = t if d == 0 else h
        cand_a = neg_t if d == 0 else neg_h
        H = hold[d]
        # group rows by relation so each uses its own adjacency
        for rel_id in range(n_rel):
            rows = np.nonzero(r == rel_id)[0]
            if len(rows) == 0:
                continue
            Ar = A[(rel_id, d)].tocsr()
            dg = deg[(rel_id, d)]
            for i in rows:
                srow = Ar[src_a[i]]
                ds = srow.nnz
                if ds == 0:
                    continue
                # intersection counts of src's target set with EVERY
                # source's target set: one sparse matvec
                inter = Ar @ srow.T
                inter = np.asarray(inter.todense()).ravel()
                cands = np.concatenate([[pos_a[i]], cand_a[i]])
                out_mx = np.full(501, -1.0, np.float32)
                out_t3 = np.full(501, -1.0, np.float32)
                for ci, c in enumerate(cands):
                    hs = H.get((rel_id, c))
                    if hs is None:
                        continue
                    hs = hs[hs != src_a[i]]
                    if len(hs) == 0:
                        continue
                    iv = inter[hs]
                    jac = iv / (dg[hs] + ds - iv)
                    out_mx[ci] = jac.max()
                    out_t3[ci] = (jac if len(jac) <= 3
                                  else np.partition(jac, -3)[-3:]).mean()
                row = d * N + int(i)
                sp_mx[row], sn_mx[row] = out_mx[0], out_mx[1:]
                sp_t3[row], sn_t3[row] = out_t3[0], out_t3[1:]
            print(f"dir {d} rel {rel_id} done ({len(rows)} rows)",
                  flush=True)

    rel_out = np.concatenate([r, r])
    for tag, sp_o, sn_o in (("jaccard", sp_mx, sn_mx),
                            ("jaccard_t3", sp_t3, sn_t3)):
        out = os.path.join(args.out_dir, f"{tag}.{args.split}.npz")
        np.savez_compressed(out, sp=sp_o, sn=sn_o, rel=rel_out)
        if args.split == "test":
            print(f"{tag}: test features cached (MRR not computed) "
                  f"-> {out}", flush=True)
        else:
            mrr = (1.0 / ((sn_o >= sp_o[:, None]).sum(1) + 1)).mean()
            cov = (sp_o > -1).mean()
            print(f"{tag}: valid MRR alone {mrr:.4f} (coverage "
                  f"{cov * 100:.0f}%) -> {out}", flush=True)


if __name__ == "__main__":
    main()
