"""Build the 'analogy' structural member for the ensemble cache.

For every valid row and every candidate c (positive + 500 negatives):
  feature(c) = max over holders(c) of cos(e_src, e_holder)^3
where holders(c) = training-set sources already connected to c under
the row's (relation, direction), excluding src itself. Popularity-free
(MAX not SUM — the drug-sideeffect lesson), label-free, and computable
at query time from the oracle graph.

Embeddings: embedding cosine from --model (unit-norm E), averaged over
--models if several are given. Output ens_cache/analogy.valid.npz in
the standard sp/sn/rel layout so ensemble_weights.py picks it up as an
18th member. Rows with no holders get -1.

Usage: python analogy_member.py --device cuda --models model_final_seed0.pt \
           --split valid --out ens_cache/analogy_s0.valid.npz \
           --out-top3 ens_cache/analogy_s0_t3.valid.npz
"""

import argparse
import os
from collections import defaultdict

import numpy as np
import torch

from resonate import ResonatE
from train_ogb import load, globalize


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True,
                   help="checkpoint(s) whose spectra define the cosine")
    p.add_argument("--device", default="cuda")
    p.add_argument("--split", default="valid", choices=["valid", "test"])
    p.add_argument("--out", required=True,
                   help="e.g. ens_cache/analogy_s0.valid.npz")
    p.add_argument("--out-top3", default=None,
                   help="also write a mean-of-top-3-holder-sims variant")
    p.add_argument("--chunk", type=int, default=256)
    p.add_argument("--data-root", default="data_ogb")
    args = p.parse_args()
    dev = torch.device(args.device)

    split, offset, n_ent, types, num_nodes = load(args.data_root)
    part = split[args.split]
    h, r, t = globalize(part, offset)
    N = len(h)
    off_h = np.array([offset[x] for x in part["head_type"]])
    off_t = np.array([offset[x] for x in part["tail_type"]])
    neg_t = part["tail_neg"] + off_t[:, None]
    neg_h = part["head_neg"] + off_h[:, None]

    # holders per (dir, rel): target entity -> np.array of train sources
    hh, rr, tt = globalize(split["train"], offset)
    hold = {}
    for d in (0, 1):
        tmp = defaultdict(list)
        if d == 0:
            for a, b, c in zip(hh, rr, tt):
                tmp[(b, c)].append(a)
        else:
            for a, b, c in zip(hh, rr, tt):
                tmp[(b, a)].append(c)
        hold[d] = {k: np.array(v, np.int64) for k, v in tmp.items()}
    print(f"holder maps built ({len(hold[0]):,} + {len(hold[1]):,} keys)",
          flush=True)

    # mean spectral-similarity matrix source-row x all entities, chunked
    Es = []
    for path in args.models:
        ck = torch.load(path, map_location=dev, weights_only=False)
        ca = ck["args"]
        m = ResonatE(n_entities=n_ent, n_relations=ck["n_rel"], k=ca["k"],
                     block=True, block_size=ca.get("block_size", 2),
                     tied_reverse=ca.get("tied_reverse", False),
                     ent_bias=ca.get("ent_bias", False),
                     rel_gain=ca.get("rel_gain", False)).to(dev)
        m.load_state_dict(ck["model"])
        with torch.no_grad():
            E = m.E / m.E.abs().pow(2).sum(-1, keepdim=True).sqrt()
        Es.append(E)
        del m
    print(f"embeddings: {len(Es)} model(s)", flush=True)

    sp_out = np.full(2 * N, -1.0, np.float16)
    sn_out = np.full((2 * N, 500), -1.0, np.float16)
    sp3 = np.full(2 * N, -1.0, np.float16)
    sn3 = np.full((2 * N, 500), -1.0, np.float16)

    for d in (0, 1):
        src_a = h if d == 0 else t
        pos_a = t if d == 0 else h
        cand_a = neg_t if d == 0 else neg_h
        H = hold[d]
        for i in range(0, N, args.chunk):
            sl = slice(i, min(i + args.chunk, N))
            src = src_a[sl]
            with torch.no_grad():
                sims = None
                for E in Es:
                    e = E[torch.from_numpy(src).to(dev)]
                    s = torch.real(e @ E.conj().T)
                    sims = s if sims is None else sims + s
                sims = (sims / len(Es)).cpu().numpy()
            for j in range(len(src)):
                row = d * N + i + j
                sv = sims[j]
                cands = np.concatenate([[pos_a[sl][j]], cand_a[sl][j]])
                out = np.full(501, -1.0, np.float32)
                out3 = np.full(501, -1.0, np.float32)
                for ci, c in enumerate(cands):
                    hs = H.get((r[sl][j], c))
                    if hs is None:
                        continue
                    x = sv[hs]
                    if src[j] in hs:
                        x = x[hs != src[j]]
                        if len(x) == 0:
                            continue
                    out[ci] = x.max()
                    out3[ci] = (x if len(x) <= 3
                                else np.partition(x, -3)[-3:]).mean()
                out = np.sign(out) * np.abs(out) ** 3
                out3 = np.sign(out3) * np.abs(out3) ** 3
                sp_out[row] = out[0]
                sn_out[row] = out[1:]
                sp3[row] = out3[0]
                sn3[row] = out3[1:]
            if (i // args.chunk) % 40 == 0:
                print(f"dir {d} row {i}/{N}", flush=True)

    rel_out = np.concatenate([r, r])
    np.savez_compressed(args.out, sp=sp_out, sn=sn_out, rel=rel_out)
    if args.out_top3:
        np.savez_compressed(args.out_top3, sp=sp3, sn=sn3, rel=rel_out)
        if args.split != "test":
            m3 = (1.0 / ((sn3 >= sp3[:, None]).sum(1) + 1)).mean()
            print(f"top3-mean variant alone: MRR {m3:.4f} -> "
                  f"{args.out_top3}", flush=True)
    cov = (sp_out > -1).mean()
    if args.split == "test":
        # committed-shot discipline: cache only, no test numbers
        print(f"test features cached (MRR not computed; holders on "
              f"{cov * 100:.0f}% of rows) -> {args.out}", flush=True)
    else:
        mrr = (1.0 / ((sn_out >= sp_out[:, None]).sum(1) + 1)).mean()
        print(f"analogy member alone: {args.split} MRR {mrr:.4f} "
              f"(positive has holders on {cov * 100:.0f}% of rows) -> "
              f"{args.out}", flush=True)


if __name__ == "__main__":
    main()
