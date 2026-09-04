"""Cache per-model OGB candidate scores for ensemble weight search.

For each checkpoint, scores the valid (default) split both directions
and saves float16 arrays to ens_cache/<model>.<split>.npz:
  sp (2N,), sn (2N,500), rel (2N,)  — tail block then head block,
  rel = base relation id for both directions.
Valid-only by default: the single committed test shot is scored
separately, once, after the ensemble config is frozen.

Usage: python cache_scores.py --device cuda --split valid --models ...
"""

import argparse
import os

import numpy as np
import torch

from resonate import ResonatE
from train_ogb import load, globalize


@torch.no_grad()
def score(model, part, offset, n_rel, dev, chunk=512):
    h, r, t = globalize(part, offset)
    off_h = np.array([offset[x] for x in part["head_type"]])
    off_t = np.array([offset[x] for x in part["tail_type"]])
    neg_h = part["head_neg"] + off_h[:, None]
    neg_t = part["tail_neg"] + off_t[:, None]
    sps, sns = [], []
    for dir_ in ("tail", "head"):
        for i in range(0, len(h), chunk):
            sl = slice(i, i + chunk)
            if dir_ == "tail":
                src, rel = h[sl], r[sl]
                pos, cand = t[sl], neg_t[sl]
            else:
                src, rel = t[sl], r[sl] + n_rel // 2
                pos, cand = h[sl], neg_h[sl]
            rel_t = torch.from_numpy(rel).to(dev)
            z = model.out(model.hop(model.embed(
                torch.from_numpy(src).to(dev)), rel_t), rel_t)
            tau = model.log_tau.exp()
            sp = torch.real((z * model.E[torch.from_numpy(pos).to(dev)]
                             .conj()).sum(-1)) * tau
            sn = torch.real(torch.einsum(
                "bm,bcm->bc", z,
                model.E[torch.from_numpy(cand).to(dev)].conj())) * tau
            if getattr(model, "b", None) is not None:
                sp = sp + model.b[torch.from_numpy(pos).to(dev)]
                sn = sn + model.b[torch.from_numpy(cand).to(dev)]
            sps.append(sp.cpu())
            sns.append(sn.cpu())
    return (torch.cat(sps).numpy().astype(np.float16),
            torch.cat(sns).numpy().astype(np.float16),
            np.concatenate([r, r]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--split", default="valid", choices=["valid", "test"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="ens_cache")
    p.add_argument("--data-root", default="data_ogb")
    args = p.parse_args()
    dev = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)

    split, offset, n_ent, types, num_nodes = load(args.data_root)
    part = split[args.split]

    for path in args.models:
        tag = os.path.splitext(os.path.basename(path))[0]
        out = os.path.join(args.out, f"{tag}.{args.split}.npz")
        if os.path.exists(out):
            print(f"skip {tag} (cached)", flush=True)
            continue
        ck = torch.load(path, map_location=dev, weights_only=False)
        ca = ck["args"]
        model = ResonatE(n_entities=n_ent, n_relations=ck["n_rel"],
                         k=ca["k"], block=True,
                         block_size=ca.get("block_size", 2),
                         tied_reverse=ca.get("tied_reverse", False),
                         ent_bias=ca.get("ent_bias", False),
                         rel_gain=ca.get("rel_gain", False)).to(dev)
        model.load_state_dict(ck["model"])
        sp, sn, rel = score(model, part, offset, ck["n_rel"], dev)
        np.savez_compressed(out, sp=sp, sn=sn, rel=rel)
        if args.split == "test":
            # committed-shot discipline: no per-member test numbers
            print(f"{tag}: test scores cached (MRR not computed) -> {out}",
                  flush=True)
        else:
            mrr = (1.0 / ((sn >= sp[:, None]).sum(1) + 1)).mean()
            print(f"{tag}: {args.split} MRR(f16) {mrr:.4f} -> {out}",
                  flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
