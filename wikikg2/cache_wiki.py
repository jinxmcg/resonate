"""Cache per-model OGB candidate scores (ogbl-wikikg2) for the blend.

For each checkpoint, scores a split in both directions and writes
ens_cache/<tag>.<split>.npz with sp (2N,), sn (2N,500), rel (2N,) in
float16 — tail block then head block, rel = base relation id. The
same layout the biokg cache_scores.py / ensemble_weights.py use.

Valid by default; the test caches are written once, after the blend
is frozen on validation (committed-shot discipline: no per-member
test MRR is printed).

Usage: python cache_wiki.py --device cuda --split valid --models a.pt b.pt
"""

import argparse
import os

import numpy as np
import torch

from train_wiki import load, load_model, score_split


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
    split, n_ent = load(args.data_root)
    part = split[args.split]
    for path in args.models:
        tag = os.path.splitext(os.path.basename(path))[0]
        out = os.path.join(args.out, f"{tag}.{args.split}.npz")
        if os.path.exists(out):
            print(f"skip {tag} (cached)", flush=True)
            continue
        model, ck = load_model(path, n_ent, dev)
        sp, sn, rel = score_split(model, part, ck["n_rel"], dev)
        sp = sp.numpy().astype(np.float16)
        sn = sn.numpy().astype(np.float16)
        np.savez(out, sp=sp, sn=sn, rel=rel.numpy())
        if args.split == "test":
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
