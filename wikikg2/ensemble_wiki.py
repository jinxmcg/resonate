"""Score-average ensemble of N checkpoints as ONE member cache (row E).

Loads each checkpoint in turn, scores the split (both directions, 501
candidates), accumulates the mean of the raw logits, and writes
ens_cache/<tag>.<split>.npz in the standard layout. The ensemble is
then blended with the retrieval members like any single model; test
is read once by blend_wiki freeze --test.

Usage: python ensemble_wiki.py --device cuda --split valid --tag ens10 \
           --models models/box2/model_wiki_s0.pt ...
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
    p.add_argument("--tag", default="ens10")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="ens_cache")
    p.add_argument("--data-root", default="data_ogb")
    args = p.parse_args()
    dev = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    split, n_ent = load(args.data_root)
    part = split[args.split]
    sp_sum = sn_sum = None
    rel = None
    for i, path in enumerate(args.models):
        model, ck = load_model(path, n_ent, dev)
        sp, sn, rel = score_split(model, part, ck["n_rel"], dev)
        sp, sn = sp.numpy().astype(np.float32), sn.numpy().astype(np.float32)
        if sp_sum is None:
            sp_sum, sn_sum = sp, sn
        else:
            sp_sum += sp; sn_sum += sn
        if args.split == "valid":
            m1 = (1.0 / ((sn >= sp[:, None]).sum(1) + 1)).mean()
            me = (1.0 / ((sn_sum >= sp_sum[:, None]).sum(1) + 1)).mean()
            print(f"[{i+1}/{len(args.models)}] {os.path.basename(path)}: alone {m1:.4f}  "
                  f"ensemble so far {me:.4f}", flush=True)
        else:
            print(f"[{i+1}/{len(args.models)}] {os.path.basename(path)} scored (test: no MRR printed)",
                  flush=True)
        del model
        torch.cuda.empty_cache()
    n = len(args.models)
    out = os.path.join(args.out, f"{args.tag}.{args.split}.npz")
    np.savez(out, sp=(sp_sum / n).astype(np.float16), sn=(sn_sum / n).astype(np.float16),
             rel=rel.numpy())
    print(f"-> {out}")


if __name__ == "__main__":
    main()
