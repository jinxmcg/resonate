"""ES1: score-average ensembles without per-model caches (disk). For each
model in turn: the model's own OGB scores (score_split) and the opposite-
direction members (reverse_wiki.reverse_scores); the running means are
written once as ens_cache/<tag>.<split>.npz, rev_raw_<tag>, rev_nov_<tag>.

Usage: python ens_cache.py --tag ens7 --split valid --models teachers/model_dist_s1.bf16.pt ...
"""
import argparse
import os
import time

import numpy as np
import torch

from train_wiki import load, load_model, score_split
from reverse_wiki import reverse_scores


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--split", default="valid", choices=["valid", "test", "holdout"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="ens_cache")
    p.add_argument("--data-root", default="data_ogb")
    p.add_argument("--no-reverse", action="store_true")
    a = p.parse_args()
    dev = torch.device(a.device)
    split, n_ent = load(a.data_root)
    part = split[a.split]
    N = len(part["head"]); rel = np.concatenate([np.asarray(part["relation"])] * 2)
    acc = {k: np.zeros((2 * N, 501), np.float32) for k in (["model"] + ([] if a.no_reverse else ["rev_raw", "rev_nov"]))}
    t0 = time.time()
    for i, path in enumerate(a.models):
        m, ck = load_model(path, n_ent, dev)
        sp, sn, _ = score_split(m, part, ck["n_rel"], dev)
        acc["model"] += np.concatenate([sp.numpy()[:, None], sn.numpy()], 1).astype(np.float32)
        if not a.no_reverse:
            raw, nov = reverse_scores(m, ck, part, dev)
            acc["rev_raw"] += raw; acc["rev_nov"] += nov
        del m; torch.cuda.empty_cache()
        print(f"[{i+1}/{len(a.models)}] {os.path.basename(path)} accumulated ({time.time()-t0:.0f}s)", flush=True)
    n = len(a.models)
    for k, v in acc.items():
        v /= n
        name = a.tag if k == "model" else f"{k}_{a.tag}"
        out = os.path.join(a.out, f"{name}.{a.split}.npz")
        np.savez(out, sp=v[:, 0].astype(np.float16), sn=v[:, 1:].astype(np.float16), rel=rel)
        if a.split != "test":
            mrr = 1.0 / ((v[:, 1:] >= v[:, :1]).sum(1) + 1)
            print(f"{name}: {a.split} MRR alone {mrr.mean():.4f} (tail {mrr[:N].mean():.4f} head {mrr[N:].mean():.4f}) -> {out}", flush=True)
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
