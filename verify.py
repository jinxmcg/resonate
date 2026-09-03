"""Self-audit for the ogbl-biokg results (submission hygiene).

1. Split disjointness: asserts train/valid/test triple sets are
   pairwise disjoint in OUR loaded arrays (OGB constructs them so,
   but we verify what we actually trained on).
2. Checkpoint reproduction: re-evaluates every campaign checkpoint
   with the official Evaluator and compares to the logged test MRRs
   (tolerance 3e-4 for cross-device float drift).

Usage: python verify.py [--device cuda] [--models-dir checkpoints]
"""

import argparse
import glob
import os
import re

import numpy as np
import torch

from resonate import ResonatE
from train_ogb import load, globalize, eval_split

# test MRR per seed as logged by the 10-seed campaign (final_seed*.log)
LOGGED = {0: 0.8135, 1: 0.8130, 2: 0.8109, 3: 0.8110, 4: 0.8111,
          5: 0.8106, 6: 0.8100, 7: 0.8119, 8: 0.8135, 9: 0.8125}


def encode(h, r, t, n_ent, n_rel):
    return (h.astype(np.int64) * n_rel + r.astype(np.int64)) * n_ent \
        + t.astype(np.int64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--models-dir", type=str, default="checkpoints")
    p.add_argument("--skip-models", action="store_true")
    p.add_argument("--data-root", type=str, default="data_ogb")
    args = p.parse_args()

    split, offset, n_ent, types, num_nodes = load(args.data_root)
    n_rel_base = int(split["train"]["relation"].max()) + 1

    print("== 1. split disjointness ==")
    codes = {}
    for name in ("train", "valid", "test"):
        h, r, t = globalize(split[name], offset)
        codes[name] = np.unique(encode(h, r, t, n_ent, n_rel_base))
        print(f"  {name}: {len(h):,} triples "
              f"({len(codes[name]):,} unique)")
    for a, b in (("train", "valid"), ("train", "test"),
                 ("valid", "test")):
        inter = np.intersect1d(codes[a], codes[b])
        assert len(inter) == 0, \
            f"LEAKAGE: {len(inter)} triples shared by {a} and {b}"
        print(f"  {a} ∩ {b} = 0  [OK]")

    if args.skip_models:
        return

    print("\n== 2. checkpoint reproduction (test MRR vs logs) ==")
    dev = torch.device(args.device)
    got = {}
    for path in sorted(glob.glob(
            os.path.join(args.models_dir, "model_final_seed*.pt"))):
        seed = int(re.search(r"seed(\d+)", path).group(1))
        ck = torch.load(path, map_location=dev, weights_only=False)
        ca = ck["args"]
        model = ResonatE(n_entities=n_ent, n_relations=ck["n_rel"],
                        k=ca["k"], block=True,
                        block_size=ca.get("block_size", 2),
                        tied_reverse=ca.get("tied_reverse", False)
                        ).to(dev)
        model.load_state_dict(ck["model"])
        mrr = eval_split(model, split["test"], offset, ck["n_rel"],
                         dev, label=f"seed{seed}")
        got[seed] = mrr
        diff = abs(mrr - LOGGED[seed])
        status = "OK" if diff < 3e-4 else "MISMATCH"
        print(f"  seed {seed}: recomputed {mrr:.4f} vs logged "
              f"{LOGGED[seed]:.4f}  (|d|={diff:.5f}) [{status}]")
        assert diff < 3e-4, f"seed {seed} does not reproduce"

    vals = np.array([got[s] for s in sorted(got)])
    print(f"\n  ALL {len(vals)} CHECKPOINTS REPRODUCE. "
          f"test MRR {vals.mean():.4f} +/- {vals.std(ddof=1):.4f}")


if __name__ == "__main__":
    main()
