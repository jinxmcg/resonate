"""Verify the released compact (tiered) biokg checkpoints.

Rebuilds each tiered checkpoint's entity table and re-scores it with the
official OGB Evaluator, checking against the value logged when it was built.
Two uses:

  * release integrity -- the files decode and reproduce on the machine that
    made them;
  * cross-machine reproduction -- the same command on a different GPU, whose
    agreement is the "<3e-4 across machines" claim VALIDATE.md makes for the
    other released rows.

Note the tiered CONSTRUCTION is not device-portable (k-means seeds from the
device RNG; see COMPACT_B.md). This verifies the released FILES, which are
device-independent once built -- reconstruction is a matmul.

Validation split by default; --split test re-scores a committed artefact and
is the VALIDATE.md-style check, not a new test read.

Usage: PYTHONPATH=.. python verify_compact.py --ckpt-dir checkpoints \
           --data-root data_ogb --device cuda
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import torch

from resonate_tiered_biokg import TieredTableResonatE
from train_ogb import load, globalize, eval_split

TOL = 3e-4


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", default="checkpoints")
    p.add_argument("--pattern", default="tiered_T2_s{s}.pt")
    p.add_argument("--expect", default="results/cpb3/expected.json",
                   help="seed -> logged valid MRR, written by the release step")
    p.add_argument("--split", default="valid", choices=["valid", "test"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--data-root", default="data_ogb")
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    a = p.parse_args()
    dev = torch.device(a.device)

    split, offset, n_ent, _, _ = load(a.data_root)
    tr = split["train"]
    h, r, t = globalize(tr, offset)
    n_rel = 2 * (int(r.max()) + 1)
    deg = np.bincount(np.concatenate([h, t]), minlength=n_ent)
    expect = json.load(open(a.expect)) if os.path.exists(a.expect) else {}
    if torch.cuda.is_available():
        print(f"device: {torch.cuda.get_device_name(0)}  torch {torch.__version__}", flush=True)

    worst, bad = 0.0, []
    for s in a.seeds:
        path = os.path.join(a.ckpt_dir, a.pattern.format(s=s))
        if not os.path.exists(path):
            print(f"seed {s}: MISSING {path}"); bad.append(s); continue
        ck = torch.load(path, map_location="cpu", weights_only=False)
        ca = ck["args"]
        cut = np.array([int(x) for x in ca["tiers"].split(",")])
        m = TieredTableResonatE(n_ent, ck["n_rel"],
                                np.searchsorted(cut, deg, side="right"),
                                [int(x) for x in ca["widths"].split(",")],
                                k=ca.get("k", 12), block_size=ca.get("block_size", 4),
                                device=dev, subspaces=[int(x) for x in ca["subspaces"].split(",")])
        m.load_state_dict(ck["model"])
        m.eval(); m.build_eval_table()
        for q in m.parameters():
            q.requires_grad_(False)
        mrr = eval_split(m, split[a.split], offset, n_rel, dev, label=f"seed {s} {a.split}")
        ref = expect.get(str(s))
        if ref is None:
            print(f"  seed {s}: {mrr:.4f}  (no logged reference)", flush=True)
        else:
            d = abs(mrr - ref)
            worst = max(worst, d)
            ok = "OK" if d < TOL else "MISMATCH"
            print(f"  seed {s}: recomputed {mrr:.4f} vs logged {ref:.4f}  "
                  f"(|d|={d:.5f}) [{ok}]", flush=True)
            if d >= TOL:
                bad.append(s)
        assert m.n_params() == 9_555_497, f"seed {s}: {m.n_params():,} params"
        del m
        torch.cuda.empty_cache()
    print()
    if bad:
        raise SystemExit(f"FAILED for seeds {bad} (tolerance {TOL})")
    print(f"ALL {len(a.seeds)} COMPACT CHECKPOINTS REPRODUCE "
          f"(max |d| = {worst:.5f}, tolerance {TOL}); 9,555,497 parameters each")


if __name__ == "__main__":
    main()
