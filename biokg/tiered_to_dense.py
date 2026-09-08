"""Expand a CP-B1 tiered checkpoint (resonate_tiered_biokg.TieredTableResonatE:
narrow per-entity coefficients + per-tier subspace banks) into the dense
resonate.ResonatE checkpoint format the public biokg scripts load
(E complex (N, M)). The tiered table is compact in STORAGE; at scoring time it
reconstructs to an ordinary entity table, so this is an exact reinterpretation
in the same sense as sparse_to_dense.py, and cache_scores.py / analogy_member.py
/ ensemble_weights.py / freeze_test.py / verify.py then run unchanged.

The stored parameter count (what CP-B1 counts) is carried into args as
"tiered_params" so the receipt cannot be mistaken for a dense 27.1M model.

Usage: python tiered_to_dense.py checkpoints/cpb1_s0_1.pt checkpoints/cpb1_w48_s0.pt
"""
import sys

import numpy as np
import torch

from resonate_tiered_biokg import TieredTableResonatE
from train_ogb import load, globalize

src, dst = sys.argv[1], sys.argv[2]
data_root = sys.argv[3] if len(sys.argv) > 3 else "data_ogb"
d = torch.load(src, map_location="cpu", weights_only=False)
ca = d["args"]
split, offset, n_ent, _, _ = load(data_root)
tr = split["train"]
h, r, t = globalize(tr, offset)
deg = np.bincount(np.concatenate([h, t]), minlength=n_ent)
cutoffs = np.array([int(x) for x in ca["tiers"].split(",")])
widths = [int(x) for x in ca["widths"].split(",")]
K = [int(x) for x in ca["subspaces"].split(",")]
m = TieredTableResonatE(n_ent, d["n_rel"], np.searchsorted(cutoffs, deg, side="right"),
                        widths, k=ca.get("k", 12), block_size=ca.get("block_size", 4),
                        device=None, rel_gain=ca.get("rel_gain", False), subspaces=K)
m.load_state_dict(d["model"])
m.eval()
sd = {"E": m.table().detach().contiguous(), "H": m.H.detach(),
      "log_tau": m.log_tau.detach()}
if getattr(m, "b", None) is not None:
    sd["b"] = m.b.detach()
args = dict(ca)
args["shell"] = "dense"
args["converted_from"] = "tiered"
args["tiered_params"] = m.n_params()
args["tiered_source"] = src
torch.save({"model": sd, "offset": d["offset"], "n_rel": d["n_rel"], "args": args}, dst)
print(dst, {k_: tuple(v.shape) for k_, v in sd.items()},
      f"stored tiered params {m.n_params():,}")
