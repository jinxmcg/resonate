"""Brute-force check of retrieval_wiki.features on a tiny random graph (CPU)."""
import numpy as np, torch
from retrieval_wiki import build_holders, features
rng = np.random.default_rng(1)
n_ent, n_rel, n_tr = 60, 3, 400
hh = rng.integers(0, n_ent, n_tr); rr = rng.integers(0, n_rel, n_tr); tt = rng.integers(0, n_ent, n_tr)
En = torch.randn(n_ent, 8); En = (En / En.norm(dim=1, keepdim=True)).half()
C, K, cap = 7, 11, 4
for d in (0, 1):
    uniq, start, cnt, hold = (torch.from_numpy(x) for x in build_holders(hh, rr, tt, n_ent, d))
    src = torch.from_numpy(rng.integers(0, n_ent, C)); rel = torch.from_numpy(rng.integers(0, n_rel, C))
    cands = torch.from_numpy(rng.integers(0, n_ent, (C, K)))
    mx, t3, nh = features(En, src, rel, cands, uniq, start, cnt, hold, n_ent, cap)
    s_ = hh if d == 0 else tt; o_ = tt if d == 0 else hh
    sims = (En.float() @ En.float().t())
    for i in range(C):
        for j in range(K):
            holders = s_[(rr == int(rel[i])) & (o_ == int(cands[i, j]))]
            holders = holders[holders != int(src[i])]
            assert abs(nh[i, j].item() - len(holders)) < 1e-6, (nh[i, j], len(holders))
            if len(holders) == 0:
                assert mx[i, j] == -1 and t3[i, j] == -1; continue
            if len(holders) <= cap:  # exact when uncapped
                v = sims[int(src[i]), torch.from_numpy(holders)]
                assert abs(mx[i, j].item() - v.max().item()) < 2e-3, (mx[i, j], v.max())
                top = v.sort(descending=True).values[:3].mean().item()
                assert abs(t3[i, j].item() - top) < 2e-3, (t3[i, j], top)
            else:
                assert mx[i, j] > -1
print("retrieval features: brute-force match OK")
