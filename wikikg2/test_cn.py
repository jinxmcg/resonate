import numpy as np, torch
from cn_wiki import build_graph, features
rng = np.random.default_rng(3)
n_ent, n_tr = 80, 500
hh, tt = rng.integers(0, n_ent, n_tr), rng.integers(0, n_ent, n_tr)
deg, indptr, nbr, keys = build_graph(hh, tt, n_ent)
adj = {i: set() for i in range(n_ent)}
for a, b in zip(hh, tt):
    if a != b: adj[a].add(b); adj[b].add(a)
assert all(len(adj[i]) == deg[i] for i in range(n_ent))
T = lambda x: torch.from_numpy(np.asarray(x))
wlog = 1.0 / torch.log(2.0 + T(deg).float())
q = T(rng.integers(0, n_ent, 6)); cands = T(rng.integers(0, n_ent, (6, 9)))
cn, aa, linked = features(q, cands, T(deg), T(indptr), T(nbr), T(keys), n_ent, 1000, wlog)
for i in range(6):
    for j in range(9):
        c, qq = int(cands[i, j]), int(q[i])
        common = (adj[c] & adj[qq]) - {c, qq}
        assert abs(cn[i, j].item() - np.log1p(len(common))) < 1e-5
        assert abs(aa[i, j].item() - sum(1/np.log(2+deg[n]) for n in common)) < 1e-4
        assert linked[i, j].item() == float(qq in adj[c])
print("cn features: brute-force match OK")
