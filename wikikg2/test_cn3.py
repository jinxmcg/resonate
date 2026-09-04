import numpy as np, torch
from cn_wiki import build_graph
from cn3_wiki import features
rng = np.random.default_rng(5)
n_ent, n_tr = 60, 300
hh, tt = rng.integers(0, n_ent, n_tr), rng.integers(0, n_ent, n_tr)
deg, indptr, nbr, keys = build_graph(hh, tt, n_ent)
adj = {i: set() for i in range(n_ent)}
for a, b in zip(hh, tt):
    if a != b: adj[a].add(b); adj[b].add(a)
T = lambda x: torch.from_numpy(np.asarray(x))
w = lambda n: 1/np.log(2+deg[n])
wlog = 1.0 / torch.log(2.0 + T(deg).float())
q = T(rng.integers(0, n_ent, 5)); cands = T(rng.integers(0, n_ent, (5, 7)))
v = features(q, cands, T(deg), T(indptr), T(nbr), T(keys), n_ent, 1000, wlog)
for i in range(5):
    for j in range(7):
        c, qq = int(cands[i, j]), int(q[i])
        ref = sum(w(n1)*w(n2) for n1 in adj[c] if n1 != qq for n2 in adj[n1] if n2 != c and n2 != qq and qq in adj[n2])
        assert abs(v[i, j].item() - ref) < 1e-4, (v[i, j].item(), ref)
print("cn3 feature: brute-force match OK")
