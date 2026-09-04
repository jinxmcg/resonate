import numpy as np, torch
from typed_paths import build_typed, path_types
rng = np.random.default_rng(7); n_ent, R, n_tr = 50, 4, 300
h, r, t = rng.integers(0, n_ent, n_tr), rng.integers(0, R, n_tr), rng.integers(0, n_ent, n_tr)
G = [torch.from_numpy(x) for x in build_typed(h, r, t, n_ent, R)]
# reference: directed typed adjacency
out = {}
for a, b, c in zip(h, r, t):
    out.setdefault(int(a), []).append((int(c), int(b))); out.setdefault(int(c), []).append((int(a), int(b) + R))
start = torch.from_numpy(rng.integers(0, n_ent, 20)); end = torch.from_numpy(rng.integers(0, n_ent, 20))
pr, ty = path_types(start, end, *G, n_ent, R, 1000)
got = {}
for p_, ty_ in zip(pr.tolist(), ty.tolist()): got.setdefault(p_, []).append(ty_)
for i in range(20):
    s, e = int(start[i]), int(end[i]); ref = []
    for n, o1 in out.get(s, []):
        if n in (s, e): continue
        ops2 = sorted({o2 for m, o2 in out.get(n, []) if m == e})[:2]
        for o2 in ops2: ref.append(o1 * 2 * R + o2)
    assert sorted(got.get(i, [])) == sorted(ref), (i, sorted(got.get(i, [])), sorted(ref))
print("typed path enumeration: brute-force match OK")
