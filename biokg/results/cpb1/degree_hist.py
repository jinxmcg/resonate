import numpy as np, sys
sys.path.insert(0, "/mnt/geocore/resonate/biokg")
sys.path.insert(0, "/mnt/geocore/resonate")
from train_ogb import load, globalize

split, offset, n_ent, types, num_nodes = load("/mnt/geocore/geocore/data_ogb")
tr = split["train"]; va = split["valid"]
h, r, t = globalize(tr, offset)
deg = np.bincount(np.concatenate([h, t]), minlength=n_ent)
vh, vr, vt = globalize(va, offset)
print(f"entities {n_ent:,}  train {len(h):,}  valid {len(vh):,}")
print("types:", {k: int(v) for k, v in num_nodes.items()})
print("degree: min %d  p1 %d  p10 %d  med %d  p90 %d  p99 %d  max %d  mean %.1f" % (
    deg.min(), *np.percentile(deg, [1, 10, 50, 90, 99]).astype(int), deg.max(), deg.mean()))
print("zero-degree entities:", int((deg == 0).sum()))
# validation answers (both directions) by degree of the ANSWER entity
ans = np.concatenate([vt, vh])
edges = [0, 1, 3, 5, 8, 16, 32, 64, 128, 256, 512, 1024, 4096, 10**9]
print("\n%-14s %10s %8s %10s %8s" % ("degree", "entities", "share", "val-ans", "share"))
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (deg >= lo) & (deg < hi)
    n = int(m.sum())
    if n == 0: continue
    a = int(m[ans].sum())
    print("%-14s %10d %7.1f%% %10d %7.1f%%" % (f"{lo}-{hi-1 if hi<10**9 else ''}", n, 100*n/n_ent, a, 100*a/len(ans)))
# per-type degree medians
off_sorted = sorted(offset.items(), key=lambda kv: kv[1])
print()
for i, (ty, o) in enumerate(off_sorted):
    end = off_sorted[i+1][1] if i+1 < len(off_sorted) else n_ent
    d = deg[o:end]
    print("%-12s n=%6d  med %5d  p10 %4d  p90 %6d  max %7d" % (ty, end-o, int(np.median(d)), int(np.percentile(d,10)), int(np.percentile(d,90)), d.max()))
np.save("/tmp/claude-1000/-mnt-geocore/b8457cab-1b15-48cd-8427-879f705091ee/scratchpad/deg.npy", deg)
