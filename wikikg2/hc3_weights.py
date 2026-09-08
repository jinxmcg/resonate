"""HC3: importance weights for the holdout fit rows (HOLDOUT_COMBINER.md, HC3).

w(row) = P_valid(cell) / P_holdout(cell), cell = (log2 bucket of head degree,
log2 bucket of tail degree), validation degrees in full TRAIN, holdout degrees
in fit-TRAIN. Cells absent from the holdout are dropped and their validation
mass reported; weights clipped to [0, CLIP] and scaled to mean 1.

Usage (WIKI_HOLDOUT unset): python hc3_weights.py --holdout holdout_wiki_row.npz --out hc3_weights.npy
"""
import argparse, json, os
import numpy as np
from train_wiki import load

CAP, CLIP = 20, 50.0


def bucket(d):
    return np.minimum(np.floor(np.log2(d + 1)).astype(int), CAP)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--holdout", default="holdout_wiki_row.npz")
    p.add_argument("--data-root", default="data_ogb")
    p.add_argument("--out", default="hc3_weights.npy")
    a = p.parse_args()
    assert not os.environ.get("WIKI_HOLDOUT")
    split, n = load(a.data_root)
    tr = split["train"]
    h, t = (np.asarray(tr[k]).astype(np.int64) for k in ("head", "tail"))
    deg = np.bincount(np.concatenate([h, t]), minlength=n)
    va = split["valid"]
    vh, vt = (np.asarray(va[k]).astype(np.int64) for k in ("head", "tail"))
    z = np.load(a.holdout)
    fit, ho = z["fit_rows"], z["holdout_rows"]
    degf = np.bincount(np.concatenate([h[fit], t[fit]]), minlength=n)
    K = CAP + 1
    cv = bucket(deg[vh]) * K + bucket(deg[vt])
    ch = bucket(degf[h[ho]]) * K + bucket(degf[t[ho]])
    Pv = np.bincount(cv, minlength=K * K) / len(cv)
    Ph = np.bincount(ch, minlength=K * K) / len(ch)
    lost = float(Pv[Ph == 0].sum())
    ratio = np.where(Ph > 0, Pv / np.maximum(Ph, 1e-12), 0.0)
    w = np.clip(ratio[ch], 0, CLIP)
    clipped = float((ratio[ch] > CLIP).mean())
    w = w / w.mean()
    ess = float(w.sum() ** 2 / (w * w).sum())
    np.save(a.out, w.astype(np.float32))
    # the matched histograms, for the record
    wv = np.bincount(ch, weights=w, minlength=K * K) / w.sum()
    summary = dict(holdout=a.holdout, rows=int(len(ho)), validation_rows=int(len(vh)), cap=CAP, clip=CLIP,
                   validation_mass_in_cells_absent_from_holdout=lost, fraction_clipped=clipped,
                   effective_sample_size=ess, max_weight=float(w.max()), zero_weight_rows=int((w == 0).sum()),
                   l1_distance_valid_vs_holdout=float(np.abs(Pv - Ph).sum()),
                   l1_distance_valid_vs_weighted_holdout=float(np.abs(Pv - wv).sum()))
    json.dump(summary, open(os.path.splitext(a.out)[0] + ".json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
