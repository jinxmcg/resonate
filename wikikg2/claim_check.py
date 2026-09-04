"""H28 — can the model's score tell a true unseen fact from a false one?
(the measured half of the "claim checker": claims arrive as triples.)

Input: a score cache written by cache_wiki.py for the TEST split
(sp (2N,), sn (2N,500), rel (2N,): the true answer's score and the
Evaluator's 500 sampled alternatives per query, tail block then head
block). Every test triple is a fact the training graph does NOT contain,
so the exact oracle labels all of them "unverified"; only the model's
score can separate them from false claims.

Claim sets, per test query:
  TRUE          the held-out fact itself                       score sp
  FALSE-uniform one of its 500 sampled alternatives, drawn at   score sn[:, j]
                random (a corrupted claim with a random entity)
  FALSE-hard    the highest-scoring alternative (the most        score sn.max(1)
                plausible wrong answer the sampler produced)
Statistics (computed on the raw score, and on the per-query rank among
the 501 candidates, which is what a checker would report for one claim):
  AUROC of TRUE vs FALSE-uniform and TRUE vs FALSE-hard (raw score);
  at rank thresholds k in {1, 3, 10}: TPR = fraction of true claims with
  rank <= k, FPR = fraction of false claims with rank <= k, where a
  claim's rank is 1 + #(other candidates of that query scoring higher).
Overall, per direction (tail / head), and for the largest test relations.
Usage: python claim_check.py ens_cache/model_wiki_s0.test.npz [--top-rels 8]
"""
import argparse
import numpy as np


def auroc(pos, neg):
    """Rank-based AUROC (Mann-Whitney), ties count half."""
    s = np.concatenate([pos, neg]).astype(np.float64)
    r = np.empty(len(s)); order = np.argsort(s, kind="mergesort"); s_sorted = s[order]
    # average ranks for ties
    i = 0; n = len(s)
    while i < n:
        j = i
        while j + 1 < n and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    rp = r[:len(pos)]
    return (rp.sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))


def report(name, sp, sn, rng):
    n = len(sp)
    j = rng.integers(0, sn.shape[1], size=n)
    f_uni = sn[np.arange(n), j]
    f_hard = sn.max(1)
    rank_true = 1 + (sn > sp[:, None]).sum(1)
    # rank of the uniform false claim among {itself, the true answer, the other 499}
    higher_negs = (sn > f_uni[:, None]).sum(1)          # negatives scoring above it (itself excluded: equal, not >)
    rank_uni = 1 + higher_negs + (sp > f_uni)
    rank_hard = 1 + (sp > f_hard)                         # it beats every other negative by construction
    out = [f"{name:14s} n={n:>7,}  AUROC true-vs-uniform {auroc(sp, f_uni):.4f}  true-vs-hard {auroc(sp, f_hard):.4f}"]
    for k in (1, 3, 10):
        out.append(f"{'':14s} rank<={k:<2d}  TPR {np.mean(rank_true <= k):.3f}   FPR uniform {np.mean(rank_uni <= k):.3f}   FPR hard {np.mean(rank_hard <= k):.3f}")
    print("\n".join(out))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cache")
    p.add_argument("--top-rels", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    d = np.load(a.cache)
    sp, sn, rel = d["sp"].astype(np.float32), d["sn"].astype(np.float32), d["rel"]
    n2 = len(sp); n = n2 // 2
    rng = np.random.default_rng(a.seed)
    print(f"cache {a.cache}: {n:,} test queries x 2 directions, {sn.shape[1]} sampled alternatives each")
    report("all", sp, sn, rng)
    report("tail queries", sp[:n], sn[:n], rng)
    report("head queries", sp[n:], sn[n:], rng)
    rels, counts = np.unique(rel, return_counts=True)
    for r in rels[np.argsort(-counts)][:a.top_rels]:
        m = rel == r
        report(f"rel {int(r)} ({m.sum():,})", sp[m], sn[m], rng)


if __name__ == "__main__":
    main()
