"""Self-augmentation: fill the training graph with the model's own
confident proposals for well-determined relations, so that the path
members have evidence for entities whose citizenship / occupation /
birthplace / country is missing.

For each relation r in --relations, the candidate set is the distinct
training tails of r (a few hundred countries, a few thousand
occupations, ...). Every entity that is a plausible head for r (has the
same "instance of" class as r's training heads, i.e. human for the
person relations) and has NO r-edge in training gets the model's top-1
tail if the softmax over the candidate set is >= --p. Only training
edges and the model are used: label-free, test-agnostic.

Writes extra_edges.npz (h, r, t) for cn_wiki / cn3_wiki / retrieval_wiki
--extra-edges. Reports per-relation counts, and the precision of the
same rule on VALIDATION rows of r (a held-out check of the proposals).

Usage: python augment_graph.py --model model_wiki_s3.pt --relations P27 P106 P19 P17 --p 0.5
"""

import argparse
import csv
import gzip

import numpy as np
import torch

from train_wiki import load, load_model


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--relations", nargs="+", default=["all"],
                   help="P-ids, or 'all': every relation with >= --min-valid "
                        "validation rows whose rule reaches --min-prec there")
    p.add_argument("--p", type=float, default=0.5, help="min softmax prob over the candidate set")
    p.add_argument("--min-prec", type=float, default=0.7,
                   help="'all' mode: keep a relation only if its top-1 proposals "
                        "at p >= --p are >= this precise on validation rows")
    p.add_argument("--min-valid", type=int, default=50)
    p.add_argument("--max-cands", type=int, default=200000,
                   help="skip relations with more distinct tails than this")
    p.add_argument("--out", default="extra_edges.npz")
    p.add_argument("--device", default="cuda")
    p.add_argument("--chunk", type=int, default=4096)
    args = p.parse_args()
    dev = torch.device(args.device)
    split, n_ent = load("data_ogb")
    with gzip.open("data_ogb/ogbl_wikikg2/mapping/reltype2relid.csv.gz", "rt") as f:
        rd = csv.reader(f); next(rd); pid = {int(i): pp for i, pp in rd}
    rid = {v: k for k, v in pid.items()}
    tr = split["train"]
    h, r, t = (np.asarray(tr[k]).astype(np.int64) for k in ("head", "relation", "tail"))
    model, ck = load_model(args.model, n_ent, dev)
    R = ck["n_rel"] // 2
    p31 = rid["P31"]
    cls = {}  # entity -> one 'instance of' class (first seen)
    m31 = r == p31
    for hh, tt in zip(h[m31], t[m31]):
        cls.setdefault(int(hh), int(tt))
    cls_arr = np.full(n_ent, -1, np.int64)
    cls_arr[list(cls.keys())] = list(cls.values())
    va = split["valid"]
    vh, vr, vt = (np.asarray(va[k]).astype(np.int64) for k in ("head", "relation", "tail"))
    out_h, out_r, out_t = [], [], []
    if args.relations == ["all"]:
        vc = np.bincount(vr, minlength=R)
        rel_list = [pid[int(x)] for x in np.argsort(-vc) if vc[x] >= args.min_valid]
        print(f"'all': {len(rel_list)} relations with >= {args.min_valid} validation rows", flush=True)
    else:
        rel_list = args.relations
    summary = []
    for prop in rel_list:
        rr = rid[prop]
        m = r == rr
        cands = np.unique(t[m])
        if len(cands) > args.max_cands or len(cands) < 2:
            continue
        # head class: the modal 'instance of' class among r's training heads
        hc = cls_arr[h[m]]
        hc = hc[hc >= 0]
        main_cls = np.bincount(hc).argmax() if len(hc) else -1
        has = np.zeros(n_ent, bool); has[h[m]] = True
        heads = np.nonzero((cls_arr == main_cls) & ~has)[0]
        cand_t = torch.from_numpy(cands).to(dev)
        E_c = model.rows(cand_t)                       # (C, M)
        bias = model.b[cand_t] if model.b is not None else None
        tau = model.log_tau.exp()
        # held-out precision of the same rule on validation rows of r
        vrows = np.nonzero(vr == rr)[0]
        def predict(src_np):
            best_i, best_p = [], []
            for i in range(0, len(src_np), args.chunk):
                src = torch.from_numpy(src_np[i:i + args.chunk]).to(dev)
                z = model.out(model.hop(model.embed(src), torch.full((len(src),), rr, device=dev)),
                              torch.full((len(src),), rr, device=dev))
                s = torch.real(z @ E_c.conj().t()) * tau
                if bias is not None:
                    s = s + bias[None, :]
                pr = torch.softmax(s, 1)
                pv, pi = pr.max(1)
                best_i.append(cands[pi.cpu().numpy()]); best_p.append(pv.cpu().numpy())
            return np.concatenate(best_i), np.concatenate(best_p)
        # calibrated threshold: sort validation rows by confidence, take the
        # lowest confidence at which cumulative precision is still >= min_prec
        prec, fire, thr = float("nan"), 0.0, None
        if len(vrows) >= 20:
            pi, pv = predict(vh[vrows])
            ok = (pi == vt[vrows]).astype(np.float64)
            o = np.argsort(-pv)
            cum = np.cumsum(ok[o]) / (np.arange(len(o)) + 1)
            good = np.nonzero((cum >= args.min_prec) & (np.arange(len(o)) >= 19))[0]
            if len(good):
                k = good.max()
                thr = max(float(pv[o][k]), args.p)
                sel = pv >= thr
                fire, prec = sel.mean(), ok[sel].mean()
        if thr is None:
            summary.append((prop, len(vrows), fire, prec, len(cands), 0, 0))
            continue
        if len(heads) == 0:
            continue
        pi, pv = predict(heads)
        keep = pv >= thr
        summary.append((prop, len(vrows), fire, prec, len(cands), len(heads), int(keep.sum())))
        print(f"{prop}: valid rows {len(vrows):,}, threshold {thr:.3f} fires {fire*100:.0f}%, precision {prec:.3f} | "
              f"{len(cands):,} candidates, {len(heads):,} class-Q{main_cls} entities lack it -> "
              f"proposing {keep.sum():,}", flush=True)
        out_h.append(heads[keep]); out_r.append(np.full(keep.sum(), rr)); out_t.append(pi[keep])
    if out_h:
        np.savez(args.out, h=np.concatenate(out_h), r=np.concatenate(out_r), t=np.concatenate(out_t))
    print(f"-> {args.out}: {sum(len(x) for x in out_h):,} proposed edges over "
          f"{sum(1 for x in summary if x[6] > 0)} relations "
          f"(skipped {sum(1 for x in summary if x[6] == 0)} below precision {args.min_prec})")


if __name__ == "__main__":
    main()
