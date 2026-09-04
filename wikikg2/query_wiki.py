"""Query CLI for the ogbl-wikikg2 model — the medicine demo, on Wikidata.

  uv run python query_wiki.py --model lever_entbias200k.pt Q42 P50 -k 10
      (Douglas Adams --author--> ?)       forward:  (h, r, ?)
  uv run python query_wiki.py --model M.pt Q42 ~P50 -k 10
      (? --author--> Douglas Adams)        reverse:  (?, r, t)
  uv run python query_wiki.py --model M.pt Q42 P50.P136        # 2-hop chain
  uv run python query_wiki.py --model M.pt "Douglas Adams" "educated at"   # names
  uv run python query_wiki.py --model M.pt "Marie Curie" "~doctoral advisor.award received"
  uv run python query_wiki.py --model M.pt --index build|check|bench

Every answer is stamped against the graph (the oracle):
  [ok]        the edge is in the TRAINING graph (a citable fact)
  [held-out]  not in training; it IS a validation/test edge — the model
              proposed a real later edge it never saw
  [NOVEL]     in no split — a proposal (missing-edge candidate)
For chains the oracle is exact traversal of the training graph.

Two engines give the same answer: `exact` scores all 2.5M rows on the
GPU (one hop + one matmul); `index` applies the compiled operator and
does one ANN lookup in an fp16 HNSW index (faiss IndexHNSWSQ, the
H19 serving form: 848 B/row incl. links on the bio graphs). --index
build writes <model>.index.faiss; check compares top-k against exact
on random queries; bench times single queries on the CPU.

Labels come from the Wikidata API (cached in wikidata_labels.json);
the model itself only ever sees integer ids.
"""

import argparse
import csv
import gzip
import json
import os
import time
import urllib.parse
import urllib.request

import numpy as np
import torch

from resonate import cnorm
from train_wiki import load, load_model

UA = "resonate-wiki-demo/0.1 (research; ogbl-wikikg2)"


# ----------------------------------------------------------------- mappings
def mappings(root="data_ogb"):
    base = os.path.join(root, "ogbl_wikikg2", "mapping")
    with gzip.open(os.path.join(base, "nodeidx2entityid.csv.gz"), "rt") as f:
        rd = csv.reader(f); next(rd)
        ent = [q for _, q in rd]
    with gzip.open(os.path.join(base, "reltype2relid.csv.gz"), "rt") as f:
        rd = csv.reader(f); next(rd)
        rel = [p for _, p in rd]
    return ent, {q: i for i, q in enumerate(ent)}, rel, \
        {p: i for i, p in enumerate(rel)}


class Labels:
    def __init__(self, path="wikidata_labels.json"):
        self.path = path
        self.d = json.load(open(path)) if os.path.exists(path) else {}

    def fetch(self, ids):
        need = [i for i in dict.fromkeys(ids) if i not in self.d]
        for i in range(0, len(need), 50):
            chunk = need[i:i + 50]
            url = ("https://www.wikidata.org/w/api.php?action=wbgetentities"
                   f"&ids={'|'.join(chunk)}&props=labels&languages=en|mul"
                   "&format=json")
            try:
                req = urllib.request.Request(url, headers={"User-Agent": UA})
                js = json.load(urllib.request.urlopen(req, timeout=20))
                for k, v in js.get("entities", {}).items():
                    lb = v.get("labels", {})
                    self.d[k] = (lb.get("en") or lb.get("mul") or {}).get(
                        "value", k)
            except Exception as e:  # offline: fall back to ids
                for k in chunk:
                    self.d.setdefault(k, k)
        json.dump(self.d, open(self.path, "w"))

    def __call__(self, i):
        return self.d.get(i, i)


# -------------------------------------------------------------------- graph
class Oracle:
    """Edge sets per split + CSR adjacency of the training graph over
    directed ops (r forward, r + R reverse)."""

    def __init__(self, split, n_ent, n_rel_base):
        self.R = n_rel_base
        self.sets = {}
        for name in ("train", "valid", "test"):
            s = split[name]
            h = np.asarray(s["head"]).astype(np.int64)
            r = np.asarray(s["relation"]).astype(np.int64)
            t = np.asarray(s["tail"]).astype(np.int64)
            key = (h * n_rel_base + r) * n_ent + t
            self.sets[name] = np.sort(key)
        tr = split["train"]
        h = np.asarray(tr["head"]).astype(np.int64)
        r = np.asarray(tr["relation"]).astype(np.int64)
        t = np.asarray(tr["tail"]).astype(np.int64)
        src = np.concatenate([h, t]); dst = np.concatenate([t, h])
        op = np.concatenate([r, r + n_rel_base])
        key = op * n_ent + src
        o = np.argsort(key, kind="stable")
        self.key, self.dst = key[o], dst[o]
        self.n_ent = n_ent

    def has(self, name, h, r, t):
        key = (h * self.R + r) * self.n_ent + t
        s = self.sets[name]
        i = np.searchsorted(s, key)
        return i < len(s) and s[i] == key

    def neighbors(self, op, src):
        k = op * self.n_ent + src
        lo = np.searchsorted(self.key, k)
        hi = np.searchsorted(self.key, k, side="right")
        return self.dst[lo:hi]

    def reach(self, src, ops, cap=200000):
        front = np.array([src], np.int64)
        for op in ops:
            nxt = [self.neighbors(op, s) for s in front[:cap]]
            front = np.unique(np.concatenate(nxt)) if nxt else front[:0]
        return front


def flag(oracle, src, ops, ans):
    if len(ops) == 1:
        op = ops[0]
        h, r, t = (src, op, ans) if op < oracle.R else (ans, op - oracle.R, src)
        if oracle.has("train", h, r, t):
            return "ok"
        if oracle.has("valid", h, r, t) or oracle.has("test", h, r, t):
            return "held-out"
        return "NOVEL"
    return None  # chains: caller uses reach()


# -------------------------------------------------------------- operators
@torch.no_grad()
def compile_ops(model, ops):
    """Chain of hops as ONE block-diagonal operator P = H_rn ... H_r1."""
    P = None
    for op in ops:
        H = model.H[op]                       # (n_blocks, b, b)
        P = H if P is None else torch.matmul(H, P)
    return P


@torch.no_grad()
def apply_op(model, P, z):
    zb = z.reshape(z.shape[0], -1, model.block_size)
    return cnorm(torch.einsum("kij,bkj->bki", P, zb).reshape(z.shape[0], -1))


@torch.no_grad()
def exact_topk(model, z, k, rel_last=None, chunk=64):
    """Exact readout over all N rows, chunked over queries (a 2.5M-row
    score matrix is 10 MB per query)."""
    ss, ii = [], []
    for i in range(0, z.shape[0], chunk):
        s = model.readout(z[i:i + chunk], rel_last)   # (b, N) incl. bias
        v, j = torch.topk(s, k, dim=1)
        ss.append(v); ii.append(j)
    return torch.cat(ss), torch.cat(ii)


# ------------------------------------------------------------------- index
def index_path(model_path):
    return model_path[:-3] + ".index.faiss"


@torch.no_grad()
def table_vectors(model):
    """[Re E, Im E, b/tau] rows: Re<z,E_i>*tau + b_i == tau * <q, v_i>
    with q = [Re z, Im z, 1]."""
    E = torch.view_as_real(model.table()).reshape(model.n_entities, -1)
    if model.b is not None:
        E = torch.cat([E, (model.b / model.log_tau.exp())[:, None]], 1)
    return E.detach().float().cpu().numpy()


def query_vector(model, z):
    q = torch.view_as_real(z).reshape(z.shape[0], -1)
    if model.b is not None:
        q = torch.cat([q, torch.ones(z.shape[0], 1, device=q.device)], 1)
    return q.detach().float().cpu().numpy()


def augment(X):
    """MIPS -> cosine reduction (Bachrach et al. 2014): append
    sqrt(R^2 - |x|^2) so every row has norm R; a query gets a 0 there,
    inner products are unchanged, and HNSW (which assumes a metric-like
    neighbourhood) stops losing high-norm hub rows. Needed on wikikg2:
    target norms are the popularity channel and span a wide range."""
    n2 = (X ** 2).sum(1)
    R2 = n2.max()
    return np.concatenate([X, np.sqrt(np.maximum(R2 - n2, 0))[:, None]], 1)


def build_index(model, path, M=32, ef_c=200):
    import faiss
    X = augment(table_vectors(model))
    d = X.shape[1]
    t0 = time.time()
    idx = faiss.IndexHNSWSQ(d, faiss.ScalarQuantizer.QT_fp16, M,
                            faiss.METRIC_INNER_PRODUCT)
    idx.hnsw.efConstruction = ef_c
    idx.train(X)
    idx.add(X)
    faiss.write_index(idx, path)
    sz = os.path.getsize(path)
    print(f"index: {X.shape[0]:,} rows x {d} -> {sz/2**30:.2f} GB "
          f"({sz/X.shape[0]:.0f} B/row incl. links), built in "
          f"{time.time()-t0:.0f}s", flush=True)
    return idx


def load_index(path, ef=128):
    import faiss
    idx = faiss.read_index(path)
    idx.hnsw.efSearch = ef
    return idx


def resolve(name, kind, of_map, cache):
    """Q/P id, or a name resolved through the Wikidata search API to the
    first hit that exists in ogbl-wikikg2 (kind: 'item' | 'property')."""
    if name in of_map:
        return name
    url = ("https://www.wikidata.org/w/api.php?action=wbsearchentities"
           f"&search={urllib.parse.quote(name)}&language=en&type={kind}"
           "&limit=10&format=json")
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    js = json.load(urllib.request.urlopen(req, timeout=20))
    hits = [(h["id"], h.get("label", h["id"])) for h in js.get("search", [])]
    for i, lab in hits:
        if i in of_map:
            cache.d[i] = lab
            return i
    raise SystemExit(f"'{name}' -> {[i for i, _ in hits][:5]}: none in ogbl-wikikg2")


# -------------------------------------------------------------------- main
def parse_ops(spec, rel_of, R):
    ops = []
    for tok in spec.split("."):
        rev = tok.startswith("~")
        pid = tok.lstrip("~")
        if pid not in rel_of:
            pid = resolve(pid, "property", rel_of, parse_ops.labels)
        ops.append(rel_of[pid] + (R if rev else 0))
    return ops


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("entity", nargs="?", help="Wikidata Q-id (head)")
    p.add_argument("path", nargs="?", help="P-ids, dotted; ~P = reverse")
    p.add_argument("-k", type=int, default=10)
    p.add_argument("--model", required=True)
    p.add_argument("--engine", choices=["exact", "index"], default="exact")
    p.add_argument("--index", choices=["build", "check", "bench"])
    p.add_argument("--n-check", type=int, default=1000)
    p.add_argument("--ef", type=int, default=128, help="HNSW efSearch")
    p.add_argument("--device", default="cuda")
    p.add_argument("--data-root", default="data_ogb")
    p.add_argument("--no-labels", action="store_true")
    p.add_argument("--edges", action="store_true",
                   help="list the entity's training edges per relation "
                        "(counts + a few examples) and exit")
    args = p.parse_args()
    dev = torch.device(args.device)

    split, n_ent = load(args.data_root)
    ent, ent_of, rel, rel_of = mappings(args.data_root)
    model, ck = load_model(args.model, n_ent, dev)
    R = ck["n_rel"] // 2
    labels = Labels()
    if args.no_labels:
        labels.fetch = lambda ids: None

    if args.index:
        ipath = index_path(args.model)
        if args.index == "build" or not os.path.exists(ipath):
            idx = build_index(model, ipath)
        else:
            idx = load_index(ipath, args.ef)
        if args.index in ("check", "bench"):
            rng = np.random.default_rng(0)
            tr = split["train"]
            pick = rng.choice(len(tr["head"]), args.n_check, replace=False)
            src = torch.from_numpy(np.asarray(tr["head"])[pick]).to(dev)
            rr = torch.from_numpy(np.asarray(tr["relation"])[pick]).to(dev)
            z = model.hop(model.embed(src), rr)
            z = model.out(z, rr)
            ex_s, ex_i = exact_topk(model, z, 10)
            ex_i = ex_i.cpu().numpy(); ex_s = ex_s.cpu().numpy()
            q = query_vector(model, z)
            t0 = time.time()
            q = np.concatenate([q, np.zeros((len(q), 1), np.float32)], 1)
            an_s, an_i = idx.search(q, 10)
            dt = time.time() - t0
            an_s = an_s * model.log_tau.exp().item()
            same = np.mean([set(a) == set(b) for a, b in zip(ex_i, an_i)])
            rec = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(ex_i, an_i)])
            top1 = np.mean(ex_i[:, 0] == an_i[:, 0])
            dev_ = np.abs(np.sort(ex_s, 1) - np.sort(an_s, 1)).max()
            print(f"index check (ef {args.ef}) on {args.n_check} train queries: identical "
                  f"top-10 {same:.3f}, recall@10 {rec:.3f}, same top-1 "
                  f"{top1:.3f}, max |score dev| {dev_:.1e}; batched "
                  f"{1000*dt/args.n_check:.2f} ms/query")
            if args.index == "bench":
                ts = []
                for i in range(200):
                    t0 = time.perf_counter()
                    idx.search(q[i:i + 1], 10)
                    ts.append(time.perf_counter() - t0)
                ts = np.array(ts) * 1000
                print(f"single-query ANN (CPU): median {np.median(ts):.2f} ms, "
                      f"p90 {np.percentile(ts, 90):.2f} ms")
                ts = []
                for i in range(50):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    exact_topk(model, z[i:i + 1], 10)
                    torch.cuda.synchronize()
                    ts.append(time.perf_counter() - t0)
                ts = np.array(ts) * 1000
                print(f"single-query exact readout over {n_ent:,} rows (GPU): "
                      f"median {np.median(ts):.2f} ms")
        if not args.entity:
            return

    if not args.entity:
        raise SystemExit("need ENTITY PATH (e.g. Q42 ~P50) or --index")
    parse_ops.labels = labels
    args.entity = resolve(args.entity, "item", ent_of, labels)
    src = ent_of[args.entity]
    oracle = Oracle(split, n_ent, R)
    if args.edges:
        labels.fetch([args.entity])
        print(f"{args.entity} ({labels(args.entity)}): training edges")
        rows = []
        for op in range(2 * R):
            nb = oracle.neighbors(op, src)
            if len(nb):
                rows.append((op, nb))
        labels.fetch([rel[op % R] for op, _ in rows]
                     + [ent[j] for _, nb in rows for j in nb[:3]])
        for op, nb in rows:
            arrow = "~" if op >= R else ""
            ex = ", ".join(labels(ent[j])[:25] for j in nb[:3])
            print(f"  {arrow}{rel[op % R]:<7} {labels(rel[op % R])[:28]:<28} "
                  f"x{len(nb):<5} e.g. {ex}")
        return
    if not args.path:
        raise SystemExit("need a relation path")
    ops = parse_ops(args.path, rel_of, R)

    with torch.no_grad():
        z0 = model.embed(torch.tensor([src], device=dev))
        P = compile_ops(model, ops)
        z = apply_op(model, P, z0)
        last = torch.tensor([ops[-1]], device=dev)
        z = model.out(z, last)
        t0 = time.time()
        if args.engine == "exact":
            s, i = exact_topk(model, z, args.k)
            s, i = s[0].cpu().numpy(), i[0].cpu().numpy()
        else:
            idx = load_index(index_path(args.model))
            q = query_vector(model, z)
            q = np.concatenate([q, np.zeros((len(q), 1), np.float32)], 1)
            s, i = idx.search(q, args.k)
            s, i = s[0] * model.log_tau.exp().item(), i[0]
        dt = (time.time() - t0) * 1000
    reach = oracle.reach(src, ops) if len(ops) > 1 else None
    ids = [args.entity] + [ent[j] for j in i] + \
        [rel[o % R] for o in ops]
    labels.fetch(ids)
    desc = " . ".join(("~" if o >= R else "") + rel[o % R]
                      + f" ({labels(rel[o % R])})" for o in ops)
    print(f"{args.entity} ({labels(args.entity)})  --[{desc}]-->  "
          f"top-{args.k} via {args.engine} in {dt:.1f} ms")
    n_known = len(oracle.reach(src, ops))
    print(f"  oracle: {n_known} answer(s) reachable in the training graph")
    for rank, (sc, j) in enumerate(zip(s, i), 1):
        if reach is None:
            f = flag(oracle, src, ops, int(j))
        else:
            f = "ok" if j in reach else "NOVEL"
        print(f"  {rank:2d}. {sc:7.3f}  {ent[j]:<12} {labels(ent[j])[:40]:<40} "
              f"[{f}]")


if __name__ == "__main__":
    main()
