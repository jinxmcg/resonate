"""ResonatE on ogbl-wikikg2 — the scale-path rehearsal.

Protocol: OGB's official split and Evaluator (500 uniformly-drawn
negatives per direction per triple, MRR). Leaderboard context (test
MRR, 2026-09-03): TransE-500 0.426, RotatE-250 0.433, PairRE-200
0.521, AutoSF 0.546, TripleRE 0.579, ComplEx-RP 0.639, InterHT 0.678,
TranS 0.694, StarGraph+TripleRE 0.729, RelEns 0.739 (ensemble).

What differs from train_ogb.py (ogbl-biokg):
- no entity types: negatives are uniform over all 2.5M entities (the
  Evaluator's own candidate distribution), batches mix relations,
  each row gets its own direction (forward op r or reverse op r+R);
- the entity table is SparseTableResonatE: (N, 2M) real view,
  F.embedding(sparse=True) gathers, so a step never touches a dense
  (N, 2M) gradient; with --opt rowadagrad the table trains under
  row-wise Adagrad (one float per row of state) — the H19 recipe —
  while the relation blocks / tau stay on Adam;
- --opt adam is the dense control (dense gather + torch Adam over the
  whole table); it fits the 5090 at N=2.5M and is the paired arm.

Usage (fail-fast first):
  python train_wiki.py --device cuda --steps 3000 --probe-every 1000
  python train_wiki.py --device cuda --steps 100000 --opt rowadagrad \
      --table-lr 0.6 --seed 0 --save model_wiki_s0.pt --eval both
"""

import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from resonate_wiki import SparseTableResonatE, score_batch, clip_grad_norm_
from resonate_comp import CompTableResonatE, build_neighbours
from rowadagrad import RowAdagrad

torch.set_num_threads(8)


def _patch_torch_load():
    """ogb 1.3.6 predates torch>=2.6's weights_only default."""
    orig = torch.load
    if getattr(torch.load, "_ogb_patched", False):
        return
    def load_wo(*a, **k):
        k.setdefault("weights_only", False)
        return orig(*a, **k)
    load_wo._ogb_patched = True
    torch.load = load_wo


def load(root="data_ogb"):
    _patch_torch_load()
    from ogb.linkproppred import LinkPropPredDataset
    d = LinkPropPredDataset("ogbl-wikikg2", root=root)
    n_ent = int(d[0]["num_nodes"])
    split = d.get_edge_split()
    return split, n_ent


@torch.no_grad()
def score_split(model, part, n_rel, dev, chunk=2048):
    """Scores for the OGB protocol: (sp (2N,), sn (2N,500), rel (2N,)),
    tail block then head block, rel = base relation id. A list of
    models is scored as their mean (score-average ensemble)."""
    models = model if isinstance(model, (list, tuple)) else [model]
    h = torch.as_tensor(np.asarray(part["head"]), dtype=torch.long)
    r = torch.as_tensor(np.asarray(part["relation"]), dtype=torch.long)
    t = torch.as_tensor(np.asarray(part["tail"]), dtype=torch.long)
    neg_h = torch.as_tensor(np.asarray(part["head_neg"]), dtype=torch.long)
    neg_t = torch.as_tensor(np.asarray(part["tail_neg"]), dtype=torch.long)
    pos_scores, neg_scores = [], []
    for dir_ in ("tail", "head"):
        for i in range(0, len(h), chunk):
            sl = slice(i, i + chunk)
            if dir_ == "tail":
                src, rel, pos, cand = h[sl], r[sl], t[sl], neg_t[sl]
            else:
                src, rel, pos, cand = t[sl], r[sl] + n_rel // 2, h[sl], neg_h[sl]
            src, rel, pos, cand = (x.to(dev, non_blocking=True)
                                   for x in (src, rel, pos, cand))
            sp = sn = 0
            for m in models:
                z = m.out(m.hop(m.embed(src), rel), rel)
                tau = m.log_tau.exp()
                sp = sp + torch.real((z * m.rows(pos).conj()).sum(-1)) * tau
                sn = sn + torch.real(torch.einsum(
                    "bm,bcm->bc", z, m.rows(cand).conj())) * tau
                if getattr(m, "b", None) is not None:
                    sp = sp + m.b[pos]
                    sn = sn + m.b[cand]
            pos_scores.append((sp / len(models)).cpu())
            neg_scores.append((sn / len(models)).cpu())
    return (torch.cat(pos_scores), torch.cat(neg_scores),
            torch.cat([r, r]))


@torch.no_grad()
def eval_split(model, part, n_rel, dev, chunk=2048, label="valid"):
    """OGB protocol: rank the true entity against the 500 provided
    negatives in both directions; official Evaluator."""
    from ogb.linkproppred import Evaluator
    ev = Evaluator(name="ogbl-wikikg2")
    t0 = time.time()
    was_training = model.training
    model.eval()
    if hasattr(model, "build_eval_table"):
        model.build_eval_table()
    sp, sn, _ = score_split(model, part, n_rel, dev, chunk)
    if hasattr(model, "eval_table"):
        model.eval_table = None
    if was_training:
        model.train()
    out = ev.eval({"y_pred_pos": sp, "y_pred_neg": sn})
    mrr = float(out["mrr_list"].mean())
    h1 = float(out["hits@1_list"].mean())
    h3 = float(out["hits@3_list"].mean())
    h10 = float(out["hits@10_list"].mean())
    print(f"[{label}] MRR {mrr:.4f}  hits@1 {h1:.4f}  hits@3 {h3:.4f}  "
          f"hits@10 {h10:.4f}  (n={len(part['head']):,}, {time.time()-t0:.0f}s)",
          flush=True)
    return mrr


def load_model(path, n_ent, dev, table_dtype=None):
    """Rebuild a SparseTableResonatE from a train_wiki checkpoint.
    table_dtype overrides the stored dtype (teachers: bf16 to fit)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    ca = ck["args"]
    dt = DTYPES[table_dtype or ca.get("table_dtype", "fp32")]
    kw = dict(k=ca["k"], block_size=ca["block_size"], sparse_grad=False,
              device=dev, ent_bias=ca.get("ent_bias", False),
              rel_gain=ca.get("rel_gain", False), table_dtype=dt)
    if ca.get("comp", 0):
        split, _ = load(ca.get("data_root", "data_ogb"))
        tr = split["train"]
        nb, ro = build_neighbours(np.asarray(tr["head"]).astype(np.int64),
                                  np.asarray(tr["relation"]).astype(np.int64),
                                  np.asarray(tr["tail"]).astype(np.int64),
                                  n_ent, ck["n_rel"] // 2, ca["comp"])
        keep = None
        deg = np.bincount(np.concatenate([np.asarray(tr["head"]), np.asarray(tr["tail"])]), minlength=n_ent)
        hub_w = (1.0 / np.log(2.0 + deg)).astype(np.float32) if ca.get("comp_hubw", False) else None
        if ca.get("comp_free_min_deg", 0) > 0:
            keep = deg >= ca["comp_free_min_deg"]
        m = CompTableResonatE(n_ent, ck["n_rel"], nb=nb, ro=ro, k_sample=ca["comp_k"],
                              p_drop=ca["comp_drop"], lam_init=ca["comp_lam"], keep=keep,
                              ent_gain=ca.get("comp_gain", False), hub_w=hub_w, **kw)
    else:
        m = SparseTableResonatE(n_ent, ck["n_rel"], **kw)
    sd = {k: (v.to(dt) if k == "E_real" else v) for k, v in ck["model"].items()
          if k not in ("nb", "ro", "keep", "hub_w")}
    m.load_state_dict(sd, strict=False)
    m.eval()
    if hasattr(m, "build_eval_table"):
        m.build_eval_table()
    for q in m.parameters():
        q.requires_grad_(False)
    return m, ck


@torch.no_grad()
def teacher_logits(teachers, src, rel, dst, negs):
    """Mean of the teachers' raw logits over [pos | shared negs] — the
    score-average rule the ensemble is evaluated with."""
    out = None
    for tm in teachers:
        lg, _, _ = score_batch(tm, src, rel, dst, negs)
        out = lg if out is None else out + lg
    return out / len(teachers)


DTYPES = {"fp32": torch.float32, "fp16": torch.float16,
          "bf16": torch.bfloat16}


def build_model(args, n_ent, n_rel, dev, split=None):
    kw = dict(k=args.k, block_size=args.block_size,
              sparse_grad=(args.opt != "adam"), device=dev,
              ent_bias=args.ent_bias, rel_gain=args.rel_gain,
              table_dtype=DTYPES[getattr(args, "table_dtype", "fp32")])
    if getattr(args, "comp", 0):
        tr = split["train"]
        nb, ro = build_neighbours(np.asarray(tr["head"]).astype(np.int64),
                                  np.asarray(tr["relation"]).astype(np.int64),
                                  np.asarray(tr["tail"]).astype(np.int64),
                                  n_ent, n_rel // 2, args.comp)
        print(f"compositional rows: K={args.comp} stored neighbours "
              f"({(nb[:, 0] >= 0).mean()*100:.0f}% of entities have >= 1), "
              f"k={args.comp_k} sampled, drop {args.comp_drop}, lam0 {args.comp_lam}",
              flush=True)
        keep = None
        deg = np.bincount(np.concatenate([np.asarray(tr["head"]), np.asarray(tr["tail"])]), minlength=n_ent)
        hub_w = (1.0 / np.log(2.0 + deg)).astype(np.float32) if args.comp_hubw else None
        if args.comp_free_min_deg > 0:
            keep = deg >= args.comp_free_min_deg
            print(f"free rows kept for {keep.sum():,} entities (degree >= {args.comp_free_min_deg}); "
                  f"{(~keep).sum():,} composed-only", flush=True)
        return CompTableResonatE(n_ent, n_rel, nb=nb, ro=ro, k_sample=args.comp_k,
                                 p_drop=args.comp_drop, lam_init=args.comp_lam, keep=keep,
                                 ent_gain=args.comp_gain, hub_w=hub_w, **kw)
    return SparseTableResonatE(n_ent, n_rel, **kw)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=100000)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--neg", type=int, default=4096)
    p.add_argument("--k", type=int, default=12)
    p.add_argument("--block-size", type=int, default=4)
    p.add_argument("--ent-bias", action="store_true")
    p.add_argument("--rel-gain", action="store_true")
    p.add_argument("--opt", choices=["adam", "rowadagrad"],
                   default="rowadagrad",
                   help="table optimizer: rowadagrad (sparse grads, one "
                        "float/row of state; H19) or adam (dense control)")
    p.add_argument("--lr", type=float, default=5e-3,
                   help="Adam lr (relation blocks, tau; whole model "
                        "under --opt adam)")
    p.add_argument("--table-lr", type=float, default=0.6,
                   help="RowAdagrad lr for the entity table (H19: 0.3 "
                        "Hetionet, 0.6 DRKG), cosine-decayed")
    p.add_argument("--table-sched", choices=["cosine", "const"],
                   default="cosine")
    p.add_argument("--lam", type=float, default=0.1,
                   help="trajectory term weight ||z - E[t]||^2")
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--distill", nargs="*", default=[],
                   help="teacher checkpoints (row C): adds distill-w * T^2 "
                        "* KL(teacher-mean || student) on each batch's "
                        "[pos | negs] logits at temperature distill-T; "
                        "teachers are loaded with bf16 tables")
    p.add_argument("--distill-w", type=float, default=1.0)
    p.add_argument("--distill-T", type=float, default=1.0)
    p.add_argument("--rel-alpha", type=float, default=1.0,
                   help="relation sampling: a relation with n triples is "
                        "drawn with probability ~ n^alpha, then a uniform "
                        "triple within it. 1.0 = uniform over triples "
                        "(default); 0.5 = sqrt-balanced; 0 = uniform over "
                        "relations")
    p.add_argument("--n3", type=float, default=0.0,
                   help="N3 regularisation weight on the batch's factors "
                        "(Lacroix et al.): cube norms of the source rows, "
                        "target rows and the relation blocks; 0 = off")
    p.add_argument("--table-dtype", choices=["fp32", "fp16", "bf16"],
                   default="fp32",
                   help="storage dtype of the entity table (rows are "
                        "upcast to fp32 for all arithmetic)")
    p.add_argument("--comp", type=int, default=0,
                   help="compositional rows: store up to K neighbours per "
                        "entity (0 = plain free table)")
    p.add_argument("--comp-k", type=int, default=8, help="neighbours sampled per row per step")
    p.add_argument("--comp-drop", type=float, default=0.3, help="neighbour dropout")
    p.add_argument("--comp-lam", type=float, default=1.0, help="initial lam")
    p.add_argument("--comp-hubw", action="store_true",
                   help="weight neighbours by 1/log(2+degree) in the composition")
    p.add_argument("--comp-gain", action="store_true",
                   help="per-entity log-gain on composed rows (popularity channel)")
    p.add_argument("--comp-free-min-deg", type=int, default=0,
                   help="only entities with >= this many training edges keep a "
                        "free row (0 = all); the rest are composed-only")
    p.add_argument("--rev-frac", type=float, default=0.5,
                   help="fraction of rows trained in the head direction "
                        "(?, r, t); 0.5 = symmetric (default)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", type=str, default="model_wiki.pt")
    p.add_argument("--eval", type=str, default="valid",
                   choices=["valid", "test", "both", "none"])
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--probe-every", type=int, default=0)
    p.add_argument("--probe-size", type=int, default=20000)
    p.add_argument("--ckpt-every", type=int, default=10000)
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--data-root", type=str, default="data_ogb")
    args = p.parse_args()

    t0 = time.time()
    split, n_ent = load(args.data_root)
    tr = split["train"]
    h = torch.as_tensor(np.asarray(tr["head"]), dtype=torch.long)
    r = torch.as_tensor(np.asarray(tr["relation"]), dtype=torch.long)
    t = torch.as_tensor(np.asarray(tr["tail"]), dtype=torch.long)
    n_rel_base = int(r.max()) + 1
    n_rel = 2 * n_rel_base
    n_train = len(h)
    print(f"ogbl-wikikg2 loaded in {time.time()-t0:.0f}s: {n_ent:,} "
          f"entities, {n_rel_base} relations, {n_train:,} train triples, "
          f"valid {len(split['valid']['head']):,}, "
          f"test {len(split['test']['head']):,}", flush=True)
    print(f"args: {vars(args)}", flush=True)

    dev = torch.device(args.device)
    torch.manual_seed(args.seed)
    model = build_model(args, n_ent, n_rel, dev, split)
    model.train()
    print(f"params: {model.n_params():,} (M={model.m}, table "
          f"{model.E_real.numel()*model.E_real.element_size()/2**30:.2f} GB "
          f"{args.table_dtype})  opt={args.opt} "
          f"lr={args.lr:g} table_lr={args.table_lr:g} "
          f"table_sched={args.table_sched}", flush=True)
    # training triples on device; sampling never touches the host
    if args.rel_alpha != 1.0:
        # sort by relation so a relation's triples are one contiguous
        # range; draw the relation by n^alpha, then a row inside it
        order = torch.argsort(r, stable=True)
        h, r, t = h[order], r[order], t[order]
        cnt = torch.bincount(r, minlength=n_rel_base).double()
        rel_off = torch.cat([torch.zeros(1, dtype=torch.long), cnt.long().cumsum(0)[:-1]])
        rel_p = (cnt.clamp(min=1) ** args.rel_alpha) * (cnt > 0)
        rel_p = (rel_p / rel_p.sum()).float().to(dev)
        rel_off, rel_cnt = rel_off.to(dev), cnt.long().to(dev)
        print(f"relation sampling alpha={args.rel_alpha}: min/max relation "
              f"prob {rel_p.min():.2e}/{rel_p.max():.2e}", flush=True)
    h, r, t = h.to(dev), r.to(dev), t.to(dev)
    gen = torch.Generator(device=dev)
    gen.manual_seed(args.seed)

    if args.opt == "adam":
        opts = [torch.optim.Adam(model.parameters(), lr=args.lr)]
    else:
        opts = [torch.optim.Adam(model.other_params(), lr=args.lr),
                RowAdagrad(model.table_params(), lr=args.table_lr)]
    scheds = [torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=args.steps)
              for o in opts]
    if args.opt == "rowadagrad" and args.table_sched == "const":
        scheds[1] = torch.optim.lr_scheduler.LambdaLR(opts[1], lambda _: 1.0)

    start_step = 0
    if args.eval_only:
        ck = torch.load(args.save, map_location=dev, weights_only=False)
        ca = ck.get("args", {})
        if ca.get("k") != args.k or ca.get("block_size") != args.block_size \
                or ca.get("ent_bias", False) != args.ent_bias \
                or ca.get("rel_gain", False) != args.rel_gain:
            args.k, args.block_size = ca["k"], ca["block_size"]
            args.ent_bias = ca.get("ent_bias", False)
            args.rel_gain = ca.get("rel_gain", False)
            model = build_model(args, n_ent, n_rel, dev)
        model.load_state_dict(ck["model"])
        print(f"loaded {args.save} for eval-only", flush=True)
        args.steps = 0
    elif os.path.exists(args.save + ".ckpt"):
        ck = torch.load(args.save + ".ckpt", map_location=dev,
                        weights_only=False)
        model.load_state_dict(ck["model"])
        for o, st in zip(opts, ck["opt"]):
            o.load_state_dict(st)
        for sc, st in zip(scheds, ck["sched"]):
            sc.load_state_dict(st)
        gen.set_state(ck["gen"].to("cpu") if hasattr(ck["gen"], "to")
                      else ck["gen"])
        start_step = ck["step"]
        print(f"resumed from checkpoint at step {start_step}", flush=True)

    probe_part = None
    if args.probe_every > 0 and args.steps > 0:
        va = split["valid"]
        pi = np.random.default_rng(0).choice(
            len(va["head"]), size=min(args.probe_size, len(va["head"])),
            replace=False)
        probe_part = {key: np.asarray(v)[pi] for key, v in va.items()}

    teachers = [load_model(pth, n_ent, dev, table_dtype="bf16")[0]
                for pth in args.distill]
    if teachers:
        print(f"distill: {len(teachers)} teachers (bf16 tables), "
              f"w={args.distill_w}, T={args.distill_T}", flush=True)
    params = list(model.parameters())
    t0 = time.time()
    t_log = time.time()
    loss_acc = 0.0
    for step in range(start_step + 1, args.steps + 1):
        if args.rel_alpha != 1.0:
            rs = torch.multinomial(rel_p, args.batch, replacement=True,
                                   generator=gen)
            u = torch.rand(args.batch, device=dev, generator=gen)
            idx = rel_off[rs] + (u * rel_cnt[rs]).long().clamp(max=rel_cnt[rs] - 1)
        else:
            idx = torch.randint(0, n_train, (args.batch,), device=dev,
                                generator=gen)
        rev = torch.rand(args.batch, device=dev, generator=gen) < args.rev_frac
        hb, rb, tb = h[idx], r[idx], t[idx]
        src = torch.where(rev, tb, hb)
        dst = torch.where(rev, hb, tb)
        rel = rb + rev.long() * n_rel_base
        negs = torch.randint(0, n_ent, (args.neg,), device=dev,
                             generator=gen)
        logits, z, e_pos = score_batch(model, src, rel, dst, negs)
        target = torch.zeros(args.batch, dtype=torch.long, device=dev)
        loss = F.cross_entropy(logits, target)
        loss = loss + args.lam * (z - e_pos).abs().pow(2).sum(-1).mean()
        if args.n3 > 0:
            e_src = model.rows(src)
            reg = (e_src.abs().pow(3).sum(-1).mean()
                   + e_pos.abs().pow(3).sum(-1).mean()
                   + model.H[rel].abs().pow(3).sum((-1, -2, -3)).mean())
            loss = loss + args.n3 * reg / 3
        if teachers:
            T = args.distill_T
            tl = teacher_logits(teachers, src, rel, dst, negs)
            kd = F.kl_div(F.log_softmax(logits / T, dim=1),
                          F.softmax(tl / T, dim=1), reduction="batchmean")
            loss = loss + args.distill_w * T * T * kd
        for o in opts:
            o.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_(params, args.clip)
        for o in opts:
            o.step()
        for sc in scheds:
            sc.step()
        loss_acc += loss.item() if step % 50 == 0 else 0.0
        if step % args.log_every == 0 or step == args.steps:
            torch.cuda.synchronize() if dev.type == "cuda" else None
            dt = time.time() - t_log
            n = min(args.log_every, step - start_step)
            print(f"step {step}/{args.steps}  loss {loss.item():.3f}  "
                  f"{1000*dt/n:.1f} ms/step  tau {model.log_tau.exp().item():.2f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
            t_log = time.time()
        if probe_part is not None and step % args.probe_every == 0:
            eval_split(model, probe_part, n_rel, dev, label=f"probe@{step}")
            t_log = time.time()
        if args.ckpt_every > 0 and step % args.ckpt_every == 0 \
                and step < args.steps:
            torch.save({"model": model.state_dict(), "step": step,
                        "opt": [o.state_dict() for o in opts],
                        "sched": [sc.state_dict() for sc in scheds],
                        "gen": gen.get_state()}, args.save + ".ckpt")
            t_log = time.time()

    if not args.eval_only and args.steps > 0:
        sd = {k: v for k, v in model.state_dict().items() if k not in ("nb", "ro", "keep", "hub_w")}
        torch.save({"model": sd, "n_rel": n_rel,
                    "n_ent": n_ent, "args": vars(args)}, args.save)
        if os.path.exists(args.save + ".ckpt"):
            os.remove(args.save + ".ckpt")
        print(f"saved {args.save}  (train {time.time()-t0:.0f}s)", flush=True)

    if args.eval != "none":
        print("\n=== OGB official evaluation (ogbl-wikikg2) ===", flush=True)
        if args.eval in ("valid", "both"):
            eval_split(model, split["valid"], n_rel, dev, label="valid")
        if args.eval in ("test", "both"):
            eval_split(model, split["test"], n_rel, dev, label="test")
        print("leaderboard context (test MRR): RotatE 0.433, PairRE 0.521, "
              "AutoSF 0.546, TripleRE 0.579, ComplEx-RP 0.639, TranS 0.694, "
              "StarGraph+TripleRE 0.729, RelEns 0.739")


if __name__ == "__main__":
    main()
