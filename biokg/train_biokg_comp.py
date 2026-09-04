"""ResonatE on ogbl-biokg — leaderboard-comparable link prediction.

Protocol: OGB's official split and Evaluator (500 fixed type-matched
negatives per direction, filtered MRR). Leaderboard context (MRR):
TransE 0.745, RotatE 0.799, DistMult 0.804, ComplEx 0.810, top ~0.86.

Adaptations vs train_hetio/train_drkg:
- pure single-hop triples (51 relations, both directions = 102 ops)
- batches grouped by relation (biokg relations have fixed type
  signatures) so sampled-CE negatives are type-matched, mirroring eval
- entity ids are per-type in OGB; we flatten with per-type offsets

Usage: python train_ogb.py --device cuda --steps 50000 --block-size 4 \
           --seed 0 --save model_final_seed0.pt --eval both
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from resonate import ResonatE, cnorm
from resonate_comp import CompTableResonatE, build_neighbours
from resonate_wiki import SparseTableResonatE, clip_grad_norm_
from rowadagrad import RowAdagrad

torch.set_num_threads(4)


def _patch_torch_load():
    """ogb 1.3.6 predates torch>=2.6's weights_only default; its split
    files are numpy pickles from the official OGB source (trusted)."""
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
    d = LinkPropPredDataset("ogbl-biokg", root=root)
    num_nodes = d[0]["num_nodes_dict"]
    types = sorted(num_nodes)
    offset = {}
    off = 0
    for t in types:
        offset[t] = off
        off += int(num_nodes[t])
    split = d.get_edge_split()
    return split, offset, off, types, num_nodes


def load_teacher(path, n_ent, n_rel, dev):
    """Frozen teacher for --distill: rebuilt from its checkpoint args."""
    ck = torch.load(path, map_location=dev, weights_only=False)
    ca = ck.get("args", {})
    tm = ResonatE(n_entities=n_ent, n_relations=n_rel, k=ca.get("k", 12),
                  block=True, block_size=ca.get("block_size", 2),
                  tied_reverse=ca.get("tied_reverse", False),
                  ent_bias=ca.get("ent_bias", False),
                  rel_gain=ca.get("rel_gain", False)).to(dev)
    tm.load_state_dict(ck["model"])
    tm.eval()
    for prm in tm.parameters():
        prm.requires_grad_(False)
    return tm


def teacher_logits(teachers, src, rel_id, dst, negs):
    """Mean of the teachers' raw logits over [pos | shared negs] —
    the score-average rule the ensemble is evaluated with, on the
    training batch's own candidate set."""
    out = None
    with torch.no_grad():
        for tm in teachers:
            zt = tm.out(tm.hop(tm.embed(src), rel_id), rel_id)
            tau = tm.log_tau.exp()
            lp = torch.real((zt * tm.E[dst].conj()).sum(-1, keepdim=True))
            ln = torch.real(zt @ tm.E[negs].conj().t())
            lg = torch.cat([lp, ln], dim=1) * tau
            if tm.b is not None:
                lg = lg + torch.cat([tm.b[dst][:, None],
                                     tm.b[negs][None, :].expand(len(dst), -1)],
                                    dim=1)
            out = lg if out is None else out + lg
    return out / len(teachers)


def globalize(part, offset):
    h = part["head"] + np.array([offset[t] for t in part["head_type"]])
    t = part["tail"] + np.array([offset[t] for t in part["tail_type"]])
    return h.astype(np.int64), part["relation"].astype(np.int64), \
        t.astype(np.int64)


def score_batch(model, src, rel, dst, negs):
    """Logits over [pos | shared negs] for one embedding model.
    Returns (logits, hopped state, positive-target embeddings)."""
    z = model.hop(model.embed(src), rel)
    e_pos = model.rows(dst)
    tau = model.log_tau.exp()
    zo = model.out(z, rel)  # readout gain of the last hop's relation
    l_pos = torch.real((zo * e_pos.conj()).sum(-1, keepdim=True)) * tau
    l_neg = torch.real(zo @ model.rows(negs).conj().t()) * tau
    if model.b is not None:
        l_pos = l_pos + model.b[dst][:, None]
        l_neg = l_neg + model.b[negs][None, :]
    return torch.cat([l_pos, l_neg], dim=1), z, e_pos


@torch.no_grad()
def eval_split(model, part, offset, n_rel, dev, chunk=512, label="valid"):
    """OGB protocol: rank the true entity against the 500 provided
    negatives, both directions; report via the official Evaluator.
    A list of models is scored as their mean (score-average ensemble)."""
    from ogb.linkproppred import Evaluator
    ev = Evaluator(name="ogbl-biokg")
    models = model if isinstance(model, (list, tuple)) else [model]
    for m in models:
        was = m.training; m.eval()
        if hasattr(m, "build_eval_table"):
            m.build_eval_table(chunk=2048)
        m._was_training = was
    h, r, t = globalize(part, offset)
    off_h = np.array([offset[x] for x in part["head_type"]])
    off_t = np.array([offset[x] for x in part["tail_type"]])
    neg_h = part["head_neg"] + off_h[:, None]
    neg_t = part["tail_neg"] + off_t[:, None]
    pos_scores, neg_scores = [], []
    for dir_ in ("tail", "head"):
        for i in range(0, len(h), chunk):
            sl = slice(i, i + chunk)
            if dir_ == "tail":
                src = torch.from_numpy(h[sl]).to(dev)
                rel = torch.from_numpy(r[sl]).to(dev)
                pos = torch.from_numpy(t[sl]).to(dev)
                cand = torch.from_numpy(neg_t[sl]).to(dev)
            else:
                src = torch.from_numpy(t[sl]).to(dev)
                rel = torch.from_numpy(r[sl] + n_rel // 2).to(dev)
                pos = torch.from_numpy(h[sl]).to(dev)
                cand = torch.from_numpy(neg_h[sl]).to(dev)
            sp = sn = 0
            for m in models:
                z = m.out(m.hop(m.embed(src), rel), rel)
                tau = m.log_tau.exp()
                e_pos = m.rows(pos)
                sp = sp + torch.real((z * e_pos.conj()).sum(-1)) * tau
                e_neg = m.rows(cand)  # (B, 500, M)
                sn = sn + torch.real(torch.einsum("bm,bcm->bc", z,
                                                  e_neg.conj())) * tau
                if getattr(m, "b", None) is not None:
                    sp = sp + m.b[pos]
                    sn = sn + m.b[cand]
            pos_scores.append((sp / len(models)).cpu())
            neg_scores.append((sn / len(models)).cpu())
    for m in models:
        if hasattr(m, "eval_table"):
            m.eval_table = None
        if getattr(m, "_was_training", False):
            m.train()
    out = ev.eval({"y_pred_pos": torch.cat(pos_scores),
                   "y_pred_neg": torch.cat(neg_scores)})
    mrr = float(out["mrr_list"].mean())
    h1 = float(out["hits@1_list"].mean())
    h3 = float(out["hits@3_list"].mean())
    h10 = float(out["hits@10_list"].mean())
    print(f"[{label}] MRR {mrr:.4f}  hits@1 {h1:.4f}  hits@3 {h3:.4f}  "
          f"hits@10 {h10:.4f}", flush=True)
    return mrr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=25000)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--neg", type=int, default=4096)
    p.add_argument("--k", type=int, default=12)
    p.add_argument("--block-size", type=int, default=2,
                   help="relation block size b (bxb blocks; H8a)")
    p.add_argument("--tied-reverse", action="store_true",
                   help="reverse ops = adjoint of forward blocks "
                        "(half the relation params; H9c)")
    p.add_argument("--ent-bias", action="store_true",
                   help="additive per-entity score bias "
                        "(popularity channel; H10)")
    p.add_argument("--rel-gain", action="store_true",
                   help="learned per-(relation,direction) diagonal "
                        "gain on the hopped state at readout; hops "
                        "stay unitary (TripleRE-style scaling; H16)")
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--sched", type=str, default="cosine",
                   choices=["cosine", "onecycle"])
    p.add_argument("--max-lr", type=float, default=3e-2,
                   help="one-cycle peak (lr_finder: steepest 3.7e-2)")
    p.add_argument("--lam", type=float, default=0.1)
    p.add_argument("--aux-rp", type=float, default=0.0,
                   help="relation-prediction aux loss weight "
                        "(ComplEx-RP recipe); 0 = off")
    p.add_argument("--n3", type=float, default=0.0,
                   help="N3 regularization weight on the batch's "
                        "factors (Lacroix et al.); 0 = off")
    p.add_argument("--compose", type=float, default=0.0,
                   help="fraction of batches extended to 2-hop chains "
                        "from the train graph (Guu-style compositional "
                        "training); 0 = off")
    p.add_argument("--distill", nargs="*", default=[],
                   help="teacher checkpoints; adds distill-w * T^2 * "
                        "KL(teacher-mean || student) on each batch's "
                        "[pos | negs] logits at temperature distill-T")
    p.add_argument("--distill-w", type=float, default=1.0)
    p.add_argument("--distill-T", type=float, default=1.0)
    p.add_argument("--peers", type=int, default=1,
                   help="H18 online codistillation: train this many "
                        "independently initialised copies on the shared "
                        "batch, each distilling (distill-w, distill-T) "
                        "from the detached mean logits of the OTHER "
                        "copies; copy 0 is saved as the model")
    p.add_argument("--peer-warmup", type=float, default=0.3,
                   help="fraction of steps before the peer KL term "
                        "starts; it ramps linearly to full weight over "
                        "the following 10%% of steps")
    p.add_argument("--comp", type=int, default=0, help="compositional rows: K stored neighbours (0 = off)")
    p.add_argument("--comp-k", type=int, default=8)
    p.add_argument("--comp-drop", type=float, default=0.3)
    p.add_argument("--comp-free-min-deg", type=int, default=0)
    p.add_argument("--comp-gain", action="store_true")
    p.add_argument("--shell", choices=["dense", "sparse"], default="dense",
                   help="sparse: the wikikg2 shell (real-view table, row-sparse "
                        "gradients, RowAdagrad on the table, Adam elsewhere)")
    p.add_argument("--table-lr", type=float, default=0.6, help="RowAdagrad lr (sparse shell)")
    p.add_argument("--table-dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", type=str, default="model_ogb.pt")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--eval", type=str, default="both",
                   choices=["valid", "test", "both"])
    p.add_argument("--eval-only", action="store_true",
                   help="skip training; load --save and evaluate it")
    p.add_argument("--probe-every", type=int, default=0,
                   help="if >0, print valid MRR on a fixed subsample "
                        "every N steps (~seconds per probe)")
    p.add_argument("--probe-size", type=int, default=5000)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--real", action="store_true",
                   help="real-valued model (E in R^{N x 2M}, real bxb "
                        "blocks); same real param count as complex")
    p.add_argument("--data-root", type=str, default="data_ogb",
                   help="ogb dataset root (auto-downloads ogbl-biokg)")
    args = p.parse_args()
    if args.quick:
        args.steps = 2000

    t0 = time.time()
    split, offset, n_ent, types, num_nodes = load(args.data_root)
    tr = split["train"]
    h, r, t = globalize(tr, offset)
    n_rel_base = int(r.max()) + 1
    n_rel = 2 * n_rel_base  # + reverse ops
    print(f"ogbl-biokg loaded in {time.time()-t0:.0f}s: {n_ent:,} "
          f"entities ({', '.join(f'{k}:{num_nodes[k]}' for k in types)}), "
          f"{n_rel_base} relations, {len(h):,} train triples", flush=True)

    # per-relation triple index + target-type ranges (for type-matched
    # negatives; every biokg relation has a fixed type signature)
    by_rel = {ri: np.where(r == ri)[0] for ri in range(n_rel_base)}
    rel_w = np.array([len(by_rel[ri]) for ri in range(n_rel_base)],
                     dtype=np.float64)
    rel_w /= rel_w.sum()
    tail_range, head_range = {}, {}
    for ri in range(n_rel_base):
        i0 = by_rel[ri][0]
        tt, ht = tr["tail_type"][i0], tr["head_type"][i0]
        tail_range[ri] = (offset[tt], offset[tt] + int(num_nodes[tt]))
        head_range[ri] = (offset[ht], offset[ht] + int(num_nodes[ht]))

    compose_data = None
    if args.compose > 0:
        assert args.aux_rp == 0, "compose+aux-rp not supported together"
        # directed-op adjacency (edge lists sorted by source) and
        # type-compatible continuations, for 2-hop chain batches
        srcs_all = np.concatenate([h, t])
        dsts_all = np.concatenate([t, h])
        ops_all = np.concatenate([r, r + n_rel_base])
        adj = {}
        for d in range(n_rel):
            m = ops_all == d
            s_, d_ = srcs_all[m], dsts_all[m]
            o = np.argsort(s_, kind="stable")
            adj[d] = (s_[o], d_[o])
        def _src_range(d):
            return (head_range[d] if d < n_rel_base
                    else tail_range[d - n_rel_base])
        def _dst_range(d):
            return (tail_range[d] if d < n_rel_base
                    else head_range[d - n_rel_base])
        comp = {}
        for d in range(n_rel):
            cands = [d2 for d2 in range(n_rel)
                     if _src_range(d2) == _dst_range(d)]
            w = np.array([len(adj[c][0]) for c in cands], np.float64)
            comp[d] = (cands, w / w.sum())
        compose_data = {"adj": adj, "comp": comp}
        print(f"compose: p={args.compose} of batches become 2-hop "
              f"chains", flush=True)

    rng = np.random.default_rng(args.seed)
    dev = torch.device(args.device)
    torch.manual_seed(args.seed)  # before init — seeds the embeddings
    if args.tied_reverse:
        assert args.aux_rp == 0, "tied-reverse+aux-rp not supported"
    if args.comp:
        nb, ro = build_neighbours(h, r, t, n_ent, n_rel_base, args.comp)
        keep = None
        if args.comp_free_min_deg > 0:
            deg = np.bincount(np.concatenate([h, t]), minlength=n_ent)
            keep = deg >= args.comp_free_min_deg
            print(f"free rows kept for {keep.sum():,} of {n_ent:,} entities (degree >= {args.comp_free_min_deg})", flush=True)
        model = CompTableResonatE(n_ent, n_rel, k=args.k, block_size=args.block_size,
                                  sparse_grad=False, device=dev, ent_bias=args.ent_bias,
                                  rel_gain=args.rel_gain, nb=nb, ro=ro, k_sample=args.comp_k,
                                  p_drop=args.comp_drop, keep=keep, ent_gain=args.comp_gain)
        model.train()
        print(f"compositional rows: K={args.comp}, k={args.comp_k}, drop {args.comp_drop}", flush=True)
    elif args.shell == "sparse":
        assert not args.tied_reverse and args.peers == 1 and args.compose == 0
        DT = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
        model = SparseTableResonatE(n_ent, n_rel, k=args.k, block_size=args.block_size,
                                    sparse_grad=True, device=dev, ent_bias=args.ent_bias,
                                    rel_gain=args.rel_gain, table_dtype=DT[args.table_dtype])
        model.train()
        print(f"sparse shell: row-sparse table grads, RowAdagrad lr {args.table_lr:g}, "
              f"table {args.table_dtype}", flush=True)
    else:
        model = ResonatE(n_entities=n_ent, n_relations=n_rel, k=args.k,
                    block=True, block_size=args.block_size,
                    tied_reverse=args.tied_reverse,
                    ent_bias=args.ent_bias,
                    rel_gain=args.rel_gain).to(dev)
    print(f"params: {model.n_params():,} (M={model.m})", flush=True)
    # H18: peer copies share the batch but have their own random init
    # (the RNG has advanced past copy 0's draw); only copy 0 survives
    peers = [model]
    if args.peers > 1:
        assert not (args.compose or args.n3 or args.aux_rp or args.distill), \
            "peers: only the plain recipe (+ent-bias/rel-gain) is supported"
        for _ in range(args.peers - 1):
            peers.append(ResonatE(n_entities=n_ent, n_relations=n_rel,
                                  k=args.k, block=True,
                                  block_size=args.block_size,
                                  tied_reverse=args.tied_reverse,
                                  ent_bias=args.ent_bias,
                                  rel_gain=args.rel_gain,
                                  ).to(dev))
        print(f"peers: {args.peers} copies, KL w={args.distill_w} "
              f"T={args.distill_T} from {args.peer_warmup:.0%} of steps",
              flush=True)
    teachers = [load_teacher(pth, n_ent, n_rel, dev) for pth in args.distill]
    if teachers:
        assert args.compose == 0, "distill+compose not supported together"
        print(f"distill: {len(teachers)} teachers (M="
              f"{[tm.m for tm in teachers]}), w={args.distill_w}, "
              f"T={args.distill_T}", flush=True)
    if args.shell == "sparse":
        opts = [torch.optim.Adam(model.other_params(), lr=args.lr),
                RowAdagrad(model.table_params(), lr=args.table_lr)]
    else:
        opts = [torch.optim.Adam([q for m in peers for q in m.parameters()],
                                 lr=args.lr)]
    if args.sched == "onecycle":
        scheds = [torch.optim.lr_scheduler.OneCycleLR(
            o, max_lr=args.max_lr, total_steps=args.steps,
            pct_start=0.1) for o in opts]
    else:
        scheds = [torch.optim.lr_scheduler.CosineAnnealingLR(
            o, T_max=args.steps) for o in opts]
    start_step = 0
    if args.eval_only:
        ck = torch.load(args.save, map_location=dev, weights_only=False)
        ca = ck.get("args", {})
        if (ca.get("k", args.k) != args.k
                or ca.get("block_size", 2) != args.block_size
                or ca.get("tied_reverse", False) != args.tied_reverse
                or ca.get("ent_bias", False) != args.ent_bias
                or ca.get("rel_gain", False) != args.rel_gain
                or ca.get("real", False) != args.real):
            model = ResonatE(n_entities=n_ent, n_relations=n_rel,
                            k=ca.get("k", args.k), block=True,
                            block_size=ca.get("block_size", 2),
                            tied_reverse=ca.get("tied_reverse", False),
                            ent_bias=ca.get("ent_bias", False),
                            rel_gain=ca.get("rel_gain", False),

                            ).to(dev)
            print(f"eval-only: rebuilt model from checkpoint args "
                  f"(k={ca.get('k')}, bs={ca.get('block_size', 2)})",
                  flush=True)
        model.load_state_dict(ck["model"])
        print(f"loaded {args.save} for eval-only", flush=True)
        args.steps = 0  # empty training loop
    elif os.path.exists(args.save + ".ckpt"):
        ck = torch.load(args.save + ".ckpt", map_location=dev,
                        weights_only=False)
        for m, sd in zip(peers, ck.get("peers", [ck["model"]])):
            m.load_state_dict(sd)
        for o, sd in zip(opts, ck["opt"]):
            o.load_state_dict(sd)
        for sc, sd in zip(scheds, ck["sched"]):
            sc.load_state_dict(sd)
        start_step = ck["step"]
        print(f"resumed from checkpoint at step {start_step}")

    probe_part = None
    if args.probe_every > 0:
        va = split["valid"]
        pi = np.random.default_rng(0).choice(
            len(va["head"]), size=min(args.probe_size, len(va["head"])),
            replace=False)
        probe_part = {key: np.asarray(v)[pi] for key, v in va.items()}

    t0 = time.time()
    for step in range(start_step + 1, args.steps + 1):
        ri = int(rng.choice(n_rel_base, p=rel_w))
        idx = by_rel[ri][rng.integers(0, len(by_rel[ri]),
                                      size=args.batch)]
        fwd = rng.random() < 0.5
        if fwd:
            src, dst = h[idx], t[idx]
            rel_id, lo_hi = ri, tail_range[ri]
        else:
            src, dst = t[idx], h[idx]
            rel_id, lo_hi = ri + n_rel_base, head_range[ri]
        z = traj = None
        if compose_data is not None and rng.random() < args.compose:
            cands, cw = compose_data["comp"][rel_id]
            d2 = int(cands[int(rng.choice(len(cands), p=cw))])
            s_arr, d_arr = compose_data["adj"][d2]
            lo = np.searchsorted(s_arr, dst)
            cnt = np.searchsorted(s_arr, dst, side="right") - lo
            ok = cnt > 0
            if int(ok.sum()) >= args.batch // 4:
                src, mid = src[ok], dst[ok]
                j = lo[ok] + (rng.random(int(ok.sum()))
                              * cnt[ok]).astype(np.int64)
                dst = d_arr[j]
                z = model.hop(model.embed(torch.from_numpy(src).to(dev)),
                              torch.full((len(src),), rel_id, device=dev))
                e_mid = model.E[torch.from_numpy(mid).to(dev)]
                traj = (z - e_mid).abs().pow(2).sum(-1).mean()
                z = model.hop(z, torch.full((len(src),), d2, device=dev))
                rel_id = d2
                lo_hi = (tail_range[d2] if d2 < n_rel_base
                         else head_range[d2 - n_rel_base])
        negs = torch.from_numpy(
            rng.integers(lo_hi[0], lo_hi[1], size=args.neg)).to(dev)
        src_t = torch.from_numpy(src).to(dev)
        dst_t = torch.from_numpy(dst).to(dev)
        rel_t = torch.full((len(src),), rel_id, device=dev)
        if z is None:
            logits, z, e_pos = score_batch(model, src_t, rel_t, dst_t, negs)
        else:  # compose path: z is already the 2-hop state
            e_pos = model.rows(dst_t)
            zo = model.out(z, rel_id)
            tau = model.log_tau.exp()
            l_pos = torch.real((zo * e_pos.conj()).sum(-1, keepdim=True)) * tau
            l_neg = torch.real(zo @ model.rows(negs).conj().t()) * tau
            if model.b is not None:
                l_pos = l_pos + model.b[dst_t][:, None]
                l_neg = l_neg + model.b[negs][None, :]
            logits = torch.cat([l_pos, l_neg], dim=1)
        tau = model.log_tau.exp()
        target = torch.zeros(z.shape[0], dtype=torch.long, device=dev)
        loss = F.cross_entropy(logits, target)
        if len(peers) > 1:
            # every copy: own CE + trajectory term; then, once warmed
            # up, KL to the detached leave-one-out mean of the others'
            # raw logits (the score-average rule the ensemble is
            # evaluated with) — the same target the offline --distill
            # uses, with the peers as teachers
            outs = [(logits, z, e_pos)]
            for m in peers[1:]:
                outs.append(score_batch(m, src_t, rel_t, dst_t, negs))
            for lg, zj, ej in outs[1:]:
                loss = loss + F.cross_entropy(lg, target) \
                    + args.lam * (zj - ej).abs().pow(2).sum(-1).mean()
            frac = step / args.steps
            w = args.distill_w * min(1.0, max(0.0, (frac - args.peer_warmup)
                                              / 0.1))
            if w > 0:
                T = args.distill_T
                L = torch.stack([lg.detach() for lg, _, _ in outs])
                tot = L.sum(0)
                for j, (lg, _, _) in enumerate(outs):
                    tl = (tot - L[j]) / (len(outs) - 1)
                    kd = F.kl_div(F.log_softmax(lg / T, dim=1),
                                  F.softmax(tl / T, dim=1),
                                  reduction="batchmean")
                    loss = loss + w * T * T * kd
        if teachers:
            T = args.distill_T
            tl = teacher_logits(teachers, torch.from_numpy(src).to(dev),
                                torch.full((len(src),), rel_id, device=dev),
                                torch.from_numpy(dst).to(dev), negs)
            kd = F.kl_div(F.log_softmax(logits / T, dim=1),
                          F.softmax(tl / T, dim=1), reduction="batchmean")
            loss = loss + args.distill_w * T * T * kd
        step_traj = (z - e_pos).abs().pow(2).sum(-1).mean()
        if traj is not None:
            step_traj = (step_traj + traj) / 2
        loss = loss + args.lam * step_traj
        if args.n3 > 0:
            # N3 on the triple's factors. Source-side E is gauge
            # (embed() renormalizes) but target-side E magnitudes act
            # as per-entity biases, and H block norms are free.
            e_src = model.rows(torch.from_numpy(src).to(dev))
            reg = (e_src.abs().pow(3).sum(-1).mean()
                   + e_pos.abs().pow(3).sum(-1).mean()
                   + model.H[rel_id].abs().pow(3).sum())
            loss = loss + args.n3 * reg / 3
        if args.aux_rp > 0:
            # relation prediction: which of the n_rel directed ops
            # links (src, dst)? Apply every relation's blocks to the
            # source state, score each against the true target.
            z0 = model.embed(torch.from_numpy(src).to(dev))
            zb = z0.reshape(args.batch, -1, model.block_size)
            z_all = torch.einsum("rkij,bkj->rbki", model.H, zb)
            z_all = cnorm(z_all.reshape(n_rel, args.batch, -1))
            l_rel = torch.real(torch.einsum("rbm,bm->br", z_all,
                                            e_pos.conj())) * tau
            loss = loss + args.aux_rp * F.cross_entropy(
                l_rel, torch.full((args.batch,), rel_id, device=dev))
        for o in opts:
            o.zero_grad()
        loss.backward()
        if args.shell == "sparse":
            clip_grad_norm_(list(model.parameters()), 1.0)   # sparse-aware
        else:
            for m in peers:  # per copy, so the clip matches the 1-model recipe
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        for o in opts:
            o.step()
        for sc in scheds:
            sc.step()
        if dev.type == "xpu" and step % 250 == 0:
            torch.xpu.synchronize()
            torch.xpu.empty_cache()
        if step % 1000 == 0 or step == args.steps:
            print(f"step {step}/{args.steps}  loss {loss.item():.3f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
        if probe_part is not None and step % args.probe_every == 0:
            eval_split(model, probe_part, offset, n_rel, dev,
                       label=f"probe@{step}")
        if step % 5000 == 0 and step < args.steps:
            torch.save({"model": model.state_dict(), "step": step,
                        "peers": [m.state_dict() for m in peers],
                        "opt": [o.state_dict() for o in opts],
                        "sched": [sc.state_dict() for sc in scheds]}, args.save + ".ckpt")

    if not args.eval_only:
        sd = {k: v for k, v in model.state_dict().items() if k not in ("nb", "ro", "keep")}
        torch.save({"model": sd, "offset": offset,
                    "n_rel": n_rel, "args": vars(args)}, args.save)
        if len(peers) > 1:  # diagnostics only; copy 0 is the model
            torch.save({"peers": [m.state_dict() for m in peers],
                        "args": vars(args)},
                       args.save[:-3] + "_peers.pt")
        if os.path.exists(args.save + ".ckpt"):
            os.remove(args.save + ".ckpt")

    print("\n=== OGB official evaluation ===", flush=True)
    if args.eval in ("valid", "both"):
        eval_split(model, split["valid"], offset, n_rel, dev,
                   label="valid")
    if args.eval in ("test", "both"):
        eval_split(model, split["test"], offset, n_rel, dev,
                   label="test")
    if len(peers) > 1:  # valid-only diagnostics: the other copies, the ensemble
        for j, m in enumerate(peers[1:], 1):
            eval_split(m, split["valid"], offset, n_rel, dev,
                       label=f"valid peer{j}")
        eval_split(peers, split["valid"], offset, n_rel, dev,
                   label=f"valid {len(peers)}-peer ensemble")
    print("leaderboard context: TransE 0.745, RotatE 0.799, "
          "DistMult 0.804, ComplEx 0.810, top ~0.86")


if __name__ == "__main__":
    main()
