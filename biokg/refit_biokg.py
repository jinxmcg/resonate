"""CP-B4: refit a CP-B1/B3 tiered table (COMPACT_B.md, CP-B4).

The tiered table as filed is INIT ONLY -- k-means + per-cluster PCA of a trained
model's rows, never trained. This trains the narrow coefficients and the per-tier
subspace banks against the ordinary biokg loss (typed negatives, batches grouped
by relation) plus T=2 distillation from the wide model the table was compressed
from. Operators, temperature, tier membership and cluster assignment stay frozen
exactly as filed, so the parameter count does not move.

Standalone by design: `train_biokg_comp.py` is being edited concurrently, and a
refit needs none of its shells.

Usage: PYTHONPATH=.. python refit_biokg.py --tiered checkpoints/cpb3_dist_s0_0.pt \
           --teacher checkpoints/dist_T2_s0.pt --steps 50000 --device cuda
"""
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from resonate import ResonatE
from resonate_tiered_biokg import TieredTableResonatE
from train_ogb import load, globalize, eval_split


def score(model, src, rel, dst, negs):
    """Logits over [pos | shared negs] through the tiered table's own gathers."""
    z = model.hop(model.embed(src), rel)
    e_pos = model.rows(dst)
    tau = model.log_tau.exp()
    zo = model.out(z, rel)
    l_pos = torch.real((zo * e_pos.conj()).sum(-1, keepdim=True)) * tau
    l_neg = torch.real(zo @ model.rows(negs).conj().t()) * tau
    return torch.cat([l_pos, l_neg], dim=1), z, e_pos


@torch.no_grad()
def teacher_logits(tm, src, rel, dst, negs):
    z = tm.out(tm.hop(tm.embed(src), rel), rel)
    tau = tm.log_tau.exp()
    lp = torch.real((z * tm.E[dst].conj()).sum(-1, keepdim=True))
    ln = torch.real(z @ tm.E[negs].conj().t())
    return torch.cat([lp, ln], dim=1) * tau


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tiered", required=True)
    p.add_argument("--teacher", required=True)
    p.add_argument("--steps", type=int, default=50000)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--neg", type=int, default=4096)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--lam", type=float, default=0.1)
    p.add_argument("--distill-T", type=float, default=2.0)
    p.add_argument("--distill-w", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--probe-every", type=int, default=10000)
    p.add_argument("--probe-size", type=int, default=5000)
    p.add_argument("--save", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--data-root", default="data_ogb")
    a = p.parse_args()
    dev = torch.device(a.device)
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)

    split, offset, n_ent, _, num_nodes = load(a.data_root)
    tr = split["train"]
    h, r, t = globalize(tr, offset)
    n_rel_base = int(r.max()) + 1
    n_rel = 2 * n_rel_base
    deg = np.bincount(np.concatenate([h, t]), minlength=n_ent)

    ck = torch.load(a.tiered, map_location="cpu", weights_only=False)
    ca = ck["args"]
    cut = np.array([int(x) for x in ca["tiers"].split(",")])
    m = TieredTableResonatE(n_ent, ck["n_rel"], np.searchsorted(cut, deg, side="right"),
                            [int(x) for x in ca["widths"].split(",")],
                            k=ca.get("k", 12), block_size=ca.get("block_size", 4),
                            device=dev, subspaces=[int(x) for x in ca["subspaces"].split(",")])
    m.load_state_dict(ck["model"])
    # operators, temperature, tiering and cluster assignment stay exactly as filed
    trainable = []
    for name, q in m.named_parameters():
        train_it = name.startswith(("coef", "proj", "mu"))
        q.requires_grad_(train_it)
        if train_it:
            trainable.append(q)
    print(f"refit {a.tiered}: {m.n_params():,} parameters, "
          f"{sum(q.numel() for q in trainable):,} trainable "
          f"(operators and temperature frozen)", flush=True)

    tck = torch.load(a.teacher, map_location="cpu", weights_only=False)
    tca = tck.get("args", {})
    tm = ResonatE(n_entities=n_ent, n_relations=n_rel, k=tca.get("k", 12), block=True,
                  block_size=tca.get("block_size", 4)).to(dev)
    tm.load_state_dict(tck["model"])
    tm.eval()
    for q in tm.parameters():
        q.requires_grad_(False)

    by_rel = {ri: np.where(r == ri)[0] for ri in range(n_rel_base)}
    rel_w = np.array([len(by_rel[ri]) for ri in range(n_rel_base)], dtype=np.float64)
    rel_w /= rel_w.sum()
    tail_range, head_range = {}, {}
    for ri in range(n_rel_base):
        i0 = by_rel[ri][0]
        tt, ht = tr["tail_type"][i0], tr["head_type"][i0]
        tail_range[ri] = (offset[tt], offset[tt] + int(num_nodes[tt]))
        head_range[ri] = (offset[ht], offset[ht] + int(num_nodes[ht]))

    opt = torch.optim.Adam(trainable, lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps)
    va = split["valid"]
    pi = np.random.default_rng(0).choice(len(va["head"]),
                                         size=min(a.probe_size, len(va["head"])), replace=False)
    probe = {k: np.asarray(v)[pi] for k, v in va.items()}

    t0 = time.time()
    m.train()
    for step in range(1, a.steps + 1):
        ri = int(rng.choice(n_rel_base, p=rel_w))
        idx = by_rel[ri][rng.integers(0, len(by_rel[ri]), size=a.batch)]
        if rng.random() < 0.5:
            src, dst, rel_id, lo_hi = h[idx], t[idx], ri, tail_range[ri]
        else:
            src, dst, rel_id, lo_hi = t[idx], h[idx], ri + n_rel_base, head_range[ri]
        negs = torch.from_numpy(rng.integers(lo_hi[0], lo_hi[1], size=a.neg)).to(dev)
        src_t = torch.from_numpy(src).to(dev)
        dst_t = torch.from_numpy(dst).to(dev)
        rel_t = torch.full((len(src),), rel_id, device=dev)
        logits, z, e_pos = score(m, src_t, rel_t, dst_t, negs)
        target = torch.zeros(len(src), dtype=torch.long, device=dev)
        loss = F.cross_entropy(logits, target)
        loss = loss + a.lam * (z - e_pos).abs().pow(2).sum(-1).mean()
        T = a.distill_T
        tl = teacher_logits(tm, src_t, rel_t, dst_t, negs)
        loss = loss + a.distill_w * T * T * F.kl_div(
            F.log_softmax(logits / T, dim=1), F.softmax(tl / T, dim=1), reduction="batchmean")
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        sched.step()
        if step % 2000 == 0 or step == a.steps:
            print(f"step {step}/{a.steps}  loss {loss.item():.3f}  "
                  f"lr {sched.get_last_lr()[0]:.2e}  ({time.time()-t0:.0f}s)", flush=True)
        if a.probe_every and step % a.probe_every == 0 and step < a.steps:
            m.eval(); m.build_eval_table()
            eval_split(m, probe, offset, n_rel, dev, label=f"probe@{step}")
            m.eval_table = None; m.train()

    m.eval(); m.build_eval_table()
    eval_split(m, va, offset, n_rel, dev, label="valid refit (full)")
    m.eval_table = None
    torch.save({"model": m.state_dict(), "offset": offset, "n_rel": ck["n_rel"],
                "args": {**ca, "refit_steps": a.steps, "refit_lr": a.lr,
                         "refit_from": a.tiered, "refit_teacher": a.teacher}}, a.save)
    print(f"saved {a.save}", flush=True)


if __name__ == "__main__":
    main()
