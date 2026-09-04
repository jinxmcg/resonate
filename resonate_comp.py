"""Compositional entity rows for ResonatE (pilot).

row(e) = free[e] + lam * mean_{(n, op) in S(e)} hop(free[n], rev(op))

where S(e) is a random subset of e's stored neighbours (up to K per
entity, precomputed from the training graph), rev(op) the reverse
directed operator, and hop the model's own block-unitary transfer
followed by cnorm. If (e, r, n) holds then hop(e, r) ~ n, so
hop(n, r+R) is the model's own estimate of e from that neighbour: a
rare entity's row is the average of what its edges imply about it.

Training: per step, k neighbours are sampled per row and each is
dropped with probability p (neighbourhood dropout — the graph analogue
of masking a token). Evaluation: all K neighbours, no dropout, and the
composed table is built once (`build_eval_table`) so scoring gathers
from it exactly as from a free table.

Only training edges feed S(e). The free row keeps popularity (norm) and
identity; lam (learnable, init 1) sets how much of a row is composed.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from resonate import cnorm
from resonate_wiki import SparseTableResonatE


def build_neighbours(h, r, t, n_ent, R, K, seed=0):
    """(n_ent, K) neighbour ids and (n_ent, K) reverse ops, -1 padded;
    a random K-subset of each entity's directed edges."""
    src = np.concatenate([h, t]); dst = np.concatenate([t, h])
    op = np.concatenate([r, r + R])                    # op of the edge src -> dst
    rev = np.where(op >= R, op - R, op + R)            # reverse op: dst -> src estimate
    perm = np.random.default_rng(seed).permutation(len(src))
    src, dst, rev = src[perm], dst[perm], rev[perm]
    o = np.argsort(src, kind="stable")
    src, dst, rev = src[o], dst[o], rev[o]
    deg = np.bincount(src, minlength=n_ent)
    ptr = np.concatenate([[0], np.cumsum(deg)])
    nb = np.full((n_ent, K), -1, np.int64); ro = np.full((n_ent, K), -1, np.int64)
    take = np.minimum(deg, K)
    rows = np.repeat(np.arange(n_ent), take)
    cols = np.arange(take.sum()) - np.repeat(np.cumsum(take) - take, take)
    idx = ptr[rows] + cols
    nb[rows, cols] = dst[idx]; ro[rows, cols] = rev[idx]
    return nb, ro


class CompTableResonatE(SparseTableResonatE):
    def __init__(self, *a, nb=None, ro=None, k_sample=8, p_drop=0.3, lam_init=1.0,
                 keep=None, ent_gain=False, hub_w=None, **kw):
        super().__init__(*a, **kw)
        dev = self.E_real.device
        # hub_w: (n_ent,) per-neighbour weight, e.g. 1/log(2+deg): a hub
        # like "human" then contributes ~0 to a composed row instead of
        # dragging every person to the same centroid
        self.register_buffer("hub_w", None if hub_w is None else torch.as_tensor(hub_w, device=dev, dtype=torch.float32))
        # per-entity log-gain: restores the popularity (norm) channel for
        # composed-only rows at one float per entity
        self.g = nn.Parameter(torch.zeros(self.n_entities, device=dev)) if ent_gain else None
        self.register_buffer("nb", torch.as_tensor(nb, device=dev))
        self.register_buffer("ro", torch.as_tensor(ro, device=dev))
        # keep: (n_ent,) bool — entities that own a free row; the others
        # have a zero free row and are represented by the composed term
        # only (the parameter-reduced variant; the table is still
        # allocated in this pilot, the reported count excludes dropped rows)
        self.register_buffer("keep", None if keep is None else torch.as_tensor(keep, device=dev))
        if self.keep is not None:
            with torch.no_grad():
                self.E_real[~self.keep] = 0
        self.k_sample = k_sample
        self.p_drop = p_drop
        self.lam = nn.Parameter(torch.tensor(float(lam_init), device=dev))
        self.eval_table = None

    def free_rows(self, idx):
        r = super().rows(idx)
        if self.keep is not None:
            r = r * self.keep[idx].unsqueeze(-1)
        return r

    def n_params(self):
        n = super().n_params()
        if self.keep is not None:
            n -= int((~self.keep).sum()) * 2 * self.m
        return n

    def compose(self, idx, train):
        """Composed rows for idx (flat). Uses k sampled neighbours with
        dropout in training, all K in eval."""
        e = self.free_rows(idx)                                   # (B, M)
        nb, ro = self.nb[idx], self.ro[idx]                       # (B, K)
        if train and self.k_sample < nb.shape[1]:
            pick = torch.rand(nb.shape, device=nb.device).argsort(1)[:, :self.k_sample]
            nb, ro = nb.gather(1, pick), ro.gather(1, pick)
        valid = nb >= 0
        if self.keep is not None:
            # a neighbour without a free row is a zero vector: cnorm(0) has
            # a NaN gradient, so only kept neighbours enter the composition
            valid = valid & self.keep[nb.clamp(min=0)]
        if train and self.p_drop > 0:
            valid = valid & (torch.rand(nb.shape, device=nb.device) >= self.p_drop)
        cnt = valid.sum(1)
        if int(valid.sum()) == 0:
            return e if self.g is None else e * self.g[idx].exp().unsqueeze(1)
        flat_n = nb[valid]; flat_o = ro[valid]
        est = self.hop(cnorm(self.free_rows(flat_n)), flat_o)     # (T, M) unit-norm estimates
        agg = torch.zeros_like(e)
        row_id = torch.arange(len(idx), device=idx.device).unsqueeze(1).expand_as(nb)[valid]
        if self.hub_w is not None:
            w = self.hub_w[flat_n]
            agg.index_add_(0, row_id, est * w.unsqueeze(1))
            den = torch.zeros(len(idx), device=idx.device).index_add_(0, row_id, w)
            agg = agg / den.clamp(min=1e-6).unsqueeze(1)
        else:
            agg.index_add_(0, row_id, est)
            agg = agg / cnt.clamp(min=1).unsqueeze(1)
        out = e + self.lam * agg
        if self.g is not None:
            out = out * self.g[idx].exp().unsqueeze(1)
        return out

    def rows(self, idx):
        if self.eval_table is not None and not self.training:
            return self.eval_table[idx]
        shape = idx.shape
        return self.compose(idx.reshape(-1), self.training).view(*shape, self.m)

    @torch.no_grad()
    def build_eval_table(self, chunk=4096):   # each row gathers K neighbour operators (64x64): keep chunks small
        rows = []
        for i in range(0, self.n_entities, chunk):
            idx = torch.arange(i, min(i + chunk, self.n_entities), device=self.E_real.device)
            rows.append(self.compose(idx, train=False))
        self.eval_table = torch.cat(rows)
        return self.eval_table

    def table(self):
        if self.eval_table is not None:
            return self.eval_table
        return super().table()
