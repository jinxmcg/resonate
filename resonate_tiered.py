"""CP2: a degree-tiered entity table for ResonatE (COMPACT_K.md, CP2).

Every entity keeps its own free row, but its WIDTH depends on its training
degree: tier t stores d_t real coefficients per entity and one shared
projection P_t (d_t x 2M) plus an offset mu_t, so the full row is
row(e) = mu_t + coef[e] @ P_t. Operators, temperature and everything else are
the parent model's (typically copied from a trained wide model and frozen).
Initialised from the wide model's rows by per-tier PCA; trained by the
ordinary loss with the wide model as distillation teacher ("refit").
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from resonate_wiki import SparseTableResonatE


class TieredTableResonatE(SparseTableResonatE):
    def __init__(self, n_entities, n_relations, tier_of, widths, k=8, block_size=64,
                 sparse_grad=True, device=None, rel_gain=False):
        # parent with a 1-row table (no 2.5M x M allocation), then the tiers
        super().__init__(1, n_relations, k=k, block_size=block_size, sparse_grad=sparse_grad,
                         device=device, ent_bias=False, rel_gain=rel_gain)
        self.n_entities = n_entities
        self.widths = list(widths)
        M2 = 2 * self.m
        tier_of = torch.as_tensor(tier_of, dtype=torch.long)
        T = len(widths)
        local = torch.zeros(n_entities, dtype=torch.long)
        self.sizes = []
        for t in range(T):
            idx = torch.nonzero(tier_of == t).squeeze(1)
            local[idx] = torch.arange(len(idx))
            self.sizes.append(int(len(idx)))
        self.register_buffer("tier_of", tier_of.to(device))
        self.register_buffer("local", local.to(device))
        self.coef = nn.ParameterList([nn.Parameter(torch.zeros(n, min(2 * w, M2), device=device))
                                      for n, w in zip(self.sizes, widths)])
        self.proj = nn.ParameterList([nn.Parameter(torch.zeros(min(2 * w, M2), M2, device=device)) for w in widths])
        self.mu = nn.ParameterList([nn.Parameter(torch.zeros(M2, device=device)) for _ in widths])
        self.eval_table = None

    # --- init from a trained wide table -------------------------------
    @torch.no_grad()
    def init_from_wide(self, E0):
        """E0: (N, 2M) real view of the wide model's rows (same k). Per-tier
        PCA: coef = (X - mu) V_d, proj = V_d^T."""
        for t in range(len(self.widths)):
            idx = torch.nonzero(self.tier_of == t).squeeze(1)
            X = E0[idx].to(self.coef[t].device).float()
            mu = X.mean(0)
            Xc = X - mu
            C = Xc.t() @ Xc / max(len(idx), 1)
            evals, evecs = torch.linalg.eigh(C)
            V = evecs.flip(1)[:, :self.coef[t].shape[1]]
            self.coef[t].copy_(Xc @ V)
            self.proj[t].copy_(V.t())
            self.mu[t].copy_(mu)
            del X, Xc, C

    # --- table access ---------------------------------------------------
    def _rows_real(self, idx):
        if self.eval_table is not None:
            return F.embedding(idx, self.eval_table)
        flat = idx.reshape(-1)
        out = torch.zeros(flat.shape[0], 2 * self.m, device=flat.device)
        tiers = self.tier_of[flat]
        for t in range(len(self.widths)):
            mask = tiers == t
            if mask.any():
                c = F.embedding(self.local[flat[mask]], self.coef[t], sparse=self.sparse_grad)
                out[mask] = c @ self.proj[t] + self.mu[t]
        return out.view(*idx.shape, 2 * self.m)

    def rows(self, idx):
        r = self._rows_real(idx)
        return torch.view_as_complex(r.view(*idx.shape, self.m, 2))

    def table(self):
        if self.eval_table is not None:
            return torch.view_as_complex(self.eval_table.view(self.n_entities, self.m, 2))
        with torch.no_grad():
            full = torch.zeros(self.n_entities, 2 * self.m, device=self.tier_of.device)
            for t in range(len(self.widths)):
                idx = torch.nonzero(self.tier_of == t).squeeze(1)
                for i in range(0, len(idx), 500000):
                    sl = idx[i:i + 500000]
                    full[sl] = self.coef[t][self.local[sl]] @ self.proj[t] + self.mu[t]
        return torch.view_as_complex(full.view(self.n_entities, self.m, 2))

    def build_eval_table(self):
        self.eval_table = torch.view_as_real(self.table()).reshape(self.n_entities, 2 * self.m).contiguous()

    def table_params(self):
        return list(self.coef)

    def other_params(self):
        ids = {id(p) for p in self.coef} | {id(self.E_real)}
        return [q for q in self.parameters() if id(q) not in ids and q.requires_grad]

    def n_params(self):
        return sum(p.numel() for p in self.coef) + sum(p.numel() for p in self.proj) + sum(p.numel() for p in self.mu)
