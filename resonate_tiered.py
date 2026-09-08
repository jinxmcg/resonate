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
                 sparse_grad=True, device=None, rel_gain=False, subspaces=None, cluster_of=None):
        """subspaces: K per tier (CP3); cluster_of: (N,) cluster id within the tier (fixed)."""
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
        self.K = list(subspaces) if subspaces is not None else [1] * T
        cl = torch.zeros(n_entities, dtype=torch.long) if cluster_of is None else torch.as_tensor(cluster_of, dtype=torch.long)
        self.register_buffer("cluster_of", cl.to(device))
        self.coef = nn.ParameterList([nn.Parameter(torch.zeros(n, min(2 * w, M2), device=device))
                                      for n, w in zip(self.sizes, widths)])
        # proj[t]: (K_t, d_t, 2M); mu[t]: (K_t, 2M)
        self.proj = nn.ParameterList([nn.Parameter(torch.zeros(K, min(2 * w, M2), M2, device=device)) for K, w in zip(self.K, widths)])
        self.mu = nn.ParameterList([nn.Parameter(torch.zeros(K, M2, device=device)) for K in self.K])
        self.eval_table = None

    # --- init from a trained wide table -------------------------------
    @torch.no_grad()
    def init_from_wide(self, E0, kmeans_iters=15):
        """E0: (N, 2M) real view of the wide model's rows (same k). Per tier:
        k-means into K_t clusters (spherical, on the rows) when K_t > 1, then
        per-cluster PCA: coef = (X - mu_c) V_c, proj_c = V_c^T."""
        dev = self.coef[0].device
        for t in range(len(self.widths)):
            idx = torch.nonzero(self.tier_of == t).squeeze(1)
            X = E0[idx].to(dev).float()
            K = self.K[t]
            if K > 1:
                g = torch.Generator(device=dev); g.manual_seed(0)
                cent = X[torch.randperm(len(X), generator=g, device=dev)[:K]].clone()
                for _ in range(kmeans_iters):
                    assign = torch.cat([torch.cdist(X[i:i + 500000], cent).argmin(1) for i in range(0, len(X), 500000)])
                    for c in range(K):
                        mc = assign == c
                        if mc.any():
                            cent[c] = X[mc].mean(0)
                assign = torch.cat([torch.cdist(X[i:i + 500000], cent).argmin(1) for i in range(0, len(X), 500000)])
            else:
                assign = torch.zeros(len(X), dtype=torch.long, device=dev)
            self.cluster_of[idx] = assign
            d = self.coef[t].shape[1]
            for c in range(K):
                mc = assign == c
                if not mc.any():
                    continue
                Xc = X[mc]
                mu = Xc.mean(0)
                Xd = Xc - mu
                C = Xd.t() @ Xd / max(int(mc.sum()), 1)
                evals, evecs = torch.linalg.eigh(C)
                V = evecs.flip(1)[:, :d]
                self.coef[t][mc] = Xd @ V
                self.proj[t][c].copy_(V.t())
                self.mu[t][c].copy_(mu)
            del X

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
                sel = flat[mask]
                c = F.embedding(self.local[sel], self.coef[t], sparse=self.sparse_grad)      # (n, d)
                cl = self.cluster_of[sel]
                if self.K[t] == 1:
                    out[mask] = c @ self.proj[t][0] + self.mu[t][0]
                else:
                    out[mask] = torch.bmm(c.unsqueeze(1), self.proj[t][cl]).squeeze(1) + self.mu[t][cl]
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
                for i in range(0, len(idx), 200000):
                    sl = idx[i:i + 200000]
                    c = self.coef[t][self.local[sl]]; cl = self.cluster_of[sl]
                    full[sl] = torch.bmm(c.unsqueeze(1), self.proj[t][cl]).squeeze(1) + self.mu[t][cl]
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
