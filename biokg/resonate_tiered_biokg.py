"""CP-B1: a degree-tiered entity table for the biokg model (COMPACT_K.md CP2/CP3,
ported from resonate_tiered.py to the dense-table model class).

`resonate_tiered.TieredTableResonatE` subclasses the wikikg2 shell
(`resonate_wiki.SparseTableResonatE`, real-view table + row-sparse gathers).
biokg's ladder is trained on `resonate.ResonatE` itself (complex `E`, typed
negatives, dense Adam, or the sparse shell converted back to this format by
`sparse_to_dense.py`), so the tiered table is re-parented here; the table
mathematics are the wikikg2 file's, unchanged.

Every entity keeps its own free row, but its WIDTH depends on its training
degree: tier t stores d_t = 2*w_t real coefficients per entity and K_t shared
projections P (d_t x 2M) with offsets mu (2M), so the full row is
row(e) = mu[c(e)] + coef[e] @ P[c(e)], with c(e) a fixed cluster id inside the
tier. Operators, temperature and everything else are the wide model's, copied
and frozen. Initialised from the wide rows by k-means + per-cluster PCA on the
real view (`init_from_wide`); trainable afterwards by the ordinary loss with
the wide model as distillation teacher (the CP2/CP3 "refit").
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from resonate import ResonatE, cnorm


class TieredTableResonatE(ResonatE):
    def __init__(self, n_entities, n_relations, tier_of, widths, k=12, block_size=4,
                 device=None, rel_gain=False, subspaces=None, cluster_of=None):
        """subspaces: K per tier (CP3); cluster_of: (N,) cluster id within the tier (fixed)."""
        # parent with a 1-row table (no N x M complex allocation), then the tiers
        super().__init__(n_entities=1, n_relations=n_relations, k=k, block=True,
                         block_size=block_size, ent_bias=False, rel_gain=rel_gain)
        del self.E                       # the class property below takes over
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
        # a full-width tier has a complete basis: one subspace reconstructs it
        # exactly, so K is clamped there (and to the tier's own size elsewhere)
        K0 = list(subspaces) if subspaces is not None else [1] * T
        self.K = [1 if 2 * w >= M2 else max(1, min(int(K0[t]), self.sizes[t]))
                  for t, w in enumerate(widths)]
        cl = torch.zeros(n_entities, dtype=torch.long) if cluster_of is None \
            else torch.as_tensor(cluster_of, dtype=torch.long)
        self.register_buffer("cluster_of", cl.to(device))
        self.coef = nn.ParameterList([nn.Parameter(torch.zeros(n, min(2 * w, M2), device=device))
                                      for n, w in zip(self.sizes, widths)])
        # proj[t]: (K_t, d_t, 2M); mu[t]: (K_t, 2M)
        self.proj = nn.ParameterList([nn.Parameter(torch.zeros(K, min(2 * w, M2), M2, device=device))
                                      for K, w in zip(self.K, widths)])
        self.mu = nn.ParameterList([nn.Parameter(torch.zeros(K, M2, device=device)) for K in self.K])
        self.eval_table = None
        self.to(device)

    # --- init from a trained wide table -------------------------------
    @torch.no_grad()
    def init_from_wide(self, E0, kmeans_iters=15, chunk=8192):
        """E0: (N, 2M) real view of the wide model's rows (same k). Per tier:
        k-means into K_t clusters (on the rows) when K_t > 1, then per-cluster
        PCA: coef = (X - mu_c) V_c, proj_c = V_c^T."""
        dev = self.coef[0].device
        kept = []
        for t in range(len(self.widths)):
            idx = torch.nonzero(self.tier_of == t).squeeze(1)
            X = E0[idx].to(dev).float()
            K = self.K[t]
            if K > 1:
                g = torch.Generator(device=dev); g.manual_seed(0)
                cent = X[torch.randperm(len(X), generator=g, device=dev)[:K]].clone()
                for _ in range(kmeans_iters):
                    assign = torch.cat([torch.cdist(X[i:i + chunk], cent).argmin(1)
                                        for i in range(0, len(X), chunk)])
                    for c in range(K):
                        mc = assign == c
                        if mc.any():
                            cent[c] = X[mc].mean(0)
                assign = torch.cat([torch.cdist(X[i:i + chunk], cent).argmin(1)
                                    for i in range(0, len(X), chunk)])
            else:
                assign = torch.zeros(len(X), dtype=torch.long, device=dev)
            self.cluster_of[idx] = assign
            d = self.coef[t].shape[1]
            num = den = 0.0
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
                ev = evals.flip(0).clamp(min=0)
                num += float(ev[:d].sum()) * int(mc.sum())
                den += float(ev.sum()) * int(mc.sum())
            kept.append(num / max(den, 1e-12))
            del X
        return kept                       # variance kept per tier

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
                c = F.embedding(self.local[sel], self.coef[t])            # (n, d)
                cl = self.cluster_of[sel]
                if self.K[t] == 1:
                    out[mask] = c @ self.proj[t][0] + self.mu[t][0]
                else:
                    out[mask] = torch.bmm(c.unsqueeze(1), self.proj[t][cl]).squeeze(1) + self.mu[t][cl]
        return out.view(*idx.shape, 2 * self.m)

    def rows(self, idx):
        r = self._rows_real(idx)
        return torch.view_as_complex(r.view(*idx.shape, self.m, 2))

    def table(self, chunk=8192):
        if self.eval_table is not None:
            return torch.view_as_complex(self.eval_table.view(self.n_entities, self.m, 2))
        with torch.no_grad():
            full = torch.zeros(self.n_entities, 2 * self.m, device=self.tier_of.device)
            for t in range(len(self.widths)):
                idx = torch.nonzero(self.tier_of == t).squeeze(1)
                # the K > 1 path gathers a (chunk, d, 2M) projection stack, so it
                # is chunked; K == 1 shares one matrix and needs no gather at all
                for i in range(0, len(idx), chunk):
                    sl = idx[i:i + chunk]
                    c = self.coef[t][self.local[sl]]
                    if self.K[t] == 1:
                        full[sl] = c @ self.proj[t][0] + self.mu[t][0]
                    else:
                        cl = self.cluster_of[sl]
                        full[sl] = torch.bmm(c.unsqueeze(1), self.proj[t][cl]).squeeze(1) + self.mu[t][cl]
        return torch.view_as_complex(full.view(self.n_entities, self.m, 2))

    @property
    def E(self):   # the biokg scripts index model.E directly
        return self.table()

    def embed(self, idx):
        return cnorm(self.rows(idx))

    def build_eval_table(self):
        self.eval_table = torch.view_as_real(self.table()).reshape(self.n_entities, 2 * self.m).contiguous()

    def table_params(self):
        return list(self.coef)

    def other_params(self):
        ids = {id(p) for p in self.coef}
        return [q for q in self.parameters() if id(q) not in ids and q.requires_grad]

    def table_n_params(self):
        return sum(p.numel() for p in self.coef) + sum(p.numel() for p in self.proj) \
            + sum(p.numel() for p in self.mu)

    def n_params(self):
        """Real-valued parameter count: the tiered table plus the operators."""
        ops = 2 * self.H.numel() + self.log_tau.numel()
        if self.gain is not None:
            ops += self.gain.numel()
        return self.table_n_params() + ops
