"""ResonatE with a scale-ready entity table (ogbl-wikikg2 and beyond).

Same model as resonate.ResonatE (block-unitary relation operators,
unit-norm complex spectra, cnorm after every hop, Re<.,.> * exp(tau)
readout), but the entity table is stored THROUGH ITS REAL VIEW as an
(N, 2M) fp32 tensor and read with F.embedding(..., sparse=True):

* the gather returns exactly the rows a batch touches, and autograd
  emits a row-sparse COO gradient for them — no (N, 2M) dense gradient
  is ever materialised (complex sparse embeddings are unsupported in
  torch, hence the real view);
* rowadagrad.RowAdagrad consumes that sparse gradient and keeps one
  accumulator per row, so a step costs O(rows touched), not O(N).

view_as_complex on the gathered (B, M, 2) rows is a free reinterpret,
so every downstream op (hop, readout) is unchanged complex arithmetic.
`selftest()` checks the sparse path against the dense parent model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from resonate import ResonatE, cnorm


class SparseTableResonatE(ResonatE):
    def __init__(self, n_entities, n_relations, k=12, block_size=4,
                 sparse_grad=True, device=None, ent_bias=False,
                 rel_gain=False, table_dtype=torch.float32,
                 low_rank=0, low_rank_local=False):
        # build the parent with a 1-row table (the parent allocates
        # its table on CPU; at 2.5M x 144 complex that is a 3 GB
        # detour), then replace it with the real-view table on device
        super().__init__(n_entities=1, n_relations=n_relations, k=k,
                         block=True, block_size=block_size,
                         ent_bias=False, rel_gain=rel_gain,
                         low_rank=low_rank, low_rank_local=low_rank_local)
        self.n_entities = n_entities
        self.sparse_grad = sparse_grad
        del self.E
        e = cnorm(torch.randn(n_entities, self.m, dtype=torch.cfloat,
                              device=device))
        # storage dtype of the table only: gathered rows are upcast to
        # fp32 before any arithmetic (operators, scores stay fp32);
        # fp16/bf16 halve the table and the optimiser writes into it
        self.E_real = nn.Parameter(
            torch.view_as_real(e).reshape(n_entities, 2 * self.m)
            .contiguous().to(table_dtype))
        if ent_bias:
            self.b = nn.Parameter(torch.zeros(n_entities, device=device))
        self.to(device)

    # --- table access -------------------------------------------------
    def rows(self, idx: torch.Tensor) -> torch.Tensor:
        """Raw complex rows E[idx], shape idx.shape + (M,), gathered so
        that the backward pass is row-sparse."""
        r = F.embedding(idx, self.E_real, sparse=self.sparse_grad).float()
        return torch.view_as_complex(r.view(*idx.shape, self.m, 2))

    def table(self) -> torch.Tensor:
        """The whole table as a complex (N, M) view (no copy)."""
        return torch.view_as_complex(
            self.E_real.float().view(self.n_entities, self.m, 2))

    @property
    def E(self):  # parent-API compatibility for read-only uses
        return self.table()

    def embed(self, idx: torch.Tensor) -> torch.Tensor:
        return cnorm(self.rows(idx))

    def readout(self, z, r=None):
        if r is not None:
            z = self.out(z, r)
        s = torch.real(z @ self.table().conj().t()) * self.log_tau.exp()
        return s if self.b is None else s + self.b

    def table_params(self):
        return [self.E_real]

    def other_params(self):
        ids = {id(self.E_real)}
        return [q for q in self.parameters() if id(q) not in ids]


def score_batch(model, src, rel, dst, negs):
    """Logits over [pos | shared negs]; returns (logits, hopped state,
    positive-target rows). rel may be per-row (B,)."""
    z = model.hop(model.embed(src), rel)
    e_pos = model.rows(dst)
    tau = model.log_tau.exp()
    zo = model.out(z, rel)
    l_pos = torch.real((zo * e_pos.conj()).sum(-1, keepdim=True)) * tau
    l_neg = torch.real(zo @ model.rows(negs).conj().t()) * tau
    if model.b is not None:
        l_pos = l_pos + model.b[dst][:, None]
        l_neg = l_neg + model.b[negs][None, :]
    return torch.cat([l_pos, l_neg], dim=1), z, e_pos


def clip_grad_norm_(params, max_norm: float):
    """Global-norm clip that accepts row-sparse (COO) gradients: the
    norm of a coalesced sparse grad is the norm of its values.
    Grads are left coalesced so the optimizer need not redo it."""
    norms = []
    for p in params:
        g = p.grad
        if g is None:
            continue
        if g.is_sparse:
            g = g.coalesce()
            p.grad = g
            v = g.values()
        else:
            v = g
        if v.is_complex():
            v = torch.view_as_real(v)
        norms.append(v.norm())
    if not norms:
        return torch.tensor(0.0)
    total = torch.stack(norms).norm()
    coef = (max_norm / (total + 1e-6)).clamp(max=1.0)
    for p in params:
        if p.grad is not None:
            p.grad.mul_(coef)
    return total


def selftest():
    """sparse == dense gradient, same scores as the parent model,
    clip handles sparse grads, RowAdagrad leaves untouched rows."""
    from rowadagrad import RowAdagrad
    torch.manual_seed(0)
    N, R, k, bs = 300, 6, 4, 2
    dense = ResonatE(n_entities=N, n_relations=R, k=k, block=True,
                     block_size=bs)
    sp = SparseTableResonatE(N, R, k=k, block_size=bs, sparse_grad=True)
    sp.E_real.data.copy_(torch.view_as_real(dense.E.data).reshape(N, -1))
    sp.H.data.copy_(dense.H.data)
    src = torch.randint(0, N, (16,))
    rel = torch.randint(0, R, (16,))
    dst = torch.randint(0, N, (16,))
    negs = torch.randint(0, N, (40,))
    # scores
    zd = dense.hop(dense.embed(src), rel)
    ld = torch.cat([torch.real((zd * dense.E[dst].conj()).sum(-1, keepdim=True)),
                    torch.real(zd @ dense.E[negs].conj().t())], 1) \
        * dense.log_tau.exp()
    ls, _, _ = score_batch(sp, src, rel, dst, negs)
    assert torch.allclose(ld, ls, atol=1e-5), "scores differ"
    # gradients
    F.cross_entropy(ld, torch.zeros(16, dtype=torch.long)).backward()
    F.cross_entropy(ls, torch.zeros(16, dtype=torch.long)).backward()
    gd = torch.view_as_real(dense.E.grad).reshape(N, -1)
    assert sp.E_real.grad.is_sparse, "expected sparse grad"
    gs = sp.E_real.grad.coalesce().to_dense()
    assert torch.allclose(gd, gs, atol=1e-6), "sparse grad != dense grad"
    assert torch.allclose(torch.view_as_real(dense.H.grad),
                          torch.view_as_real(sp.H.grad), atol=1e-6)
    # clip
    tot = clip_grad_norm_(list(sp.parameters()), 0.5)
    assert abs(sp.E_real.grad.coalesce().values().norm() ** 2
               + torch.view_as_real(sp.H.grad).norm() ** 2
               + sp.log_tau.grad ** 2 - 0.25) < 1e-4 or tot < 0.5
    # RowAdagrad leaves untouched rows bit-identical
    before = sp.E_real.data.clone()
    RowAdagrad(sp.table_params(), lr=0.3).step()
    touched = torch.zeros(N, dtype=torch.bool)
    touched[torch.cat([src, dst, negs])] = True
    assert torch.equal(sp.E_real.data[~touched], before[~touched])
    assert not torch.allclose(sp.E_real.data[touched], before[touched])
    # eval-style 2-D gather
    cand = torch.randint(0, N, (5, 7))
    assert sp.rows(cand).shape == (5, 7, k * k)
    assert torch.allclose(sp.rows(cand), sp.table()[cand])
    print("SparseTableResonatE selftest OK")


if __name__ == "__main__":
    selftest()
