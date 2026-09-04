"""Row-wise Adagrad for embedding tables.

The optimizer PyTorch-BigGraph, DGL-KE and TorchRec (ROWWISE_ADAGRAD)
use for tables too large for Adam state: ONE accumulator per row — the
running sum of that row's mean squared gradient — instead of two full
moment tensors. State is N floats (0.4 GB at 100M rows) instead of
2 x N x M (230 GB at 100M rows, k=12).

Two properties matter for ResonatE:

* no decay: a row with zero gradient is left exactly alone, so a dense
  run at 47k / 94k entities is *identical* to the sparse, sharded run
  the same optimizer would do at 100M — whatever it scores here is what
  it scores there;
* per-row annealing: a hub touched 1e5 times takes tiny steps, an
  entity touched 8 times keeps a large one — the natural schedule for a
  power-law degree graph.

Complex parameters are updated through their real view (the row's mean
square is taken over 2M reals). Sparse gradients (from
``F.embedding(..., sparse=True)``) update only the rows they carry.
"""

import torch


class RowAdagrad(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-2, eps=1e-10):
        super().__init__(params, dict(lr=lr, eps=eps))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["acc"] = torch.zeros(p.shape[0], device=p.device)
                acc = state["acc"]
                pr = torch.view_as_real(p) if p.is_complex() else p
                g = p.grad
                if g.is_sparse:
                    # row-sparse COO as F.embedding(sparse=True) emits:
                    # indices (1, nnz), values (nnz, M[, 2])
                    g = g.coalesce()
                    assert g.sparse_dim() == 1, "expected row-sparse grad"
                    rows, gv = g.indices()[0], g.values()
                    gr = torch.view_as_real(gv) if gv.is_complex() else gv
                    gr = gr.float()  # fp16/bf16 tables: math in fp32
                    acc.index_add_(0, rows, gr.pow(2).flatten(1).mean(1))
                    scale = group["lr"] / (acc[rows].sqrt() + group["eps"])
                    upd = gr * scale.view(-1, *[1] * (gr.dim() - 1))
                    pr.index_add_(0, rows, upd.to(pr.dtype), alpha=-1.0)
                else:
                    gr = torch.view_as_real(g) if g.is_complex() else g
                    acc.add_(gr.pow(2).flatten(1).mean(1))
                    scale = group["lr"] / (acc.sqrt() + group["eps"])
                    pr.addcmul_(gr, scale.view(-1, *[1] * (gr.dim() - 1)),
                                value=-1.0)
        return loss


def split_params(model):
    """(entity-table params, everything else) — the table gets RowAdagrad,
    the relation operators / tau / biases stay on Adam."""
    table = [model.E]
    ids = {id(q) for q in table}
    rest = [q for q in model.parameters() if id(q) not in ids]
    return table, rest


if __name__ == "__main__":
    # dense == sparse, complex == real-view, untouched rows untouched
    torch.manual_seed(0)
    N, M = 50, 6
    base = torch.randn(N, M, dtype=torch.cfloat)
    rows = torch.tensor([3, 7, 7, 20])
    outs = []
    for sparse in (False, True):
        p = torch.nn.Parameter(base.clone())
        opt = RowAdagrad([p], lr=0.1)
        for _ in range(3):
            opt.zero_grad()
            loss = (p[rows].abs() ** 2).sum()
            loss.backward()
            if sparse:
                p.grad = p.grad.to_sparse(1)  # row-sparse, like F.embedding
            opt.step()
        outs.append(p.detach().clone())
    assert torch.allclose(outs[0], outs[1], atol=1e-6), "dense != sparse"
    touched = torch.zeros(N, dtype=torch.bool)
    touched[rows] = True
    assert torch.equal(outs[0][~touched], base[~touched]), "untouched moved"
    assert not torch.allclose(outs[0][touched], base[touched]), "touched didn't"
    print("RowAdagrad: dense == sparse, untouched rows bit-identical  OK")
