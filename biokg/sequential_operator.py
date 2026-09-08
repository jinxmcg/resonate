"""H35B: two small operator banks composed with or without a fixed shuffle."""

import torch

from biokg.dual_operator import DualOperator
from resonate import cnorm


class SequentialOperator(DualOperator):
    def __init__(self, entity, operator, log_tau, mixing):
        if mixing not in ("local", "shuffle"):
            raise ValueError("mixing must be local or shuffle")
        super().__init__(entity, operator, log_tau, "mean", perturbation=0)
        self.mode = "single"  # score only the composed query, not a branch blend
        self.mixing = mixing
        width, bs = entity.shape[-1], self.block_size
        nblocks = width // bs
        if nblocks % bs:
            raise ValueError("Shuffle pilot requires block count divisible by block size")
        permutation = torch.arange(width, device=entity.device)
        if mixing == "shuffle":
            permutation = permutation.reshape(nblocks, bs).T.flatten()
        self.register_buffer("permutation", permutation)
        self.register_buffer("inverse_permutation", torch.argsort(permutation))
        with torch.no_grad():
            self.H_b.copy_(torch.eye(bs, dtype=operator.dtype, device=operator.device)
                           .expand_as(operator))

    def transform(self, x, relation):
        first = torch.einsum("bkij,bkj->bki", self.H_a[relation],
                             x.reshape(len(x), -1, self.block_size)).reshape_as(x)
        shuffled = first[:, self.permutation].reshape(len(x), -1, self.block_size)
        second = torch.einsum("bkij,bkj->bki", self.H_b[relation], shuffled).reshape_as(x)
        return first, second[:, self.inverse_permutation]

    def queries(self, source, relation):
        first, second = self.transform(cnorm(self.E[source]), relation)
        return cnorm(first), cnorm(second)


def from_student(checkpoint, mixing, device="cpu"):
    state = checkpoint["model"]
    if set(state) != {"E", "H", "log_tau"}:
        raise ValueError("H35B requires plain source student")
    return SequentialOperator(state["E"].to(device), state["H"].to(device),
                              state["log_tau"].to(device), mixing)


def restore_sequential(checkpoint, device="cpu"):
    if checkpoint.get("model_type") != "H35BSequentialOperator":
        raise ValueError("Not an H35B checkpoint")
    state = checkpoint["model"]
    model = SequentialOperator(state["E"].to(device), state["H_a"].to(device),
                               state["log_tau"].to(device), checkpoint["mixing"])
    model.load_state_dict(state)
    return model.eval().requires_grad_(False)
