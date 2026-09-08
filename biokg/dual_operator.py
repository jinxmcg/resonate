"""H35: a shared frozen entity table with one or two complex operator banks.

Standalone pilot model: never change the legacy ResonatE scorer implicitly.
Only H_b is trainable. H_a (when present), E and log_tau are frozen parameters.
"""

import math

import torch
from torch import nn

from resonate import cnorm


MODES = ("single", "mean", "or", "and")


def combine_scores(a, b, mode, temperature=1.0):
    if temperature <= 0:
        raise ValueError("Combination temperature must be positive")
    if mode == "single":
        return b
    if mode == "mean":
        return (a + b) * .5
    if mode == "or":
        return temperature * (torch.logaddexp(a / temperature, b / temperature) - math.log(2))
    if mode == "and":
        return -temperature * (torch.logaddexp(-a / temperature, -b / temperature) - math.log(2))
    raise ValueError(f"Unknown mode: {mode}")


def branch_b_responsibility(a, b, mode, temperature=1.0):
    """Derivative of the combined score w.r.t. B, for diagnostics only."""
    if mode == "single":
        return torch.ones_like(b)
    if mode == "mean":
        return torch.full_like(b, .5)
    sign = 1 if mode == "or" else -1
    return torch.sigmoid(sign * (b - a) / temperature)


class DualOperator(nn.Module):
    def __init__(self, entity, operator, log_tau, mode, temperature=1.0,
                 perturbation=.05, init_seed=35003):
        super().__init__()
        if mode not in MODES or temperature <= 0 or perturbation < 0:
            raise ValueError("Invalid mode, temperature or perturbation")
        if entity.dtype != torch.complex64 or operator.dtype != torch.complex64:
            raise ValueError("H35 requires complex64/fp32")
        if operator.ndim != 4 or operator.shape[-1] != operator.shape[-2]:
            raise ValueError("Expected (relations, blocks, b, b) operator")
        if entity.shape[1] != operator.shape[1] * operator.shape[-1]:
            raise ValueError("Entity width and block layout differ")
        self.mode, self.temperature = mode, float(temperature)
        self.block_size = operator.shape[-1]
        self.E = nn.Parameter(entity.detach().clone(), requires_grad=False)
        self.log_tau = nn.Parameter(log_tau.detach().clone(), requires_grad=False)
        self.H_a = None if mode == "single" else nn.Parameter(
            operator.detach().clone(), requires_grad=False)
        initial_b = operator.detach().clone()
        if mode != "single" and perturbation:
            generator = torch.Generator(device="cpu").manual_seed(init_seed)
            noise = torch.randn(operator.shape, dtype=torch.complex64, generator=generator)
            noise = noise.to(operator.device)
            norm = lambda x: x.abs().square().sum((-2, -1), keepdim=True).sqrt()
            initial_b += perturbation * noise * norm(operator) / norm(noise).clamp_min(1e-12)
        self.H_b = nn.Parameter(initial_b)

    def query(self, source, relation, bank):
        x = cnorm(self.E[source])
        blocks = x.reshape(len(x), -1, self.block_size)
        y = torch.einsum("bkij,bkj->bki", bank[relation], blocks)
        return cnorm(y.reshape(len(x), -1))

    def queries(self, source, relation):
        b = self.query(source, relation, self.H_b)
        a = b if self.H_a is None else self.query(source, relation, self.H_a)
        return a, b

    def training_scores(self, source, relation, positive, negatives):
        """One positive and the same shared typed negatives for all rows."""
        qa, qb = self.queries(source, relation)
        ep, en = self.E[positive], self.E[negatives]
        tau = self.log_tau.exp()
        def scores(q):
            pos = (q * ep.conj()).sum(-1, keepdim=True).real
            neg = (q @ en.conj().T).real
            return torch.cat([pos, neg], dim=1) * tau
        b = scores(qb)
        a = b if self.H_a is None else scores(qa)
        return combine_scores(a, b, self.mode, self.temperature)

    def candidate_scores(self, source, relation, candidates):
        """All candidate columns treated identically, for official evaluation."""
        qa, qb = self.queries(source, relation)
        target = self.E[candidates].conj()
        tau = self.log_tau.exp()
        b = torch.einsum("bm,bcm->bc", qb, target).real * tau
        a = b if self.H_a is None else torch.einsum("bm,bcm->bc", qa, target).real * tau
        combined = combine_scores(a, b, self.mode, self.temperature)
        return combined, a, b, (qa * qb.conj()).sum(-1).real

    def n_params(self, trainable_only=False):
        return sum(p.numel() * (2 if p.is_complex() else 1) for p in self.parameters()
                   if not trainable_only or p.requires_grad)


def from_checkpoint(checkpoint, mode, device="cpu", **kwargs):
    ca, state = checkpoint["args"], checkpoint["model"]
    if any(ca.get(name, False) for name in (
            "comp", "tied_reverse", "ent_bias", "rel_gain", "low_rank", "real")):
        raise ValueError("H35 supports only the plain complex block student")
    if set(state) != {"E", "H", "log_tau"}:
        raise ValueError("Unexpected source checkpoint state")
    return DualOperator(state["E"].to(device), state["H"].to(device),
                        state["log_tau"].to(device), mode, **kwargs)


def restore_adapted(checkpoint, device="cpu"):
    """Restore a self-contained H35 inference checkpoint, no source model needed."""
    if checkpoint.get("model_type") != "H35DualOperator":
        raise ValueError("Not an H35 checkpoint")
    state = checkpoint["model"]
    model = DualOperator(state["E"].to(device), state["H_b"].to(device),
                         state["log_tau"].to(device), checkpoint["mode"],
                         temperature=checkpoint["temperature"], perturbation=0)
    model.load_state_dict(state)
    return model.eval().requires_grad_(False)
