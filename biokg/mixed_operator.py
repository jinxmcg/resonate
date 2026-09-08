"""H35E: private dot/L1 branch with a TRAIN-learned directed-relation gate."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from biokg.joint_operator import JointOperator, from_student as base_from_student


MODES = ("single", "dot", "distance")
WIDTH = 32
INIT_SEED = 35130


class MixedOperator(nn.Module):
    def __init__(self, base, mode, width=WIDTH, init_seed=INIT_SEED):
        super().__init__()
        if mode not in MODES or width < 1:
            raise ValueError("Invalid H35E mode or width")
        self.a, self.mode, self.width = base, mode, width
        if mode != "single":
            generator = torch.Generator(device="cpu").manual_seed(init_seed)
            entity = F.normalize(torch.randn(len(base.E), width, generator=generator), dim=-1)
            device, n_rel = base.E.device, len(base.H_b)
            self.B = nn.Parameter(entity.to(device))
            self.head_scale = nn.Parameter(torch.ones(n_rel, width, device=device))
            self.tail_scale = nn.Parameter(torch.ones(n_rel, width, device=device))
            self.shift = nn.Parameter(torch.zeros(n_rel, width, device=device))
            self.gate_logit = nn.Parameter(torch.full((n_rel,), math.log(.05 / .95), device=device))
            self.log_tau_b = nn.Parameter(torch.zeros((), device=device))

    @property
    def E(self):
        return self.a.E

    def source_b(self, source, relation):
        return F.normalize(self.B[source], dim=-1) * self.head_scale[relation] + self.shift[relation]

    def target_b(self, target, relation):
        scale = self.tail_scale[relation]
        if target.ndim == 2:
            scale = scale[:, None, :]
        return F.normalize(self.B[target], dim=-1) * scale

    def paired_b(self, source, target):
        if self.mode == "dot":
            return (source * target).sum(-1) * math.sqrt(self.width)
        if self.mode == "distance":
            return -(source - target).abs().sum(-1)
        raise ValueError("Single model has no private branch")

    def combine(self, a, b, relation):
        weight = self.gate_logit[relation].sigmoid()[:, None]
        return a + weight * self.log_tau_b.exp() * b

    def training_outputs(self, source, relation, positive, negatives):
        a, query, target_a = self.a.training_outputs(source, relation, positive, negatives)
        if self.mode == "single":
            return a, query, target_a
        if not torch.all(relation == relation[0]):
            raise ValueError("H35E shared-negative training requires one directed relation per batch")
        u = self.source_b(source, relation)
        pos = self.paired_b(u, self.target_b(positive, relation))
        v = F.normalize(self.B[negatives], dim=-1) * self.tail_scale[relation[0]]
        if self.mode == "dot":
            neg = (u @ v.T) * math.sqrt(self.width)
        else:
            # Native cdist avoids explicitly constructing batch x negatives x width.
            neg = -torch.cdist(u, v, p=1)
        b = torch.cat([pos[:, None], neg], dim=1)
        return self.combine(a, b, relation), query, target_a

    def candidate_outputs(self, source, relation, candidates):
        a = self.a.candidate_scores(source, relation, candidates)[0]
        if self.mode == "single":
            return a, a, None
        u = self.source_b(source, relation)
        v = self.target_b(candidates, relation)
        b = self.paired_b(u[:, None, :], v) * self.log_tau_b.exp()
        combined = a + self.gate_logit[relation].sigmoid()[:, None] * b
        return combined, a, b

    def n_params(self):
        return sum(p.numel() * (2 if p.is_complex() else 1) for p in self.parameters())

    def optimizer_groups(self):
        groups = [dict(params=list(self.a.parameters()), lr=.0001, group_name="a")]
        if self.mode != "single":
            groups.append(dict(params=[p for name, p in self.named_parameters() if not name.startswith("a.")],
                               lr=.001, group_name="b"))
        return groups


def from_student(checkpoint, mode, device="cpu", width=WIDTH):
    return MixedOperator(base_from_student(checkpoint, "single", device), mode, width)


def restore_mixed(checkpoint, device="cpu"):
    if checkpoint.get("model_type") != "H35EMixedOperator":
        raise ValueError("Not an H35E checkpoint")
    state = checkpoint["model"]
    base = JointOperator(state["a.E"].to(device), state["a.H_b"].to(device),
                         state["a.log_tau"].to(device), "single")
    model = MixedOperator(base, checkpoint["mode"], checkpoint["width"])
    model.load_state_dict(state)
    return model.eval().requires_grad_(False)
