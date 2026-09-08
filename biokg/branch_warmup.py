"""H35F training-only warm-up helpers. H35E inference architecture unchanged."""

import math

import torch
from torch.nn import functional as F

from biokg.joint_operator import retained_loss


ARMS = ("single", "joint", "warmup")
WARM_STEPS = 10000
STEPS = 15000


def configure_phase(model, warm_only):
    if warm_only and model.mode != "dot":
        raise ValueError("Warm-up requires dot B")
    model.requires_grad_(True)
    if warm_only:
        model.a.requires_grad_(False)
        model.gate_logit.requires_grad_(False)
    for parameter in model.parameters():
        parameter.grad = None


def b_training_logits(model, source, relation, positive, negatives):
    """B's scaled logits only: neither A nor gate is read or evaluated."""
    if model.mode != "dot" or not torch.all(relation == relation[0]):
        raise ValueError("Expected dot B and one directed relation per training batch")
    u = model.source_b(source, relation)
    pos = (u * model.target_b(positive, relation)).sum(-1, keepdim=True)
    v = F.normalize(model.B[negatives], dim=-1) * model.tail_scale[relation[0]]
    neg = u @ v.T
    return torch.cat([pos, neg], dim=1) * math.sqrt(model.width) * model.log_tau_b.exp()


def direct_loss(logits, teacher_target):
    if teacher_target.requires_grad or logits.shape != teacher_target.shape:
        raise ValueError("Expected same-shaped frozen teacher logits")
    ce = F.cross_entropy(logits, torch.zeros(len(logits), dtype=torch.long, device=logits.device))
    kd = 4 * F.kl_div(F.log_softmax(logits / 2, dim=1), F.softmax(teacher_target / 2, dim=1),
                      reduction="batchmean")
    return ce + kd, dict(ce=ce, kd_t_squared=kd, trajectory=logits.new_zeros(()))


def training_loss(model, batch, teacher_target, warm_only):
    if warm_only:
        return direct_loss(b_training_logits(model, *batch), teacher_target)
    return retained_loss(model.training_outputs(*batch), teacher_target)


def effective_lr(opt, warm_only, has_b):
    rates = {group["group_name"]: group["lr"] for group in opt.param_groups}
    rates["a"] = 0. if warm_only else rates["a"]
    if has_b:
        rates["gate"] = 0. if warm_only else rates["b"]
    return rates


def assert_gradients(model, warm_only):
    norms = {}
    for name, parameter in model.named_parameters():
        frozen = warm_only and (name.startswith("a.") or name == "gate_logit")
        if frozen:
            assert not parameter.requires_grad and parameter.grad is None, name
            norms[name] = None
        else:
            assert parameter.requires_grad and parameter.grad is not None, name
            norm = float(parameter.grad.norm())
            assert math.isfinite(norm) and norm > 0, (name, norm)
            norms[name] = norm
    return norms


def phase_change_audit(initial, final, warm_only):
    changed = {name: initial[name] != final[name] for name in final}
    for name, value in changed.items():
        should_change = not (warm_only and (name.startswith("a.") or name == "gate_logit"))
        assert value == should_change, (name, value, should_change)
    return dict(initial_hashes=initial, final_hashes=final, changed=changed)
