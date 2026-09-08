"""H35C: all student parameters co-adapt; the original KD loss is retained."""

import torch
import torch.nn.functional as F

from biokg.dual_operator import DualOperator, combine_scores


class JointOperator(DualOperator):
    def __init__(self, entity, operator, log_tau, mode, perturbation=.05):
        if mode not in ("single", "or"):
            raise ValueError("H35C fixes a single versus smooth-OR comparison")
        super().__init__(entity, operator, log_tau, mode, temperature=1.,
                         perturbation=perturbation, init_seed=35003)
        self.requires_grad_(True)

    def training_outputs(self, source, relation, positive, negatives):
        qa, qb = self.queries(source, relation)
        ep, en = self.E[positive], self.E[negatives]
        def score(q):
            pos = (q * ep.conj()).sum(-1, keepdim=True).real
            neg = (q @ en.conj().T).real
            return torch.cat([pos, neg], 1) * self.log_tau.exp()
        a = score(qa)
        b = a if self.mode == "single" else score(qb)
        # The first bank retains the original trajectory role. B receives
        # gradients through combined CE/KD; there is no B trajectory loss.
        return combine_scores(a, b, self.mode), qa, ep


def retained_loss(outputs, teacher_mean_logits, temperature=2., kd_weight=1., lam=.1):
    logits, query_a, positive_embedding = outputs
    if teacher_mean_logits.requires_grad:
        raise ValueError("Teacher targets must be detached")
    if logits.shape != teacher_mean_logits.shape or temperature <= 0:
        raise ValueError("Incompatible teacher scores or temperature")
    labels = torch.zeros(len(logits), dtype=torch.long, device=logits.device)
    ce = F.cross_entropy(logits, labels)
    kd = temperature ** 2 * F.kl_div(
        F.log_softmax(logits / temperature, dim=1),
        F.softmax(teacher_mean_logits / temperature, dim=1), reduction="batchmean")
    trajectory = (query_a - positive_embedding).abs().square().sum(-1).mean()
    return ce + kd_weight * kd + lam * trajectory, dict(ce=ce, kd_t_squared=kd, trajectory=trajectory)


def from_student(checkpoint, mode, device="cpu", perturbation=.05):
    state, ca = checkpoint["model"], checkpoint["args"]
    if set(state) != {"E", "H", "log_tau"} or any(ca.get(k, False) for k in (
            "comp", "real", "tied_reverse", "ent_bias", "rel_gain", "low_rank")):
        raise ValueError("Expected the plain complex block student")
    return JointOperator(state["E"].to(device), state["H"].to(device),
                         state["log_tau"].to(device), mode, perturbation)


def restore_joint(checkpoint, device="cpu"):
    if checkpoint.get("model_type") != "H35CJointOperator":
        raise ValueError("Not an H35C checkpoint")
    state = checkpoint["model"]
    model = JointOperator(state["E"].to(device), state["H_b"].to(device),
                          state["log_tau"].to(device), checkpoint["mode"], perturbation=0)
    model.load_state_dict(state)
    return model.eval().requires_grad_(False)
