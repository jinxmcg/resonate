"""CFKD1: extra conditional KD on detached leading TRAIN candidates."""

import torch

from biokg.joint_operator import retained_loss


ARMS = ("control", "focused")
STEPS, SEED, TOP_K, FOCUS_WEIGHT = 5000, 3603, 32, .25


@torch.no_grad()
def candidate_mask(student, teacher, positives, k=TOP_K):
    if student.ndim != 2 or student.shape != teacher.shape or not 1 <= k <= student.shape[1]:
        raise ValueError("Invalid score shape or candidate budget")
    if teacher.requires_grad:
        raise ValueError("Teacher targets must be detached")
    if positives.shape != (len(student),) or positives.dtype != torch.long:
        raise ValueError("Expected one TRAIN-positive column index per query")
    student_cutoff = student.topk(k, dim=1).values[:, -1:]
    teacher_cutoff = teacher.topk(k, dim=1).values[:, -1:]
    mask = (student >= student_cutoff) | (teacher >= teacher_cutoff)
    mask.scatter_(1, positives[:, None], True)
    return mask


def conditional_kd(student, teacher, positives, k=TOP_K, temperature=2.):
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    mask = candidate_mask(student.detach(), teacher, positives, k)
    log_student = (student / temperature).masked_fill(~mask, -torch.inf).log_softmax(1)
    log_teacher = (teacher / temperature).masked_fill(~mask, -torch.inf).log_softmax(1)
    # Replace masked log probabilities before multiplying: 0 * (-inf) is NaN.
    log_student = log_student.masked_fill(~mask, 0.)
    log_teacher = log_teacher.masked_fill(~mask, 0.)
    teacher_p = log_teacher.exp().masked_fill(~mask, 0.)
    loss = temperature ** 2 * (teacher_p * (log_teacher - log_student)).sum(1).mean()
    return loss, mask


def focused_loss(outputs, teacher, enabled):
    total, parts = retained_loss(outputs, teacher)
    logits = outputs[0]
    if enabled:
        positives = torch.zeros(len(logits), dtype=torch.long, device=logits.device)
        focus, mask = conditional_kd(logits, teacher, positives)
        selected = mask.sum(1).float().mean()
        # This describes retained probability mass in the unchanged full target.
        mass = (torch.softmax(teacher / 2., 1) * mask).sum(1).mean()
        total = total + FOCUS_WEIGHT * focus
    else:
        focus, selected, mass = (logits.new_zeros(()) for _ in range(3))
    return total, dict(parts, focused_kd_t_squared=focus, selected_candidates=selected,
                       selected_teacher_mass=mass)
