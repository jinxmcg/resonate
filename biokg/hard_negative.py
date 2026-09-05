"""H31: an auxiliary training-only hard-negative CE, not full masking.

The original sampled CE remains intact. Selection is detached and uses
only scores of the current training query and known TRAIN positive masks.
The random control has the same candidate eligibility/count as top-k.
"""
import torch
import torch.nn.functional as F


def mining_weight(step, maximum, warmup_steps, ramp_steps):
    return maximum * min(1., max(0., (step - warmup_steps) / ramp_steps))


@torch.no_grad()
def select_negatives(scores, candidate_ids, known_positive_mask, count, mode, generator=None):
    """Select up to count distinct, training-unknown candidates per query.

    The original candidate pool may contain duplicates. Only its first
    occurrence can enter the auxiliary term; the original CE is untouched.
    If fewer than count are eligible, invalid selected slots are flagged.
    """
    if scores.ndim != 2 or candidate_ids.ndim != 1 or scores.shape[1] != len(candidate_ids):
        raise ValueError("Expected BxK scores and K shared candidate IDs")
    if known_positive_mask.dtype != torch.bool or known_positive_mask.shape != scores.shape:
        raise ValueError("Known-positive mask must be boolean and match scores")
    if not 1 <= count <= scores.shape[1] or mode not in ("random", "topk"):
        raise ValueError("Invalid selection mode/count")
    if mode == "random" and generator is None:
        raise ValueError("Random control requires its own RNG, separate from training sampling")
    order = torch.argsort(candidate_ids, stable=True)
    sorted_ids = candidate_ids[order]
    first_sorted = torch.cat((torch.ones(1, dtype=torch.bool, device=scores.device),
                              sorted_ids[1:] != sorted_ids[:-1]))
    first = torch.zeros_like(first_sorted).scatter_(0, order, first_sorted)
    eligible = ~known_positive_mask & first[None, :]
    selection_scores = (scores if mode == "topk" else
                        torch.rand(scores.shape, device=scores.device, generator=generator))
    selected = selection_scores.masked_fill(~eligible, -torch.inf).topk(count, dim=1).indices
    return selected, eligible.gather(1, selected)


def mixed_negative_ce(logits, selected, selected_valid, weight):
    """(1-w) original CE + w auxiliary CE; gradients use original logits.

    Empty auxiliary rows fall back to the original per-query CE, so they
    do not silently reduce that query's training weight. The designated
    target is always kept. At weight zero use the exact original operation.
    """
    if not 0 <= weight <= 1:
        raise ValueError("Mixture weight must be in [0, 1]")
    target = torch.zeros(len(logits), dtype=torch.long, device=logits.device)
    original = F.cross_entropy(logits, target)
    if weight == 0:
        return original
    if selected.shape != selected_valid.shape or selected_valid.dtype != torch.bool:
        raise ValueError("Selected indices and boolean validity mask must align")
    negative_scores = logits[:, 1:].gather(1, selected).masked_fill(~selected_valid, -torch.inf)
    auxiliary_logits = torch.cat((logits[:, :1], negative_scores), dim=1)
    auxiliary_rows = F.cross_entropy(auxiliary_logits, target, reduction="none")
    original_rows = F.cross_entropy(logits, target, reduction="none")
    auxiliary = torch.where(selected_valid.any(1), auxiliary_rows, original_rows).mean()
    return (1 - weight) * original + weight * auxiliary
