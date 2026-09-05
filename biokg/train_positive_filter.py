"""Exact per-query negative masks constructed exclusively from training triples.

Each relation/direction has a bit-packed adjacency over its active training
entities. Index 0 is an empty sentinel for entities unseen in that role.
There is no split loader here: callers supply ONLY globalized training edges.
The index is not a model parameter and must never be applied at evaluation.
"""
import hashlib

import numpy as np
import torch


class TrainPositiveFilter:
    def __init__(self, head, relation, tail, n_entities, n_rel_base, device="cpu"):
        arrays = [np.asarray(x, dtype=np.int64) for x in (head, relation, tail)]
        h, r, t = arrays
        if any(x.ndim != 1 or len(x) != len(h) for x in arrays):
            raise ValueError("Training head/relation/tail must be equally sized vectors")
        if n_entities < 1 or n_rel_base < 1:
            raise ValueError("Entity and relation counts must be positive")
        if any(np.any((x < 0) | (x >= upper)) for x, upper in
               ((h, n_entities), (r, n_rel_base), (t, n_entities))):
            raise ValueError("Training ID outside declared entity/relation range")
        self.device = torch.device(device)
        self.tables = {}
        self.nbytes = 0
        digest = hashlib.sha256(f"{n_entities}:{n_rel_base}".encode())
        for x in arrays:
            digest.update(np.ascontiguousarray(x).view(np.uint8))
        self.train_sha256 = digest.hexdigest()
        for ri in range(n_rel_base):
            selected = r == ri
            src, src_idx = np.unique(h[selected], return_inverse=True)
            dst, dst_idx = np.unique(t[selected], return_inverse=True)
            for direction in (0, 1):
                ss, dd, si, di = ((src, dst, src_idx, dst_idx) if direction == 0
                                  else (dst, src, dst_idx, src_idx))
                src_map = np.zeros(n_entities, dtype=np.int32)
                dst_map = np.zeros(n_entities, dtype=np.int32)
                src_map[ss] = np.arange(1, len(ss) + 1, dtype=np.int32)
                dst_map[dd] = np.arange(1, len(dd) + 1, dtype=np.int32)
                packed = np.zeros((len(ss) + 1, (len(dd) + 8) // 8), dtype=np.uint8)
                # OR-at handles duplicates and several targets in the same byte.
                np.bitwise_or.at(packed, (si + 1, (di + 1) // 8),
                                 (1 << ((di + 1) % 8)).astype(np.uint8))
                table = tuple(torch.from_numpy(x).to(self.device) for x in (src_map, dst_map, packed))
                self.tables[ri + direction * n_rel_base] = table
                self.nbytes += sum(x.numel() * x.element_size() for x in table)

    @torch.no_grad()
    def mask(self, source, relation_id, negatives):
        """Return BxK membership for one directed relation and shared negatives.

        The same candidate may be a positive for one source and a valid
        negative for another. Every sampled occurrence of a known positive
        is masked, including copies of the current training target.
        """
        if source.ndim != 1 or negatives.ndim != 1:
            raise ValueError("Expected one-dimensional sources and shared negatives")
        src_map, dst_map, packed = self.tables[int(relation_id)]
        rows = src_map[source].long()
        columns = dst_map[negatives].long()
        values = packed[rows[:, None], (columns // 8)[None, :]]
        return ((values >> (columns % 8).to(torch.uint8)[None, :]) & 1).bool()


def mask_negative_logits(logits, negative_mask):
    """Leave column 0 (the training target) intact; remove masked negatives.

    -inf is safe for single-target cross entropy, even when all negatives
    are masked. This helper is not a KL-distillation implementation.
    """
    if negative_mask.dtype != torch.bool or negative_mask.shape != logits[:, 1:].shape:
        raise ValueError("Expected a boolean mask matching the negative logits")
    return torch.cat((logits[:, :1], logits[:, 1:].masked_fill(negative_mask, -torch.inf)), dim=1)
