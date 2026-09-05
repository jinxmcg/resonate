"""Fixed head/tail sampling probabilities from TRAIN triples only.

Fanout is an ambiguity proxy, not a measured validation difficulty. The
policy is computed once, has no RNG or fitted parameters, and is unused at
inference. Relation sampling weights and within-relation edge sampling are
not changed here.
"""
import hashlib

import numpy as np


def train_fanout_policy(head, relation, tail, n_entities, n_rel_base):
    """Prefer the direction with greater edge-weighted distinct fanout.

    For unique TRAIN edges of a relation, a uniformly drawn edge encounters
    mean fanout sum(degree**2)/edge_count. Tail prediction uses head degrees;
    head prediction uses tail degrees. Square-root temper the two fanouts,
    normalize, and cap the sampling ratio at 2:1 in either direction.
    Empty relations get 50/50. Duplicate edges do not change the policy.
    """
    raw = [np.asarray(x) for x in (head, relation, tail)]
    if any(x.ndim != 1 or len(x) != len(raw[0]) for x in raw):
        raise ValueError("Training head/relation/tail must be equally sized vectors")
    if any(x.dtype.kind not in "iu" for x in raw):
        raise ValueError("Training IDs must be integers")
    if n_entities < 1 or n_rel_base < 1:
        raise ValueError("Entity and relation counts must be positive")
    if any(np.any((x < 0) | (x >= upper)) for x, upper in
           zip(raw, (n_entities, n_rel_base, n_entities))):
        raise ValueError("Training ID outside declared entity/relation range")
    h, r, t = arrays = [np.asarray(x, dtype=np.int64) for x in raw]
    digest = hashlib.sha256(f"{n_entities}:{n_rel_base}".encode())
    for x in arrays:
        digest.update(np.ascontiguousarray(x).view(np.uint8))
    probabilities, rows = [], []
    for ri in range(n_rel_base):
        mask = r == ri
        pairs = np.unique(np.column_stack((h[mask], t[mask])), axis=0)
        edges = len(pairs)
        if edges:
            _, hc = np.unique(pairs[:, 0], return_counts=True)
            _, tc = np.unique(pairs[:, 1], return_counts=True)
            tail_fanout = float(np.square(hc.astype(np.float64)).sum() / edges)
            head_fanout = float(np.square(tc.astype(np.float64)).sum() / edges)
            a, b = np.sqrt(tail_fanout), np.sqrt(head_fanout)
            p_tail = float(np.clip(a / (a + b), 1 / 3, 2 / 3))
        else:
            tail_fanout = head_fanout = 0.
            p_tail = .5
        probabilities.append(p_tail)
        rows.append(dict(relation=ri, unique_train_edges=edges,
                         tail_fanout=tail_fanout, head_fanout=head_fanout,
                         p_tail=p_tail, p_head=1 - p_tail))
    return dict(mode="train-fanout", train_sha256=digest.hexdigest(),
                formula="clip(sqrt(tail_fanout)/(sqrt(tail_fanout)+sqrt(head_fanout)), 1/3, 2/3)",
                tail_probabilities=probabilities, relations=rows)
