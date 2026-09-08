"""MN1 core: TRAIN-only context assistance; ordinary single-model inference."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

from resonate import cnorm
from resonate_wiki import SparseTableResonatE, clip_grad_norm_
from rowadagrad import RowAdagrad


SPEC = dict(protocol='MN1', base_steps=50000, continuation_steps=12500, batch=2048, negatives=4096,
    seed=0, stream_seed=36850, context_seed=36851, probe_seed=36852, evaluation_seed=36853,
    bootstrap_seed=36854, probe_rows=1024, probe_every=5000, eval_chunk=256, k=12, block_size=4,
    context_count=8, context_fraction=.25, context_loss_weight=.25, trajectory=.1,
    base_adam_lr=.005, base_table_lr=.3, continuation_adam_lr=.001, continuation_table_lr=.06)
SOURCE_FILES = ('resonate.py', 'resonate_wiki.py', 'rowadagrad.py',
    'biokg/masked_neighborhood.py', 'biokg/masked_neighborhood_run.py',
    'biokg/test_masked_neighborhood.py', 'biokg/run_masked_neighborhood_vast.py',
    'biokg/MASKED_NEIGHBORHOOD.md', 'biokg/scripts/run_masked_neighborhood.sh',
    'biokg/scripts/masked_neighborhood.supervisor.conf')
FIT_SHA = 'aa13b00c4f4e5f93f18a2af02e895f85375fedc873d09a25778090e97913796f'
HOLDOUT_SHA = '783a55949f188b1ac09194df3bf03ed5a371284e228a17b7ca8282003b4e7e2f'


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def verify(hashes):
    for path, expected in hashes.items():
        if sha(path) != expected:
            raise ValueError(f'Hash mismatch: {path}')


def state_digest(value):
    digest = hashlib.sha256()
    def add(node):
        if isinstance(node, torch.Tensor):
            arr = node.detach().resolve_conj().cpu().contiguous().numpy()
            digest.update(str((arr.shape, arr.dtype)).encode())
            digest.update(arr.tobytes())
        elif isinstance(node, dict):
            for key in sorted(node, key=str):
                digest.update(str(key).encode())
                add(node[key])
        elif isinstance(node, (list, tuple)):
            for item in node:
                add(item)
        else:
            digest.update(repr(node).encode())
    add(value)
    return digest.hexdigest()


def check_access(path, data_dir, out, phase):
    path, data_dir, out = [Path(p).resolve() for p in (path, data_dir, out)]
    if path.is_relative_to(data_dir):
        allowed = {'manifest.json', 'fit_train.npz'}
        if phase in ('evaluate', 'audit'):
            allowed.add('holdout_train.npz')
        name = str(path.relative_to(data_dir))
        if name not in allowed:
            raise PermissionError(f'MN1 {phase} forbids {name}')
        return name
    if path.suffix in ('.pt', '.pth', '.pkl', '.npy', '.npz', '.safetensors', '.bin') and not path.is_relative_to(out):
        raise PermissionError(f'MN1 forbids foreign artifact: {path}')
    return None


def install_guard(data_dir, out, phase, opened):
    def guard(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes)):
            path = args[0].decode() if isinstance(args[0], bytes) else args[0]
            name = check_access(path, data_dir, out, phase)
            if name:
                opened.add(name)
    sys.addaudithook(guard)


def pair_bucket(h, t, n):
    key = np.minimum(h, t).astype(np.uint64)*np.uint64(n)+np.maximum(h, t).astype(np.uint64)
    x = key+np.uint64(36840)+np.uint64(0x9e3779b97f4a7c15)
    x = (x ^ (x >> np.uint64(30)))*np.uint64(0xbf58476d1ce4e5b9)
    x = (x ^ (x >> np.uint64(27)))*np.uint64(0x94d049bb133111eb)
    return ((x ^ (x >> np.uint64(31))) % np.uint64(100)).astype(np.uint8)


def load_part(data_dir, name, manifest):
    expected = FIT_SHA if name == 'fit' else HOLDOUT_SHA
    path = Path(data_dir)/(name+'_train.npz')
    assert manifest['artifact_sha256'][path.name] == expected
    verify({str(path): expected})
    with np.load(path, allow_pickle=False) as data:
        assert set(data.files) == {'head', 'relation', 'tail', 'head_type_id', 'tail_type_id', 'row_id'}
        part = {k: data[k] for k in data.files}
    offsets = np.array([manifest['offsets'][t] for t in manifest['entity_types']], np.int64)
    counts = np.array([manifest['counts'][t] for t in manifest['entity_types']], np.int64)
    for side in ('head', 'tail'):
        types = part[side+'_type_id']
        assert ((types >= 0) & (types < len(counts))).all()
        assert ((part[side] >= 0) & (part[side] < counts[types])).all()
        part[side] = part[side]+offsets[types]
    role = pair_bucket(part['head'], part['tail'], int(counts.sum())) >= 95
    assert (role if name == 'holdout' else ~role).all()
    assert len(part['head']) == manifest['summary']['roles'][name]['rows']
    return part


class Stream:
    def __init__(self, fit, manifest, seed):
        self.fit, self.manifest = fit, manifest
        self.n_rel = int(fit['relation'].max())+1
        self.rows = [np.flatnonzero(fit['relation'] == r) for r in range(self.n_rel)]
        if any(not len(rows) for rows in self.rows):
            raise ValueError('All original relations need fit coverage')
        self.weights = np.array([len(rows) for rows in self.rows], np.float64)
        self.weights /= self.weights.sum()
        self.ranges, self.drug_relations = {}, set()
        names = manifest['entity_types']
        for r, rows in enumerate(self.rows):
            types = [int(fit[side+'_type_id'][rows[0]]) for side in ('head', 'tail')]
            for side, tid in zip(('head', 'tail'), types):
                assert (fit[side+'_type_id'][rows] == tid).all()
            for d, tid in enumerate((types[1], types[0])):
                off = manifest['offsets'][names[tid]]
                self.ranges[r+d*self.n_rel] = (off, off+manifest['counts'][names[tid]])
            if all(names[tid] == 'drug' for tid in types):
                self.drug_relations.update((r, r+self.n_rel))
        self.rng = np.random.default_rng(seed)
        self.digest = hashlib.sha256()

    def sample(self, batch, negatives, device):
        r = int(self.rng.choice(self.n_rel, p=self.weights))
        ids = self.rows[r][self.rng.integers(len(self.rows[r]), size=batch)]
        forward = self.rng.random() < .5
        s, t = (self.fit['head'][ids], self.fit['tail'][ids]) if forward else (self.fit['tail'][ids], self.fit['head'][ids])
        directed = r if forward else r+self.n_rel
        neg = self.rng.integers(*self.ranges[directed], size=negatives)
        rr = np.full(batch, directed, np.int64)
        for array in (s, rr, t, neg):
            self.digest.update(array.tobytes())
        return tuple(torch.as_tensor(a, device=device) for a in (s, rr, t, neg))


class NeighborIndex:
    def __init__(self, fit, n_entities, n_rel):
        self.n = n_entities
        self.keys, self.gpu = {}, {}
        for rel in range(n_rel):
            rows = fit['relation'] == rel
            h, t = fit['head'][rows], fit['tail'][rows]
            self.keys[rel] = np.unique(h*n_entities+t)
            self.keys[rel+n_rel] = np.unique(t*n_entities+h)

    def known(self, source, candidate, directed):
        keys = self.keys[directed]
        query = np.asarray(source)*self.n+np.asarray(candidate)
        at = np.searchsorted(keys, query)
        return (at < len(keys)) & (keys[np.minimum(at, len(keys)-1)] == query)

    def sample_without_target(self, source, target, directed, count, generator):
        dev = source.device
        cache_key = directed, str(dev)
        if cache_key not in self.gpu:
            keys = self.keys[directed]
            lengths = np.bincount(keys//self.n, minlength=self.n)
            ptr = np.r_[0, np.cumsum(lengths)]
            self.gpu[cache_key] = tuple(torch.as_tensor(a, device=dev) for a in (keys, ptr, keys%self.n))
        keys, ptr, targets = self.gpu[cache_key]
        position = torch.searchsorted(keys, source*self.n+target)
        if not bool(((position < len(keys)) & (keys[position.clamp(max=len(keys)-1)] == source*self.n+target)).all()):
            raise ValueError('Masked reconstruction target must be a fit neighbor')
        start, length = ptr[source], ptr[source+1]-ptr[source]
        remaining = length-1
        random = torch.rand((len(source), count), device=dev, generator=generator)
        draw = (random*remaining.clamp(min=1)[:, None]).long()
        draw += (draw >= (position-start)[:, None]).long()
        valid = remaining > 0
        loc = torch.where(valid[:, None], start[:, None]+draw, torch.zeros_like(draw))
        selected = targets[loc]
        if bool(((selected == target[:, None]) & valid[:, None]).any()):
            raise RuntimeError('Hidden target leaked into sampled context')
        return selected, valid


def logits_from_query(model, q, positive, negatives):
    return torch.cat([(q*positive.conj()).sum(-1, keepdim=True).real,
                      (q @ negatives.conj().T).real], dim=1)*model.log_tau.exp()


def objective(model, batch, context=None, valid=None, fraction=.25, weight=.25):
    source, relation, positive, negatives = batch
    q = model.hop(model.embed(source), relation)
    ep, en = model.rows(positive), model.rows(negatives)
    logits = logits_from_query(model, q, ep, en)
    ce = F.cross_entropy(logits, torch.zeros(len(source), dtype=torch.long, device=source.device), reduction='none')
    traj = (q-ep).abs().square().sum(-1).mean()
    if context is None or not bool(valid.any()) or weight == 0 or fraction == 0:
        return ce.mean()+.1*traj, dict(ce=float(ce.detach().mean()), trajectory=float(traj.detach()), context_rows=0)
    c = cnorm(cnorm(model.rows(context)).mean(1))
    assisted = cnorm((1-fraction)*q+fraction*c)
    assisted = torch.where(valid[:, None], assisted, q)
    ctx = F.cross_entropy(logits_from_query(model, assisted, ep, en),
                          torch.zeros(len(source), dtype=torch.long, device=source.device), reduction='none')
    combined = torch.where(valid, (1-weight)*ce+weight*ctx, ce).mean()
    return combined+.1*traj, dict(ce=float(ce.detach().mean()), context_ce=float(ctx.detach().mean()),
                                  trajectory=float(traj.detach()), context_rows=int(valid.sum()))


def make_model(n, n_rel, device='cuda'):
    return SparseTableResonatE(n, n_rel, k=12, block_size=4, sparse_grad=True, device=device)


def make_optimizers(model, adam_lr=.005, table_lr=.3):
    return [torch.optim.Adam(model.other_params(), lr=adam_lr), RowAdagrad(model.table_params(), lr=table_lr)]


def update(model, opts, batch, context=None, valid=None):
    for opt in opts:
        opt.zero_grad(set_to_none=True)
    loss, parts = objective(model, batch, context, valid)
    if not bool(torch.isfinite(loss)):
        raise RuntimeError('Nonfinite training loss')
    loss.backward()
    norm = clip_grad_norm_(list(model.parameters()), 1.)
    if not bool(torch.isfinite(norm)):
        raise RuntimeError('Nonfinite training gradients')
    for opt in opts:
        opt.step()
    return float(loss.detach()), parts


def score_candidates(model, source, relation, candidates):
    q = model.hop(model.embed(source), relation)
    return (torch.einsum('bm,bcm->bc', q, model.rows(candidates).conj()).real)*model.log_tau.exp()


def ranks(scores):
    if not bool(torch.isfinite(scores).all()):
        raise RuntimeError('Nonfinite evaluation scores')
    return 1+((scores[:, 1:] > scores[:, :1]).sum(1)+(scores[:, 1:] >= scores[:, :1]).sum(1)).float()/2


if __name__ == '__main__':
    from biokg.masked_neighborhood_run import main
    main()
