"""H37 model/loss: unchanged 4x4 operators, exact typed squared likelihood."""

import copy
import math
from pathlib import Path

import torch

from resonate_wiki import SparseTableResonatE, clip_grad_norm_
from biokg.masked_neighborhood import make_optimizers, objective as original_objective, sha, state_digest


SPEC = dict(protocol='H37', steps=50000, seed=0, stream_seed=0, batch=2048,
    negatives=4096, epsilon=1e-6, k=12, block_size=4, trajectory=.1,
    adam_lr=.005, table_lr=.3, probe_seed=36852, probe_rows=1024,
    probe_every=5000, evaluation_seed=36853, bootstrap_seed=36854,
    eval_chunk=256, cpu_threads=4, pilot_seed=37055,
    pilot_max_pair_seconds=.020, pilot_max_allocated_bytes=8*1024**3,
    train_max_seconds=1800)
NAMES = ('linear', 'squared')
SOURCE_FILES = ('resonate.py', 'resonate_wiki.py', 'rowadagrad.py',
    'biokg/masked_neighborhood.py', 'biokg/masked_neighborhood_run.py',
    'biokg/test_masked_neighborhood.py', 'biokg/squared_operator.py',
    'biokg/squared_operator_run.py', 'biokg/test_squared_operator.py',
    'biokg/H37.md', 'biokg/run_h37_vast.py', 'biokg/scripts/run_h37.sh',
    'biokg/scripts/h37.supervisor.conf')


def real_view(z):
    return torch.view_as_real(z).reshape(*z.shape[:-1], 2*z.shape[-1])


def squared_weight(dot, log_tau, epsilon=SPEC['epsilon']):
    return epsilon+(log_tau.exp()*dot).square()


def gram_normalizer(q, candidates, log_tau, epsilon=SPEC['epsilon']):
    """Exact sum of epsilon + (tau * real complex inner product)^2.

    The full Gram matrix remains attached to autograd: every eligible entity
    participates, including those not present in the positive batch.
    """
    if candidates.ndim != 2 or q.ndim != 2 or not len(candidates):
        raise ValueError('Expected nonempty rank-two query and candidate arrays')
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('A positive fixed smoothing floor is required')
    qr, vr = real_view(q), real_view(candidates)
    gram = vr.T @ vr
    return epsilon*len(candidates)+log_tau.exp().square()*((qr @ gram)*qr).sum(-1)


class SquaredOperator(SparseTableResonatE):
    def __init__(self, n_entities, n_relations, arm, device='cpu'):
        if arm not in NAMES:
            raise ValueError('Unknown H37 arm')
        super().__init__(n_entities, n_relations, k=12, block_size=4,
                         sparse_grad=(arm == 'linear'), device=device)
        self.arm = arm

    def candidate_scores(self, source, relation, candidates):
        q = self.hop(self.embed(source), relation)
        dot = torch.einsum('bm,bcm->bc', q, self.rows(candidates).conj()).real
        return dot*self.log_tau.exp() if self.arm == 'linear' else squared_weight(dot, self.log_tau)


def make_pair(n, nr, device):
    linear = SquaredOperator(n, nr, 'linear', device)
    squared = copy.deepcopy(linear)
    squared.arm, squared.sparse_grad = 'squared', False
    assert state_digest(linear.state_dict()) == state_digest(squared.state_dict())
    assert all(a.data_ptr() != b.data_ptr() for a, b in zip(linear.parameters(), squared.parameters()))
    return [linear, squared]


def objective(model, batch, typed_range):
    if model.arm == 'linear':
        return original_objective(model, batch)
    source, relation, positive, _ = batch
    lo, hi = typed_range
    if not (0 <= lo < hi <= model.n_entities):
        raise ValueError('Invalid eligible entity range')
    if not bool(((positive >= lo) & (positive < hi)).all()):
        raise ValueError('Positive outside eligible type support')
    if not bool((relation == relation[0]).all()):
        raise ValueError('H37 batches must share a directed relation')
    q = model.hop(model.embed(source), relation)
    ep = model.rows(positive)
    dot = (q*ep.conj()).sum(-1).real
    numerator = squared_weight(dot, model.log_tau)
    denominator = gram_normalizer(q, model.table()[lo:hi], model.log_tau)
    if not bool(torch.isfinite(denominator).all() & (denominator > 0).all()):
        raise RuntimeError('Invalid exact squared normalizer')
    nll = (denominator.log()-numerator.log()).mean()
    trajectory = (q-ep).abs().square().sum(-1).mean()
    return nll+SPEC['trajectory']*trajectory, dict(nll=float(nll.detach()),
        trajectory=float(trajectory.detach()), eligible_entities=hi-lo,
        positive_negative_dot_fraction=float((dot.detach() < 0).float().mean()),
        mean_epsilon_denominator_share=float((SPEC['epsilon']*(hi-lo)/denominator.detach()).mean()),
        tau=float(model.log_tau.detach().exp()))


def update(model, opts, batch, typed_range):
    for opt in opts:
        opt.zero_grad(set_to_none=True)
    loss, parts = objective(model, batch, typed_range)
    if not bool(torch.isfinite(loss)):
        raise RuntimeError('Nonfinite H37 training loss')
    loss.backward()
    norm = clip_grad_norm_(list(model.parameters()), 1.)
    if not bool(torch.isfinite(norm)):
        raise RuntimeError('Nonfinite H37 gradients')
    for opt in opts:
        opt.step()
    return float(loss.detach()), parts


def snapshot(model, opts, out, steps):
    for opt in opts:
        opt.zero_grad(set_to_none=True)
    ck = dict(model_type='H37Single', spec=SPEC, arm=model.arm, steps=steps,
        n_entities=model.n_entities, n_relations=model.n_relations,
        model=model.state_dict(), optimizers=[o.state_dict() for o in opts])
    path = Path(out)/(model.arm+'.pt')
    if path.exists():
        raise FileExistsError(path)
    torch.save(ck, path)
    return dict(model=state_digest(ck['model']), optimizer=state_digest(ck['optimizers']),
                checkpoint_sha256=sha(path), steps=steps, arm=model.arm)


def restore(path, device='cpu'):
    ck = torch.load(path, map_location=device, weights_only=False)
    if ck['model_type'] != 'H37Single' or ck['spec'] != SPEC or ck['arm'] not in NAMES:
        raise ValueError('Not an own H37 fixed-recipe checkpoint')
    if Path(path).stem != ck['arm']:
        raise ValueError('Checkpoint filename/arm mismatch')
    model = SquaredOperator(ck['n_entities'], ck['n_relations'], ck['arm'], device)
    model.load_state_dict(ck['model'], strict=True)
    return model.eval().requires_grad_(False)
