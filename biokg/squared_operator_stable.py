"""H37N: numerical normalization repair, unchanged storage/loss/hyperparameters.

The original H37 source and failed attempt remain immutable. The finite
runner adapter below explicitly reuses its audited harness with these model
factories and distinct output/receipt identity; no originals are edited.
"""

import copy
from pathlib import Path

import torch

from biokg import squared_operator as original
from biokg.masked_neighborhood import sha, state_digest


SPEC = dict(original.SPEC, protocol='H37N', normalization='fp32_tiny_clamp_before_sqrt')
SOURCE_FILES = (*original.SOURCE_FILES, 'biokg/squared_operator_stable.py',
    'biokg/run_h37_stable_vast.py', 'biokg/test_squared_stable.py',
    'biokg/H37_NUMERICS.md', 'biokg/scripts/run_h37_stable.sh',
    'biokg/scripts/h37_stable.supervisor.conf')


def stable_cnorm(z, eps=1e-8):
    # The floor is the dtype's smallest NORMAL squared magnitude, not a new
    # learned threshold. sqrt(tiny) << one fp32 ULP of the existing eps.
    norm = z.abs().pow(2).sum(-1, keepdim=True).clamp_min(torch.finfo(z.real.dtype).tiny).sqrt()
    return z/(norm+eps)


class StableSquaredOperator(original.SquaredOperator):
    def embed(self, idx):
        return stable_cnorm(self.rows(idx))

    def hop(self, z, relation):
        blocks = z.reshape(len(z), 36, 4)
        hopped = torch.einsum('bkij,bkj->bki', self.H[relation], blocks).reshape_as(z)
        return stable_cnorm(hopped)


def make_pair(n, nr, device):
    linear = StableSquaredOperator(n, nr, 'linear', device)
    squared = copy.deepcopy(linear)
    squared.arm, squared.sparse_grad = 'squared', False
    assert state_digest(linear.state_dict()) == state_digest(squared.state_dict())
    assert all(a.data_ptr() != b.data_ptr() for a, b in zip(linear.parameters(), squared.parameters()))
    return [linear, squared]


def snapshot(model, opts, out, steps):
    for opt in opts:
        opt.zero_grad(set_to_none=True)
    ck = dict(model_type='H37NSingle', spec=SPEC, arm=model.arm, steps=steps,
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
    if ck['model_type'] != 'H37NSingle' or ck['spec'] != SPEC or ck['arm'] not in original.NAMES:
        raise ValueError('Not an own H37N fixed-recipe checkpoint')
    if Path(path).stem != ck['arm']:
        raise ValueError('Checkpoint filename/arm mismatch')
    model = StableSquaredOperator(ck['n_entities'], ck['n_relations'], ck['arm'], device)
    model.load_state_dict(ck['model'], strict=True)
    return model.eval().requires_grad_(False)


def configured_runner():
    """Explicit in-process adapter; no change to the pinned H37 source files.

    Old objective/update and all candidate/CI logic are reused unchanged. This
    must run in a fresh process; never mix original and repaired experiments.
    """
    from biokg import squared_operator_run as runner
    if runner.SPEC['protocol'] != 'H37':
        raise RuntimeError('H37N adapter must configure a fresh runner once')
    runner.SPEC = SPEC
    runner.SOURCE_FILES = SOURCE_FILES
    runner.make_pair = make_pair
    runner.restore = restore
    runner.snapshot = snapshot
    old_summary = runner.summary
    def summary(*args, **kwargs):
        result = old_summary(*args, **kwargs)
        result['protocol'] = SPEC['protocol']
        result['normalization_repair'] = SPEC['normalization']
        return result
    runner.summary = summary
    return runner
