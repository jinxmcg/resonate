# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["torch==2.11.0+cu128", "numpy==2.5.2", "scipy==1.18.1"]
# [tool.uv.sources]
# torch = { index = "pytorch-cu128" }
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
# ///
"""Bounded TRAIN-only replay of H37's nonfinite-gradient stop; no new recipe."""

import argparse
import json
from pathlib import Path
import time
import traceback
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from biokg.masked_neighborhood import (
    Stream, install_guard, load_part, make_optimizers, sha, state_digest, write_json,
)
from biokg.squared_operator import SPEC, make_pair, objective
from biokg.squared_operator_run import verify_receipt
from resonate_wiki import clip_grad_norm_


def gradient_stats(model):
    result = {}
    for name, param in model.named_parameters():
        g = param.grad
        if g is None:
            result[name] = dict(present=False)
            continue
        value = g.coalesce().values() if g.is_sparse else g
        result[name] = dict(present=True, finite=bool(torch.isfinite(value).all()),
            nan_count=int(torch.isnan(value).sum()), inf_count=int(torch.isinf(value).sum()),
            norm=float(value.norm()) if bool(torch.isfinite(value.norm())) else 'nonfinite')
    return result


def run(root, data, out, original):
    out.mkdir(parents=True, exist_ok=False)
    opened = set()
    install_guard(data, out, 'train', opened)
    receipt = json.loads((original/'prerun.json').read_text())
    verify_receipt(receipt, root, data)
    write_json(out/'prerun.json', dict(original_receipt_sha256=sha(original/'prerun.json'),
        diagnostic_source_sha256=sha(Path(__file__)), max_steps=5000,
        no_recipe_changes=True, no_holdout_evaluation=True))
    manifest = json.loads((data/'manifest.json').read_text())
    fit = load_part(data, 'fit', manifest)
    stream = Stream(fit, manifest, SPEC['stream_seed'])
    torch.manual_seed(SPEC['seed'])
    models = make_pair(sum(manifest['counts'].values()), 2*stream.n_rel, 'cuda')
    opts = [make_optimizers(m, SPEC['adam_lr'], SPEC['table_lr']) for m in models]
    scheds = [[torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=SPEC['steps']) for o in oo] for oo in opts]
    initial = state_digest(models[0].state_dict())
    start = time.monotonic()
    for step in range(1, 5001):
        batch = stream.sample(SPEC['batch'], SPEC['negatives'], 'cuda')
        support = stream.ranges[int(batch[1][0])]
        for m, oo, sched in zip(models, opts, scheds):
            for opt in oo:
                opt.zero_grad(set_to_none=True)
            loss, parts = objective(m, batch, support)
            loss.backward()
            # Check without clipping first, so multiplying by a nonfinite clip
            # coefficient cannot hide which parameter gradient first failed.
            stats = gradient_stats(m)
            if any(not v.get('finite', True) or v.get('norm') == 'nonfinite' for v in stats.values()):
                with torch.no_grad():
                    x = m.rows(batch[0])
                    embedded = m.embed(batch[0])
                    block = embedded.reshape(len(embedded), 36, 4)
                    z = torch.einsum('bkij,bkj->bki', m.H[batch[1]], block).reshape_as(embedded)
                    squared_norms = x.abs().square().sum(-1)
                    z_norms = z.abs().square().sum(-1)
                    q = m.hop(embedded, batch[1])
                    details = dict(source_zero_norm_rows=int((squared_norms == 0).sum()),
                        source_min_squared_norm=float(squared_norms.min()),
                        prehop_finite=bool(torch.isfinite(z).all()), hop_zero_norm_rows=int((z_norms == 0).sum()),
                        query_finite=bool(torch.isfinite(q).all()), tau=float(m.log_tau.exp()),
                        all_weights_finite=all(bool(torch.isfinite(p).all()) for p in m.parameters()))
                torch.save(dict(arm=m.arm, step=step, model=m.state_dict(), optimizers=[o.state_dict() for o in oo],
                    batch=[x.cpu() for x in batch], typed_range=support, spec=SPEC,
                    diagnostic_only_not_an_endpoint=True), out/'failure_state.pt')
                for opt in oo:
                    opt.zero_grad(set_to_none=True)
                anomaly = 'not reproduced'
                try:
                    with torch.autograd.detect_anomaly(check_nan=True):
                        again, _ = objective(m, batch, support)
                        again.backward()
                except RuntimeError:
                    anomaly = traceback.format_exc()
                verify_receipt(receipt, root, data)
                assert sorted(opened) == ['fit_train.npz', 'manifest.json']
                result = dict(reproduced=True, failure_step=step, arm=m.arm, loss=float(loss.detach()),
                    loss_parts=parts, gradients_before_clipping=stats, tensor_diagnostics=details,
                    anomaly=anomaly, adam_lr=float(oo[0].param_groups[0]['lr']),
                    table_lr=float(oo[1].param_groups[0]['lr']), initial_model=initial,
                    source_files_unchanged=True, dataset_files_opened=sorted(opened),
                    heldout_opened=False, official_validation_loaded=False, official_test_loaded=False,
                    recipe_changed=False, failure_state_sha256=sha(out/'failure_state.pt'),
                    seconds=time.monotonic()-start)
                write_json(out/'summary.json', result)
                print(json.dumps(result, allow_nan=False), flush=True)
                return
            norm = clip_grad_norm_(list(m.parameters()), 1.)
            if not bool(torch.isfinite(norm)):
                raise RuntimeError('Unexpected aggregate-only norm overflow: stop replay')
            for opt in oo:
                opt.step()
            for sc in sched:
                sc.step()
        if step % 1000 == 0:
            print(json.dumps(dict(stage='failure_replay', step=step, seconds=time.monotonic()-start)), flush=True)
    raise RuntimeError('Failure not reproduced within fixed diagnostic cap')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='biokg/data')
    parser.add_argument('--original', default='biokg/results/h37/s0')
    parser.add_argument('--out', default='biokg/results/h37/failure_replay')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available() or '5090' not in torch.cuda.get_device_name():
        raise RuntimeError('Expected authorized RTX5090')
    run(Path(__file__).resolve().parents[1], Path(args.data).resolve(), Path(args.out).resolve(), Path(args.original).resolve())


if __name__ == '__main__':
    main()
