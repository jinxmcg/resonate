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
"""H37N finite numeric-repair runner, preserving the original failed attempt."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


def reproduce(root):
    import torch
    from biokg.masked_neighborhood import sha, state_digest, write_json
    from biokg.squared_operator import SquaredOperator, objective
    from biokg.squared_operator_stable import StableSquaredOperator
    from biokg.squared_operator_run import verify_sources
    original = root/'biokg/results/h37/failure_replay'
    receipt = json.loads((original/'summary.json').read_text())
    assert receipt['reproduced'] and receipt['failure_step'] == 1426 and receipt['recipe_changed'] is False
    assert sha(original/'failure_state.pt') == receipt['failure_state_sha256']
    verify_sources(json.loads((root/'biokg/results/h37/pilot.json').read_text())['receipt'], root)
    ck = torch.load(original/'failure_state.pt', map_location='cuda', weights_only=False)
    assert ck['diagnostic_only_not_an_endpoint'] and ck['step'] == 1426 and ck['arm'] == 'squared'
    batch = tuple(x.to('cuda') for x in ck['batch'])
    models = [cls(len(ck['model']['E_real']), len(ck['model']['H']), 'squared', 'cuda')
              for cls in (SquaredOperator, StableSquaredOperator)]
    for m in models:
        m.load_state_dict(ck['model'])
    losses = [objective(m, batch, ck['typed_range'])[0] for m in models]
    queries = [m.hop(m.embed(batch[0]), batch[1]) for m in models]
    assert torch.equal(queries[0], queries[1]) and torch.equal(losses[0], losses[1])
    for loss in losses:
        loss.backward()
    assert not torch.isfinite(models[0].E_real.grad).all()
    assert all(torch.isfinite(p.grad).all() and torch.isfinite(p.grad.norm()) for p in models[1].parameters())
    torch.testing.assert_close(models[0].H.grad, models[1].H.grad, atol=0, rtol=0)
    torch.testing.assert_close(models[0].log_tau.grad, models[1].log_tau.grad, atol=0, rtol=0)
    assert all(state_digest(m.state_dict()) == state_digest(ck['model']) for m in models)
    out = root/'biokg/results/h37_stable/reproducer.json'
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result = dict(passed=True, failure_step=1426, original_failure_state_sha256=receipt['failure_state_sha256'],
        identical_forward_query_and_loss=True, original_gradients_nonfinite=True,
        repaired_gradients_and_norms_finite=True, unchanged_H_and_tau_gradients=True,
        model_weights_unchanged=True, optimizer_updates=0, holdout_loaded=False,
        official_validation_loaded=False, official_test_loaded=False,
        repaired_entity_gradient_norm=float(models[1].E_real.grad.norm()),
        repaired_source_sha256=sha(root/'biokg/squared_operator_stable.py'),
        runner_sha256=sha(Path(__file__)))
    write_json(out, result)
    print(json.dumps(dict(stage='reproducer_complete', **result)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_mutually_exclusive_group()
    for flag in ('tests', 'reproducer', 'pilot', 'evaluate', 'audit'):
        phases.add_argument('--'+flag, action='store_true')
    args = parser.parse_args()
    import torch
    import numpy as np
    if not torch.cuda.is_available() or '5090' not in torch.cuda.get_device_name() or torch.version.cuda != '12.8':
        raise RuntimeError('Expected authorized RTX5090 CUDA12.8 environment')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print(json.dumps(dict(stage='H37N_environment', torch=torch.__version__, numpy=np.__version__,
        cuda=torch.version.cuda, gpu=torch.cuda.get_device_name())), flush=True)
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    if args.tests:
        raise SystemExit(subprocess.run([sys.executable, '-m', 'unittest', 'biokg.test_squared_stable',
            'biokg.test_squared_operator', 'biokg.test_masked_neighborhood'], cwd=root).returncode)
    if args.reproducer:
        reproduce(root)
        return
    from biokg.squared_operator_stable import configured_runner
    from biokg.masked_neighborhood import sha
    record = json.loads((root/'biokg/results/h37_stable/reproducer.json').read_text())
    assert record['passed']
    assert record['repaired_source_sha256'] == sha(root/'biokg/squared_operator_stable.py')
    assert record['runner_sha256'] == sha(Path(__file__))
    sys.argv = [sys.argv[0], '--out', str(root/'biokg/results/h37_stable/s0'),
                '--pilot-path', str(root/'biokg/results/h37_stable/pilot.json')]
    for flag in ('pilot', 'evaluate', 'audit'):
        if getattr(args, flag):
            sys.argv.append('--'+flag)
    configured_runner().main()


if __name__ == '__main__':
    main()
