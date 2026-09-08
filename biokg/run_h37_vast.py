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
"""H37 isolated CUDA runner; no automatic data or trained-weight download."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_mutually_exclusive_group()
    for flag in ('check-environment', 'tests', 'pilot', 'evaluate', 'audit'):
        phases.add_argument('--'+flag, action='store_true')
    args = parser.parse_args()
    import torch
    import numpy as np
    if not torch.cuda.is_available() or '5090' not in torch.cuda.get_device_name() or torch.version.cuda != '12.8':
        raise RuntimeError('Expected RTX5090 with pinned CUDA12.8 wheel')
    torch.ones(1, device='cuda').sum().item()
    print(json.dumps(dict(stage='environment_verified', python=sys.version, torch=torch.__version__,
        numpy=np.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name())), flush=True)
    if args.check_environment:
        return
    root = Path(__file__).resolve().parents[1]
    if args.tests:
        command = [sys.executable, '-m', 'unittest', 'biokg.test_squared_operator', 'biokg.test_masked_neighborhood']
    else:
        command = [sys.executable, '-m', 'biokg.squared_operator_run']
        for flag in ('pilot', 'evaluate', 'audit'):
            if getattr(args, flag):
                command.append('--'+flag)
    raise SystemExit(subprocess.run(command, cwd=root).returncode)


if __name__ == '__main__':
    main()
