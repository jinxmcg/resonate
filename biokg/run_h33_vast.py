# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "torch==2.11.0+cu128",
#   "numpy==2.5.2",
#   "scipy==1.18.1",
#   "ogb==1.3.6",
#   "pandas==3.0.5",
# ]
# [tool.uv.sources]
# torch = { index = "pytorch-cu128" }
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
# ///
"""Install the declared CUDA stack with uv and run the fixed H33 experiment.

Usage: uv run --script biokg/run_h33_vast.py
Only official TRAIN/VALID files are needed; never download or load test.pt.
"""
import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-environment", action="store_true",
                        help="install/check dependencies and CUDA without loading data or starting runs")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    import torch
    if not torch.cuda.is_available() or torch.version.cuda != "12.8":
        raise RuntimeError("The pinned CUDA 12.8 GPU environment is not available")
    torch.ones(1, device="cuda").sum().item()
    if args.check_environment:
        print(json.dumps(dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
                              gpu=torch.cuda.get_device_name(0), cuda_smoke_test="passed")), flush=True)
        return
    data = Path(os.environ.get("DATA", root / "data_ogb")).resolve()
    dataset = data / "ogbl_biokg"
    if (dataset / "split/random/test.pt").exists():
        raise RuntimeError("This isolated campaign must not contain the test split")
    required = [dataset / path for path in (
        "split/random/train.pt", "split/random/valid.pt", "processed/data_processed", "RELEASE_v1.txt",
        "mapping/relidx2relname.csv.gz", "mapping/disease_entidx2name.csv.gz",
        "mapping/drug_entidx2name.csv.gz", "mapping/function_entidx2name.csv.gz",
        "mapping/protein_entidx2name.csv.gz", "mapping/sideeffect_entidx2name.csv.gz")]
    if not all(path.is_file() for path in required):
        raise RuntimeError("Transfer the declared dataset files before starting; no automatic dataset download")
    fingerprints = {}
    for path in required:
        with path.open("rb") as handle:
            fingerprints[str(path.relative_to(data))] = hashlib.file_digest(handle, "sha256").hexdigest()
    receipt = dict(started_utc=datetime.now(timezone.utc).isoformat(), python=sys.version,
                   executable=sys.executable, platform=platform.platform(),
                   packages={dist.metadata["Name"]: dist.version for dist in metadata.distributions()},
                   torch=torch.__version__, cuda=torch.version.cuda,
                   gpu=torch.cuda.get_device_name(0),
                   float32_matmul_precision=torch.get_float32_matmul_precision(),
                   cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                   cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                   nvidia_smi=subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                                       "--format=csv,noheader"], text=True).strip(),
                   data_sha256=fingerprints, test_present=False)
    with (root / "h33_environment.json").open("x") as handle:
        json.dump(receipt, handle, indent=2)
        handle.write("\n")
    print(f"Environment frozen: {receipt['gpu']}, torch {torch.__version__}; no test split", flush=True)
    env = dict(os.environ, PYTHON=sys.executable, DATA=str(data), DEVICE="cuda",
               PYTHONUNBUFFERED="1")
    result = subprocess.run(["bash", "biokg/scripts/run_h33.sh"], env=env)
    with (root / "h33_exit.json").open("x") as handle:
        json.dump(dict(completed_utc=datetime.now(timezone.utc).isoformat(), returncode=result.returncode), handle)
        handle.write("\n")
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
