#!/bin/bash
# Finite NL1 job. Stop on failed verification; never auto-restart or sweep.
set -eo pipefail
. /opt/supervisor-scripts/utils/environment.sh
. /opt/supervisor-scripts/utils/logging.sh /var/log/portal/biokg_nl1.log
export PYTHONUNBUFFERED=1
export UV_NO_CACHE=false
export UV_CACHE_DIR=/workspace/biokg_mn1/.uv-cache
cd /workspace/biokg_nl1
/usr/local/bin/uv run --script biokg/run_nonlinear_vast.py --tests
/usr/local/bin/uv run --script biokg/run_nonlinear_vast.py --smoke
/usr/local/bin/uv run --script biokg/run_nonlinear_vast.py
/usr/local/bin/uv run --script biokg/run_nonlinear_vast.py --evaluate
/usr/local/bin/uv run --script biokg/run_nonlinear_vast.py --audit
