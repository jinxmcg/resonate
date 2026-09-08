#!/bin/bash
set -eo pipefail
. /opt/supervisor-scripts/utils/environment.sh
. /opt/supervisor-scripts/utils/logging.sh /var/log/portal/biokg_h37_stable.log
export PYTHONUNBUFFERED=1
export UV_NO_CACHE=false
export UV_CACHE_DIR=/workspace/biokg_mn1/.uv-cache
cd /workspace/biokg_h37
/usr/local/bin/uv run --script biokg/run_h37_stable_vast.py --tests
/usr/local/bin/uv run --script biokg/run_h37_stable_vast.py --reproducer
/usr/local/bin/uv run --script biokg/run_h37_stable_vast.py --pilot
/usr/local/bin/uv run --script biokg/run_h37_stable_vast.py
/usr/local/bin/uv run --script biokg/run_h37_stable_vast.py --evaluate
/usr/local/bin/uv run --script biokg/run_h37_stable_vast.py --audit
