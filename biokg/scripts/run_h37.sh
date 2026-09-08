#!/bin/bash
# Finite H37 pair. A failed test/pilot/phase prevents later phases.
set -eo pipefail
. /opt/supervisor-scripts/utils/environment.sh
. /opt/supervisor-scripts/utils/logging.sh /var/log/portal/biokg_h37.log
export PYTHONUNBUFFERED=1
export UV_NO_CACHE=false
export UV_CACHE_DIR=/workspace/biokg_mn1/.uv-cache
cd /workspace/biokg_h37
/usr/local/bin/uv run --script biokg/run_h37_vast.py --tests
/usr/local/bin/uv run --script biokg/run_h37_vast.py --pilot
/usr/local/bin/uv run --script biokg/run_h37_vast.py
/usr/local/bin/uv run --script biokg/run_h37_vast.py --evaluate
/usr/local/bin/uv run --script biokg/run_h37_vast.py --audit
