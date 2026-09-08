#!/bin/bash
# Finite MN1 job. Stop on any failed stage; never auto-resume/restart.
set -eo pipefail
. /opt/supervisor-scripts/utils/environment.sh
. /opt/supervisor-scripts/utils/logging.sh /var/log/portal/biokg_mn1.log
export PYTHONUNBUFFERED=1
# The image defaults to no cache; keep our pinned uv environment between stages.
export UV_NO_CACHE=false
export UV_CACHE_DIR=/workspace/biokg_mn1/.uv-cache
cd /workspace/biokg_mn1
/usr/local/bin/uv run --script biokg/run_masked_neighborhood_vast.py
/usr/local/bin/uv run --script biokg/run_masked_neighborhood_vast.py --evaluate
/usr/local/bin/uv run --script biokg/run_masked_neighborhood_vast.py --audit
