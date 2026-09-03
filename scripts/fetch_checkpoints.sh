#!/usr/bin/env bash
# Download the twenty released checkpoints (10 campaign models + 10
# distilled students, ~108 MB each) into checkpoints/ so the results
# can be verified without retraining:
#
#   scripts/fetch_checkpoints.sh            # everything
#   scripts/fetch_checkpoints.sh campaign   # only model_final_seed*.pt
#
# Needs the GitHub CLI (gh) authenticated with read access to the repo.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=${REPO:-jinxmcg/resonate}
TAG=${TAG:-v1.0-biokg}
mkdir -p checkpoints
case "${1:-all}" in
  campaign) pat="model_final_seed*.pt" ;;
  students) pat="dist27_s*.pt" ;;
  all)      pat="*.pt" ;;
  *) echo "usage: $0 [all|campaign|students]" >&2; exit 2 ;;
esac
gh release download "$TAG" --repo "$REPO" --dir checkpoints --pattern "$pat" \
    --skip-existing
ls -l checkpoints
