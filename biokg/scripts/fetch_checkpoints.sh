#!/usr/bin/env bash
# Download the released checkpoints into checkpoints/ and check them against the
# release's SHA256SUMS, so the results can be verified without retraining:
#
#   scripts/fetch_checkpoints.sh            # the submitted biokg ladder: sparse_s*.pt + dist_T2_s*.pt (20 x 108 MB)
#   scripts/fetch_checkpoints.sh dense      # the first ladder: model_final_seed*.pt + dist27_s*.pt (release v1.0-biokg)
#   scripts/fetch_checkpoints.sh wikikg2    # the ten bf16 teachers + seven surviving T=2 students (17 x 644 MB) into ../wikikg2/teachers
#
# Needs the GitHub CLI (gh) authenticated with read access to the repo.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=${REPO:-jinxmcg/resonate}
what=${1:-sparse}
case "$what" in
  sparse)  TAG=${TAG:-v2.0-two-boards}; dir=checkpoints;          pats=("sparse_s*.pt" "dist_T2_s*.pt") ;;
  dense)   TAG=${TAG:-v1.0-biokg};      dir=checkpoints;          pats=("model_final_seed*.pt" "dist27_s*.pt") ;;
  wikikg2) TAG=${TAG:-v2.0-two-boards}; dir=../wikikg2/teachers;  pats=("model_wiki_s*.bf16.pt" "model_dist_s*.bf16.pt") ;;
  *) echo "usage: $0 [sparse|dense|wikikg2]" >&2; exit 2 ;;
esac
mkdir -p "$dir"
gh release download "$TAG" --repo "$REPO" --dir "$dir" --pattern SHA256SUMS --clobber
for pat in "${pats[@]}"; do
  gh release download "$TAG" --repo "$REPO" --dir "$dir" --pattern "$pat" --skip-existing
  grep -E " (${pat//\*/.*})$" "$dir/SHA256SUMS" | (cd "$dir" && sha256sum -c --strict)
done
