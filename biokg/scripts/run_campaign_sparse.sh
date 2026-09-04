#!/usr/bin/env bash
# Row A, sparse shell (the submitted ladder) — ten 27M models on ogbl-biokg trained
# through the wikikg2 training shell: real-view entity table with row-sparse gradients
# and row-wise Adagrad (lr 0.3), Adam elsewhere; everything else is the row-A recipe.
#
#   scripts/run_campaign_sparse.sh              # seeds 0..9
#   SEEDS="3 4" scripts/run_campaign_sparse.sh
#
# --eval both: valid AND test with the official Evaluator, once, at the end.
# Each checkpoint is then re-saved in the dense format (sparse_to_dense.py: an exact
# reinterpretation of the real-view table) so the retrieval scripts read it unchanged.
# ~4.5 min per seed on an RTX 5090.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH:-}"
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9"}; DEVICE=${DEVICE:-cuda}; DATA=${DATA:-data_ogb}
CKPT=${CKPT:-checkpoints}; OUT=${OUT:-runs}; mkdir -p "$CKPT" "$OUT"
for s in $SEEDS; do
  echo "== sparse seed $s =="
  uv run python train_biokg_comp.py --device "$DEVICE" --data-root "$DATA" \
      --steps 50000 --k 12 --block-size 4 --shell sparse --table-lr 0.3 --table-dtype fp32 \
      --seed "$s" --save "$CKPT/biokg_sparse_s$s.pt" --eval both \
      2>&1 | tee "$OUT/h24_sparse_s$s.log"
  uv run python sparse_to_dense.py "$CKPT/biokg_sparse_s$s.pt" "$CKPT/sparse_s$s.pt"
done
