#!/usr/bin/env bash
# Row C, sparse shell — ten students distilled at training time from the ten sparse
# row-A checkpoints (checkpoints/sparse_s{0..9}.pt, dense format), temperature T
# (the submitted row uses T=2; T=1 was run first and is reported as the ablation),
# then the retrieval features and the single frozen test read per seed.
#
#   scripts/run_distill_sparse.sh              # T=2, seeds 0..9
#   T=1.0 SEEDS="0" scripts/run_distill_sparse.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH:-}"
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9"}; DEVICE=${DEVICE:-cuda}; DATA=${DATA:-data_ogb}
CKPT=${CKPT:-checkpoints}; OUT=${OUT:-runs}; T=${T:-2.0}; mkdir -p "$CKPT" "$OUT"
case "$T" in 2.0|2) row=distT2; tag=dist_T2;; 1.0|1) row=distT1; tag=dist_sparse;; *) echo "T must be 1 or 2 for the shipped receipt names" >&2; exit 2;; esac
teachers=""; for t in 0 1 2 3 4 5 6 7 8 9; do teachers="$teachers $CKPT/sparse_s$t.pt"; done
for s in $SEEDS; do
  echo "== student seed $s, T=$T =="
  uv run python train_biokg_comp.py --device "$DEVICE" --data-root "$DATA" \
      --steps 50000 --k 12 --block-size 4 --shell sparse --table-lr 0.3 --table-dtype fp32 \
      --seed "$s" --distill $teachers --distill-w 1.0 --distill-T "$T" --eval valid \
      --save "$CKPT/biokg_${tag}_s$s.pt" 2>&1 | tee "$OUT/h2x_${tag}_s$s.log"
  uv run python sparse_to_dense.py "$CKPT/biokg_${tag}_s$s.pt" "$CKPT/${tag}_s$s.pt"
done
ROW=$row SEEDS="$SEEDS" DEVICE="$DEVICE" DATA="$DATA" CKPT="$CKPT" OUT="$OUT" exec bash scripts/run_retrieval.sh
