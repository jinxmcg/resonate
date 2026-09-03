#!/usr/bin/env bash
# Row A — ten independent 27M ResonatE models on ogbl-biokg.
#
#   scripts/run_campaign.sh              # seeds 0..9, sequential
#   SEEDS="3 4" scripts/run_campaign.sh  # a subset
#
# Each run: 50k steps, batch 2048, 4096 type-matched negatives, 4x4
# unitary-initialised relation blocks, Adam 5e-3 cosine. Evaluates
# valid AND test with the official OGB Evaluator once, at the end
# (--eval both): the test number of a run is read exactly once.
# ~4 min per seed on an RTX 5090, ~18 min on a GTX 1080 Ti.
set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9"}
DEVICE=${DEVICE:-cuda}
DATA=${DATA:-data_ogb}
CKPT=${CKPT:-checkpoints}
OUT=${OUT:-runs}
mkdir -p "$CKPT" "$OUT"

for s in $SEEDS; do
  echo "== seed $s =="
  uv run python train_ogb.py --device "$DEVICE" --data-root "$DATA" \
      --steps 50000 --block-size 4 --seed "$s" \
      --save "$CKPT/model_final_seed$s.pt" --eval both \
      2>&1 | tee "$OUT/final_seed$s.log"
done
