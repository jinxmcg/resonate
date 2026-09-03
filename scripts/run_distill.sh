#!/usr/bin/env bash
# Row C — ten 27M students distilled at training time from the ten
# row-A checkpoints, then the same retrieval features and blend.
#
#   scripts/run_distill.sh              # needs checkpoints/model_final_seed{0..9}.pt
#   SEEDS="1 2" scripts/run_distill.sh
#
# The student has the same architecture and recipe as row A plus
# KL(mean teacher distribution || student) over each batch's
# [pos | 4096 negs] logits (--distill-w 1, --distill-T 1). Teachers are
# frozen and see only the training triples the student sees. Students
# are evaluated on valid only; their test number is read once, inside
# the blend (freeze_test.py). ~26 min per student on an RTX 5090 with
# three running concurrently.
set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9"}
DEVICE=${DEVICE:-cuda}
DATA=${DATA:-data_ogb}
CKPT=${CKPT:-checkpoints}
OUT=${OUT:-runs}
mkdir -p "$CKPT" "$OUT"

teachers=""
for t in 0 1 2 3 4 5 6 7 8 9; do teachers="$teachers $CKPT/model_final_seed$t.pt"; done

for s in $SEEDS; do
  echo "== student seed $s =="
  uv run python train_ogb.py --device "$DEVICE" --data-root "$DATA" \
      --steps 50000 --block-size 4 --seed "$s" --eval valid \
      --distill $teachers --distill-w 1.0 --distill-T 1.0 \
      --save "$CKPT/dist27_s$s.pt" \
      2>&1 | tee "$OUT/dist27_s$s.log"
done

ROW=dist SEEDS="$SEEDS" DEVICE="$DEVICE" DATA="$DATA" CKPT="$CKPT" OUT="$OUT" \
    exec bash scripts/run_retrieval.sh
