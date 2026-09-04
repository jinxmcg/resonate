#!/bin/bash
# Row C: one student distilled from the ten k=8 teachers (bf16 files in teachers/), then row-B+ style blend.
#   SEED=0 STEPS=400000 bash run_distill.sh
set -e
cd "$(dirname "$0")/.."                      # wikikg2/
export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH:-}"   # shared modules at the repo root
U=${UV:-uv}
SEED="${SEED:-0}"; STEPS="${STEPS:-400000}"; DT="${DT:-1.0}"; CKPT="${CKPT:-100000}"   # distillation temperature; checkpoint interval (0 = none)
T=$(ls teachers/model_wiki_s*.bf16.pt | tr '\n' ' ')
ck=model_dist_s$SEED.pt
$U run python train_wiki.py --device cuda --steps "$STEPS" --k 8 --block-size 64 --rev-frac 0.75 \
    --opt rowadagrad --table-lr 0.6 --seed "$SEED" --save "$ck" --eval both \
    --probe-every 100000 --log-every 50000 --ckpt-every "$CKPT" \
    --distill $T --distill-w 1.0 --distill-T "$DT" 2>&1 | tee logs/distill_s$SEED.log
echo DISTILL_TRAIN_DONE
