#!/usr/bin/env bash
# H29 initial screen. Training graph only; validation evaluation only.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
PYTHON=${PYTHON:-python}
DATA=${DATA:-biokg/data_ogb}
OUT=${OUT:-biokg/runs/h29}
DEVICE=${DEVICE:-cuda}
for arm in baseline global4 local4; do
  if [[ -e "$OUT/$arm" ]]; then
    echo "Refusing to reuse $OUT/$arm; select a fresh OUT directory." >&2
    exit 2
  fi
done
mkdir -p "$OUT"
for arm in baseline global4 local4; do
  mkdir "$OUT/$arm"
  extra=()
  case "$arm" in
    global4) extra=(--low-rank 4);;
    local4) extra=(--low-rank 4 --low-rank-local);;
  esac
  "$PYTHON" -u biokg/train_biokg_comp.py --device "$DEVICE" --data-root "$DATA" \
    --shell sparse --table-dtype fp32 --table-lr 0.3 --k 12 --block-size 4 \
    --steps 12500 --batch 2048 --neg 4096 --lr 0.005 --lam 0.1 --seed 0 \
    --eval valid --save "$OUT/$arm/model.pt" "${extra[@]}" \
    2>&1 | tee "$OUT/$arm/train.log"
done
