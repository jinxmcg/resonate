#!/usr/bin/env bash
# H30: unchanged vs training-positive-masked sampled CE, validation only.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
PYTHON=${PYTHON:-python}
DATA=${DATA:-biokg/data_ogb}
OUT=${OUT:-biokg/runs/h30}
DEVICE=${DEVICE:-cuda}
if [[ -e "$OUT" ]]; then
  echo "Refusing to reuse $OUT; select a fresh OUT directory." >&2
  exit 2
fi
if [[ "$DEVICE" == cuda ]]; then
  "$PYTHON" -c 'import torch; torch.empty(1, device="cuda"); print(torch.cuda.get_device_name(0))'
fi
"$PYTHON" -m unittest biokg.test_train_positive_filter -v
mkdir -p "$OUT"
for arm in baseline filtered; do
  mkdir "$OUT/$arm"
  extra=()
  if [[ "$arm" == filtered ]]; then extra=(--filter-train-positives); fi
  "$PYTHON" -u -m biokg.train_biokg_comp --device "$DEVICE" --data-root "$DATA" \
    --shell sparse --table-dtype fp32 --table-lr 0.3 --k 12 --block-size 4 \
    --steps 12500 --batch 2048 --neg 4096 --lr 0.005 --lam 0.1 --seed 0 \
    --eval valid --save "$OUT/$arm/model.pt" "${extra[@]}" \
    2>&1 | tee "$OUT/$arm/train.log"
done
