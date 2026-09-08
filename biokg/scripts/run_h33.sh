#!/usr/bin/env bash
# H33: six fresh 50k-step runs, random auxiliary versus original CE.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
PYTHON=${PYTHON:-python}
DATA=${DATA:-biokg/data_ogb}
OUT=${OUT:-biokg/runs/h33}
REPORT=${REPORT:-biokg/results/h33}
DIAG_PREFIX=${DIAG_PREFIX:-biokg/results/error_analysis/h33}
DEVICE=${DEVICE:-cuda}
for path in "$OUT" "$REPORT"; do
  if [[ -e "$path" ]]; then
    echo "Refusing to reuse $path; choose a fresh path." >&2
    exit 2
  fi
done
for seed in 0 1 2; do
  for arm in baseline random; do
    if [[ -e "${DIAG_PREFIX}_${arm}_s${seed}" ]]; then
      echo "Refusing to reuse ${DIAG_PREFIX}_${arm}_s${seed}." >&2
      exit 2
    fi
  done
done
if [[ "$DEVICE" == cuda ]]; then
  "$PYTHON" -c 'import torch; torch.empty(1, device="cuda"); print(torch.cuda.get_device_name(0))'
fi
"$PYTHON" -m unittest discover -s biokg -p 'test_*.py' -v
"$PYTHON" -m biokg.summarize_h33 --phase initialize --run-root "$OUT" --report-root "$REPORT" --diag-prefix "$DIAG_PREFIX"
mkdir -p "$OUT"
for seed in 0 1 2; do
  for arm in baseline random; do
    "$PYTHON" -m biokg.summarize_h33 --phase check --run-root "$OUT" --report-root "$REPORT" --diag-prefix "$DIAG_PREFIX"
    directory="$OUT/${arm}_s${seed}"
    mkdir "$directory"
    extra=()
    if [[ "$arm" == random ]]; then extra=(--mining-mode random); fi
    "$PYTHON" -u -m biokg.train_biokg_comp --device "$DEVICE" --data-root "$DATA" \
      --shell sparse --table-dtype fp32 --table-lr 0.3 --k 12 --block-size 4 \
      --steps 50000 --batch 2048 --neg 4096 --lr 0.005 --lam 0.1 --seed "$seed" \
      --direction-sampling uniform --mining-count 64 --mining-weight 0.1 \
      --mining-warmup 2500 --mining-ramp 1250 --eval valid \
      --probe-every 5000 --probe-size 5000 \
      --save "$directory/model.pt" "${extra[@]}" 2>&1 | tee "$directory/train.log"
    "$PYTHON" -m biokg.summarize_h33 --phase check --run-root "$OUT" --report-root "$REPORT" --diag-prefix "$DIAG_PREFIX"
    "$PYTHON" -u -m biokg.analyze_errors --checkpoint "$directory/model.pt" \
      --data-root "$DATA" --out "${DIAG_PREFIX}_${arm}_s${seed}" --device "$DEVICE" \
      2>&1 | tee "$directory/analysis.log"
  done
  "$PYTHON" -m biokg.summarize_h33 --phase pair --seed "$seed" --run-root "$OUT" --report-root "$REPORT" --diag-prefix "$DIAG_PREFIX"
done
"$PYTHON" -m biokg.summarize_h33 --phase aggregate --run-root "$OUT" --report-root "$REPORT" --diag-prefix "$DIAG_PREFIX"
