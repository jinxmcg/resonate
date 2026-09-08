#!/usr/bin/env bash
# H32: 2x2 random auxiliary x train-only head/tail skew, four fresh runs.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
PYTHON=${PYTHON:-python}
DATA=${DATA:-biokg/data_ogb}
OUT=${OUT:-biokg/runs/h32}
REPORT=${REPORT:-biokg/results/h32}
DIAG_PREFIX=${DIAG_PREFIX:-biokg/results/error_analysis/h32}
DEVICE=${DEVICE:-cuda}
for path in "$OUT" "$REPORT"; do
  if [[ -e "$path" ]]; then
    echo "Refusing to reuse $path; select fresh paths." >&2
    exit 2
  fi
done
for arm in baseline random skew random_skew; do
  if [[ -e "${DIAG_PREFIX}_${arm}_s0" ]]; then
    echo "Refusing to reuse ${DIAG_PREFIX}_${arm}_s0." >&2
    exit 2
  fi
done
if [[ "$DEVICE" == cuda ]]; then
  "$PYTHON" -c 'import torch; torch.empty(1, device="cuda"); print(torch.cuda.get_device_name(0))'
fi
"$PYTHON" -m unittest discover -s biokg -p 'test_*.py' -v
"$PYTHON" -m biokg.summarize_h32 --initialize --run-root "$OUT" --report-root "$REPORT" --diag-prefix "$DIAG_PREFIX"
mkdir -p "$OUT"
for arm in baseline random skew random_skew; do
  mkdir "$OUT/$arm"
  extra=()
  case "$arm" in
    random) extra=(--mining-mode random) ;;
    skew) extra=(--direction-sampling train-fanout) ;;
    random_skew) extra=(--mining-mode random --direction-sampling train-fanout) ;;
  esac
  "$PYTHON" -u -m biokg.train_biokg_comp --device "$DEVICE" --data-root "$DATA" \
    --shell sparse --table-dtype fp32 --table-lr 0.3 --k 12 --block-size 4 \
    --steps 12500 --batch 2048 --neg 4096 --lr 0.005 --lam 0.1 --seed 0 \
    --mining-count 64 --mining-weight 0.1 --mining-warmup 2500 --mining-ramp 1250 \
    --eval valid --save "$OUT/$arm/model.pt" "${extra[@]}" \
    2>&1 | tee "$OUT/$arm/train.log"
done
for arm in baseline random skew random_skew; do
  "$PYTHON" -u -m biokg.analyze_errors --checkpoint "$OUT/$arm/model.pt" \
    --data-root "$DATA" --out "${DIAG_PREFIX}_${arm}_s0" --device "$DEVICE" \
    2>&1 | tee "$OUT/$arm/analysis.log"
done
"$PYTHON" -m biokg.summarize_h32 --run-root "$OUT" --report-root "$REPORT" --diag-prefix "$DIAG_PREFIX"
