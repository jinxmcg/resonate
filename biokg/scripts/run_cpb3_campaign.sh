#!/usr/bin/env bash
# CP-B3: the compact distilled row (entry 2) over ten seeds.
#
# Per seed: compress the released T=2 student's table by degree tier
# (k-means + per-cluster PCA, no training), expand to the public dense
# checkpoint format, build its two analogy members, take the held-out
# estimate on validation halves, then ONE frozen test read.
#
# The shared Jaccard pair is model-free and must already be in $CACHE
# (built once by jaccard_member.py for valid and test).
#
#   SEEDS="0" scripts/run_cpb3_campaign.sh          # one seed
#   scripts/run_cpb3_campaign.sh                    # all ten
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH:-}"

SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9"}
DEVICE=${DEVICE:-cuda}
DATA=${DATA:-data_ogb}
CKPT=${CKPT:-checkpoints}
CACHE=${CACHE:-ens_cache}
OUT=${OUT:-runs}
PY=${PY:-python}
# the CP-B1 frontier configuration: 9,555,497 parameters
CUTOFFS=${CUTOFFS:-"5 8 32 128 1024"}
CFG=${CFG:-"2,4,8,48,144,144:256,256,256,32,1,1"}
mkdir -p "$CACHE" "$OUT"

for sp in valid test; do
  [ -f "$CACHE/jaccard_t3.$sp.npz" ] || { echo "missing $CACHE/jaccard_t3.$sp.npz - run jaccard_member.py --split $sp first" >&2; exit 2; }
done

for s in $SEEDS; do
  src="$CKPT/dist_T2_s$s.pt"; m="cpb3_dist_w48_s$s"; a="analogy_$m"
  echo "== CP-B3 seed $s: $src -> $m =="
  # 1. compress (init only, ~10 s) and expand to the dense format
  if [ ! -f "$CKPT/$m.pt" ]; then
    $PY compress_biokg.py --model "$src" --cutoffs $CUTOFFS --configs "$CFG" \
        --device "$DEVICE" --data-root "$DATA" --no-wide-eval \
        --save-prefix "$CKPT/cpb3_dist_s$s" 2>&1 | tee "$OUT/cpb3_compress_s$s.log"
    $PY tiered_to_dense.py "$CKPT/cpb3_dist_s${s}_0.pt" "$CKPT/$m.pt" "$DATA"
  fi
  # 2. model scores and analogy members, both splits (no test MRR printed)
  for sp in valid test; do
    $PY cache_scores.py --device "$DEVICE" --split "$sp" --models "$CKPT/$m.pt" \
        --out "$CACHE" --data-root "$DATA"
    if [ ! -f "$CACHE/${a}_t3.$sp.npz" ]; then
      $PY analogy_member.py --device "$DEVICE" --split "$sp" --models "$CKPT/$m.pt" \
          --data-root "$DATA" --out "$CACHE/$a.$sp.npz" --out-top3 "$CACHE/${a}_t3.$sp.npz" \
          2>&1 | tee "$OUT/$a.$sp.log"
    fi
  done
  members="$m $a ${a}_t3 jaccard jaccard_t3"
  # 3. held-out estimate on validation halves
  $PY ensemble_weights.py --members $members --norm z --seed 0 --min-rows 2000 \
      --cache-dir "$CACHE" --data-root "$DATA" 2>&1 | tee "$OUT/pure_cpb3_s$s.log"
  # 4. THE test shot: weights frozen on the full valid split, applied once
  $PY freeze_test.py --members $members --norm z --min-rows 4000 \
      --cache-dir "$CACHE" --data-root "$DATA" \
      --out "$OUT/frozen_cpb3_s$s.npz" --result "$OUT/committed_cpb3_s$s.json" \
      2>&1 | tee "$OUT/committed_cpb3_s$s.log"
done
