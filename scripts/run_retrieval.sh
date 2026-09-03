#!/usr/bin/env bash
# Rows B and C — one model + its four label-free retrieval features,
# blended per (relation, direction) with weights selected on valid.
#
#   scripts/run_retrieval.sh                 # row B: campaign models
#   ROW=dist scripts/run_retrieval.sh        # row C: distilled students
#   SEEDS="3" ROW=single scripts/run_retrieval.sh
#
# Per seed, in order:
#   1. cache_scores.py     model scores on valid and test  (~1 min)
#   2. analogy_member.py   analogy max + top-3 mean, valid and test
#                          (~25-40 min per split; ~6 GB RAM)
#   3. jaccard_member.py   edge-overlap Jaccard max + top-3 mean, valid
#                          and test — model-free, computed ONCE and
#                          shared by every seed and both rows
#   4. ensemble_weights.py held-out estimate on valid halves (fit on a
#                          random half of triples, report on the other)
#   5. freeze_test.py      THE test shot: weights frozen on the full
#                          valid split, applied once to the test caches,
#                          official Evaluator MRR -> committed json
# Test caches are written without printing any per-member test number;
# the only test read is step 5.
set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9"}
ROW=${ROW:-single}
DEVICE=${DEVICE:-cuda}
DATA=${DATA:-data_ogb}
CKPT=${CKPT:-checkpoints}
CACHE=${CACHE:-ens_cache}
OUT=${OUT:-runs}
mkdir -p "$CACHE" "$OUT"

case "$ROW" in
  single) model_tag() { echo "model_final_seed$1"; }
          feat_tag()  { echo "analogy_s$1"; }
          pure_log()  { echo "$OUT/pure_s$1.log"; } ;;
  dist)   model_tag() { echo "dist27_s$1"; }
          feat_tag()  { echo "analogy_dist27_s$1"; }
          pure_log()  { echo "$OUT/pure_dist_s$1.log"; } ;;
  *) echo "ROW must be single or dist" >&2; exit 2 ;;
esac

# 3. shared Jaccard features (train graph only; no model involved)
for sp in valid test; do
  if [ ! -f "$CACHE/jaccard_t3.$sp.npz" ]; then
    echo "== jaccard $sp =="
    uv run python jaccard_member.py --split "$sp" --out-dir "$CACHE" \
        --data-root "$DATA" 2>&1 | tee "$OUT/jaccard.$sp.log"
  fi
done

for s in $SEEDS; do
  m=$(model_tag "$s"); a=$(feat_tag "$s")
  echo "== $ROW seed $s: model $m, features $a =="
  for sp in valid test; do
    # 1. model scores (skips if cached)
    uv run python cache_scores.py --device "$DEVICE" --split "$sp" \
        --models "$CKPT/$m.pt" --out "$CACHE" --data-root "$DATA"
    # 2. analogy features
    if [ ! -f "$CACHE/${a}_t3.$sp.npz" ]; then
      uv run python analogy_member.py --device "$DEVICE" --split "$sp" \
          --models "$CKPT/$m.pt" --data-root "$DATA" \
          --out "$CACHE/$a.$sp.npz" --out-top3 "$CACHE/${a}_t3.$sp.npz" \
          2>&1 | tee "$OUT/$a.$sp.log"
    fi
  done
  members="$m $a ${a}_t3 jaccard jaccard_t3"
  # 4. held-out estimate (valid only)
  uv run python ensemble_weights.py --members $members --norm z --seed 0 \
      --min-rows 2000 --cache-dir "$CACHE" --data-root "$DATA" \
      2>&1 | tee "$(pure_log "$s")"
  # 5. the committed test shot
  uv run python freeze_test.py --members $members --norm z --min-rows 4000 \
      --cache-dir "$CACHE" --data-root "$DATA" \
      --out "$OUT/frozen_${ROW}_s$s.npz" \
      --result "$OUT/committed_${ROW}_s$s.json" \
      2>&1 | tee "$OUT/committed_${ROW}_s$s.log"
done
