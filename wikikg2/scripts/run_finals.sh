#!/bin/bash
# Seeded finals for ogbl-wikikg2 — one test read per run per row.
#   row A: the single model (train_wiki --eval both reads test once)
#   row B: model + retrieval members, weights frozen on FULL valid,
#          applied once to the test caches (blend_wiki freeze --test)
#
#   SEEDS="0 1 2 3 4" STEPS=800000 bash run_finals.sh          # k=8 dense recipe (default FLAGS)
#   FLAGS="--k 12 --block-size 144 --rev-frac 0.75" SEEDS="0 1" bash run_finals.sh   # k=12 twin
#   SEED0=combo400k_s0.pt bash run_finals.sh   # also finish seed 0 from
#                                              # its valid-only run
set -e
cd "$(dirname "$0")/.."                      # wikikg2/
export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH:-}"   # shared modules at the repo root
U=${UV:-uv}
STEPS="${STEPS:-400000}"
NEG="${NEG:-4096}"
# recipe settled 2026-09-03: k=8, dense 64x64 relation operators, 75% head-direction rows
FLAGS="${FLAGS:---k 8 --block-size 64 --rev-frac 0.75}"
# blend guard: (relation, direction) groups with >= MINROWS validation rows get
# their own weights (biokg default 4000 starved the test-dominant relations)
MINROWS="${MINROWS:-500}"
PREFIX="${PREFIX:-model_wiki}"   # checkpoint / member name prefix (k=12 twin: PREFIX=model_k12)
mkdir -p logs results
# model-free members (holders, cn, cn_aa, linked) are shared by all seeds
for sp in valid test; do
  [ -f ens_cache/cn.$sp.npz ] || $U run python cn_wiki.py --device cuda --split $sp \
      2>&1 | grep -v "^dir . row" | tee -a logs/members_$sp.log
  [ -f ens_cache/cn3_aa.$sp.npz ] || $U run python cn3_wiki.py --device cuda --split $sp \
      --cap 16 2>&1 | grep -v "^dir . row" | tee -a logs/members_$sp.log
done

finish_seed() {  # $1 = seed, $2 = checkpoint
  local s=$1 ck=$2 tag at
  tag=$(basename "$ck" .pt); at=${PREFIX#model_}_s$s   # analogy member tag
  for sp in valid test; do
    $U run python cache_wiki.py --device cuda --split $sp --models "$ck" \
        2>&1 | tee -a logs/final_${PREFIX}_s$s.log
    $U run python retrieval_wiki.py --device cuda --model "$ck" --tag $at \
        --split $sp 2>&1 | grep -v "^dir . row" | tee -a logs/final_${PREFIX}_s$s.log
  done
  $U run python blend_wiki.py freeze --members "$tag" analogy_$at \
      analogy_${at}_t3 holders cn cn_aa linked cn3_aa --test --min-rows "$MINROWS" \
      --out results/frozen_${PREFIX}_s$s.npz \
      --result results/blend_${PREFIX}_s$s.json 2>&1 | tee -a logs/final_${PREFIX}_s$s.log
  rm -f ens_cache/analogy_$at.*.npz ens_cache/analogy_${at}_t3.*.npz \
        ens_cache/$tag.test.npz   # keep disk in check (valid cache stays)
}

if [ -n "$SEED0" ]; then
  # row A test read for the valid-only seed-0 run
  $U run python train_wiki.py --device cuda --eval-only --save "$SEED0" \
      --eval test 2>&1 | tee logs/final_${PREFIX}_s0.log
  finish_seed 0 "$SEED0"
fi

for s in $SEEDS; do
  ck=${PREFIX}_s$s.pt
  $U run python train_wiki.py --device cuda --steps "$STEPS" --neg "$NEG" \
      $FLAGS --opt rowadagrad --table-lr 0.6 --seed "$s" --save "$ck" \
      --eval both --probe-every 100000 --log-every 50000 \
      --ckpt-every 100000 2>&1 | tee logs/final_${PREFIX}_s$s.log
  finish_seed "$s" "$ck"
done
echo FINALS_DONE
