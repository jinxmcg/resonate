#!/bin/bash
# HC2: row-level holdout (holdout_wiki_row.npz, uploaded from jinx), 97% teacher on its fit-TRAIN,
# holdout members, then the two fits against the existing validation caches.
#   nohup bash scripts/hc2_box.sh > logs/hc2_box.log 2>&1 &
set -x
cd "$(dirname "$0")/.."
export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH:-}"
export PATH="$HOME/.local/bin:$PATH"
U="uv run python"
mkdir -p logs ens_cache_holdout_row results/hc2
st() { echo "$(date +%H:%M:%S) $*" >> logs/hc2_status.log; }
[ -f holdout_wiki_row.npz ] || { st "MISSING holdout_wiki_row.npz"; exit 1; }
export WIKI_HOLDOUT="$PWD/holdout_wiki_row.npz"
H=ens_cache_holdout_row
if [ ! -f model_fit97r_s0.pt ]; then
  nohup $U train_wiki.py --device cuda --steps 800000 --neg 4096 --k 8 --block-size 64 --rev-frac 0.75 \
      --opt rowadagrad --table-lr 0.6 --seed 0 --save model_fit97r_s0.pt --eval valid \
      --probe-every 100000 --log-every 50000 --ckpt-every 100000 > logs/hc2_teacher_s0.log 2>&1 &
  TPID=$!
fi
[ -f $H/cn_aa.holdout.npz ] || $U cn_wiki.py --device cuda --split holdout --out-dir $H 2>&1 | grep -v "^dir . row" > logs/hc2_cn_holdout.log
st "cn holdout done"
[ -f $H/cn3_aa.holdout.npz ] || $U cn3_wiki.py --device cuda --split holdout --cap 16 --out-dir $H 2>&1 | grep -v "^dir . row" > logs/hc2_cn3_holdout.log
st "cn3 holdout done"
[ -f $H/typed.holdout.npz ] || $U typed_paths.py --device cuda --split holdout --out-dir $H --table typed_lo_fit97r.npz 2>&1 | grep -v "^dir . row" > logs/hc2_typed_holdout.log
st "typed holdout done"
[ -n "${TPID:-}" ] && wait $TPID
st "teacher done"
[ -f $H/model_fit97r_s0.holdout.npz ] || $U cache_wiki.py --device cuda --split holdout --models model_fit97r_s0.pt --out $H 2>&1 | tee logs/hc2_cache_holdout.log
[ -f $H/analogy_f0r_t3.holdout.npz ] || $U retrieval_wiki.py --device cuda --model model_fit97r_s0.pt --tag f0r --split holdout --out-dir $H 2>&1 | grep -v "^dir . row" > logs/hc2_retrieval_holdout.log
st "HC2_BOX_DONE"
unset WIKI_HOLDOUT
FIT="model_fit97r_s0 analogy_f0r_t3 holders cn_aa linked cn3_aa typed"
$U holdout_blend.py --device cuda --fit-cache-dir $H --cache-dir ens_cache --label dist_s1 --out-dir results/hc2 \
    --fit-members $FIT --apply-members model_dist_s1.bf16 analogy_d1_t3 holders cn_aa linked cn3_aa typed 2>&1 | tee results/hc2/dist_s1.log
st "blend dist_s1 done"
$U holdout_blend.py --device cuda --fit-cache-dir $H --cache-dir ens_cache --label wiki_s0 --out-dir results/hc2 \
    --fit-members $FIT --apply-members model_wiki_s0.bf16 analogy_s0_t3 holders cn_aa linked cn3_aa typed 2>&1 | tee results/hc2/wiki_s0.log
st "HC2_ALL_DONE"
echo HC2_ALL_DONE
