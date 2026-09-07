#!/bin/bash
# HC1 fit world on the rented 5090: holdout split, 97% teacher, holdout members,
# plus the student's validation retrieval member and typed.valid (GPU-heavy, not for the 1080 Ti).
#   nohup bash scripts/hc1_box.sh > logs/hc1_box.log 2>&1 &
set -x
cd "$(dirname "$0")/.."                      # wikikg2/
export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH:-}"
export PATH="$HOME/.local/bin:$PATH"
U="uv run python"
mkdir -p logs ens_cache ens_cache_holdout teachers results/hc1
st() { echo "$(date +%H:%M:%S) $*" >> logs/hc1_status.log; }
[ -d data_ogb/ogbl_wikikg2 ] || { echo y | $U download_data.py; }
[ -f teachers/model_dist_s1.bf16.pt ] || curl -fsSL -o teachers/model_dist_s1.bf16.pt https://github.com/jinxmcg/resonate/releases/download/v2.0-two-boards/model_dist_s1.bf16.pt
st "setup done"
# 0. the split (official TRAIN, no env var)
[ -f holdout_wiki.npz ] || $U holdout_wiki.py --out holdout_wiki.npz 2>&1 | tee logs/hc1_holdout.log
st "holdout built"
export WIKI_HOLDOUT="$PWD/holdout_wiki.npz"
# 1. teacher on fit-TRAIN (row-A recipe), in the background
if [ ! -f model_fit97_s0.pt ]; then
  nohup $U train_wiki.py --device cuda --steps 800000 --neg 4096 --k 8 --block-size 64 --rev-frac 0.75 \
      --opt rowadagrad --table-lr 0.6 --seed 0 --save model_fit97_s0.pt --eval valid \
      --probe-every 100000 --log-every 50000 --ckpt-every 100000 > logs/hc1_teacher_s0.log 2>&1 &
  TPID=$!
fi
# 2. model-free members on the holdout, from fit-TRAIN (concurrent with training)
[ -f ens_cache_holdout/cn_aa.holdout.npz ] || $U cn_wiki.py --device cuda --split holdout --out-dir ens_cache_holdout 2>&1 | grep -v "^dir . row" > logs/hc1_cn_holdout.log
st "cn holdout done"
[ -f ens_cache_holdout/cn3_aa.holdout.npz ] || $U cn3_wiki.py --device cuda --split holdout --cap 16 --out-dir ens_cache_holdout 2>&1 | grep -v "^dir . row" > logs/hc1_cn3_holdout.log
st "cn3 holdout done"
[ -f ens_cache_holdout/typed.holdout.npz ] || $U typed_paths.py --device cuda --split holdout --out-dir ens_cache_holdout --table typed_lo_fit97.npz 2>&1 | grep -v "^dir . row" > logs/hc1_typed_holdout.log
st "typed holdout done"
# 3. apply-world members that are missing on jinx (all of TRAIN: env var unset)
[ -f ens_cache/typed.valid.npz ] || env -u WIKI_HOLDOUT $U typed_paths.py --device cuda --split valid --out-dir ens_cache --table typed_lo.npz 2>&1 | grep -v "^dir . row" > logs/hc1_typed_valid.log
st "typed valid done"
[ -f ens_cache/analogy_d1_t3.valid.npz ] || env -u WIKI_HOLDOUT $U retrieval_wiki.py --device cuda --model teachers/model_dist_s1.bf16.pt --tag d1 --split valid --out-dir ens_cache 2>&1 | grep -v "^dir . row" > logs/hc1_retrieval_d1_valid.log
st "analogy d1 valid done"
[ -n "${TPID:-}" ] && wait $TPID
st "teacher done"
# 4. teacher-dependent holdout members
[ -f ens_cache_holdout/model_fit97_s0.holdout.npz ] || $U cache_wiki.py --device cuda --split holdout --models model_fit97_s0.pt --out ens_cache_holdout 2>&1 | tee logs/hc1_cache_holdout.log
[ -f ens_cache_holdout/analogy_f0_t3.holdout.npz ] || $U retrieval_wiki.py --device cuda --model model_fit97_s0.pt --tag f0 --split holdout --out-dir ens_cache_holdout 2>&1 | grep -v "^dir . row" > logs/hc1_retrieval_f0_holdout.log
st "HC1_BOX_DONE"
echo HC1_BOX_DONE
