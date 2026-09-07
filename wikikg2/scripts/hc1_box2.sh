#!/bin/bash
# HC1 apply world + the fit, on the same box (the jinx<->box link is ~4 MB/s, so nothing multi-GB crosses it).
# Runs alongside hc1_box.sh: waits for the dataset, builds the validation caches of the released
# student (s1) and teacher (s0) with all of TRAIN, then waits for the fit world and runs holdout_blend.py.
#   nohup bash scripts/hc1_box2.sh > logs/hc1_box2.log 2>&1 &
set -x
cd "$(dirname "$0")/.."
export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH:-}"
export PATH="$HOME/.local/bin:$PATH"
unset WIKI_HOLDOUT
U="uv run python"
st() { echo "$(date +%H:%M:%S) $*" >> logs/hc1_status.log; }
mkdir -p ens_cache ens_cache_s0 teachers results/hc1
until grep -q "holdout built" logs/hc1_status.log 2>/dev/null; do sleep 30; done
[ -f teachers/model_wiki_s0.bf16.pt ] || curl -fsSL -o teachers/model_wiki_s0.bf16.pt https://github.com/jinxmcg/resonate/releases/download/v2.0-two-boards/model_wiki_s0.bf16.pt
[ -f ens_cache/cn_aa.valid.npz ] || $U cn_wiki.py --device cuda --split valid --out-dir ens_cache 2>&1 | grep -v "^dir . row" > logs/hc1_cn_valid.log
st "cn valid done"
[ -f ens_cache/cn3_aa.valid.npz ] || $U cn3_wiki.py --device cuda --split valid --cap 16 --out-dir ens_cache 2>&1 | grep -v "^dir . row" > logs/hc1_cn3_valid.log
st "cn3 valid done"
$U cache_wiki.py --device cuda --split valid --models teachers/model_dist_s1.bf16.pt teachers/model_wiki_s0.bf16.pt --out ens_cache 2>&1 | tee logs/hc1_cache_valid.log
st "model caches valid done"
if [ ! -f ens_cache/analogy_s0_t3.valid.npz ]; then
  $U retrieval_wiki.py --device cuda --model teachers/model_wiki_s0.bf16.pt --tag s0 --split valid --out-dir ens_cache_s0 2>&1 | grep -v "^dir . row" > logs/hc1_retrieval_s0_valid.log
  cp ens_cache_s0/analogy_s0_t3.valid.npz ens_cache/
fi
st "analogy s0 valid done"
until grep -q "HC1_BOX_DONE" logs/hc1_status.log 2>/dev/null; do sleep 30; done
FIT="model_fit97_s0 analogy_f0_t3 holders cn_aa linked cn3_aa typed"
$U holdout_blend.py --device cuda --fit-cache-dir ens_cache_holdout --cache-dir ens_cache --label dist_s1 \
    --fit-members $FIT --apply-members model_dist_s1.bf16 analogy_d1_t3 holders cn_aa linked cn3_aa typed 2>&1 | tee results/hc1/dist_s1.log
st "blend dist_s1 done"
$U holdout_blend.py --device cuda --fit-cache-dir ens_cache_holdout --cache-dir ens_cache --label wiki_s0 \
    --fit-members $FIT --apply-members model_wiki_s0.bf16 analogy_s0_t3 holders cn_aa linked cn3_aa typed 2>&1 | tee results/hc1/wiki_s0.log
st "HC1_ALL_DONE"
echo HC1_ALL_DONE
