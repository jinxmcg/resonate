#!/bin/bash
# HC1 apply-world model caches on jinx's 1080 Ti (torch 2.6 venv, fp32 tables), then the fit + report
# once the box's caches have been pulled into /mnt/geocore/wiki/ens_cache{,_holdout}.
#   bash scripts/hc1_local.sh caches     # student s1 + teacher s0 scored on validation
#   bash scripts/hc1_local.sh blend      # holdout fit -> validation report (student, then teacher diagnostic)
set -x
cd "$(dirname "$0")/.."
export PYTHONPATH="$(cd .. && pwd)"
PY=/mnt/geocore/geocore/.venv/bin/python
W=/mnt/geocore/wiki
case "${1:-caches}" in
  caches)
    $PY cache_wiki.py --device cuda --split valid --models $W/teachers/model_dist_s1.bf16.pt $W/teachers/model_wiki_s0.bf16.pt \
        --out $W/ens_cache --data-root $W/data_ogb --table-dtype fp32 ;;
  blend)
    mkdir -p results/hc1
    $PY holdout_blend.py --device cuda --fit-cache-dir $W/ens_cache_holdout --cache-dir $W/ens_cache --label dist_s1 \
        --fit-members model_fit97_s0 analogy_f0_t3 holders cn_aa linked cn3_aa typed \
        --apply-members model_dist_s1.bf16 analogy_d1_t3 holders cn_aa linked cn3_aa typed 2>&1 | tee results/hc1/dist_s1.log
    $PY holdout_blend.py --device cuda --fit-cache-dir $W/ens_cache_holdout --cache-dir $W/ens_cache --label wiki_s0 \
        --fit-members model_fit97_s0 analogy_f0_t3 holders cn_aa linked cn3_aa typed \
        --apply-members model_wiki_s0.bf16 analogy_s0_t3 holders cn_aa linked cn3_aa typed 2>&1 | tee results/hc1/wiki_s0.log ;;
esac
