#!/bin/bash
# STAB1: stability of the held-out numbers over the half-split seed (1, 2) for the two candidate rows, standard and rich. CPU, validation only.
set -x
cd "$(dirname "$0")/.."; export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH:-}"; export PATH="$HOME/.local/bin:$PATH"; unset WIKI_HOLDOUT
U="uv run python"; mkdir -p results/stab1
until grep -q SEL1_DONE results/sel1/status.log 2>/dev/null; do sleep 60; done
S=model_dist_s1.bf16; M9="analogy_d1_t3 holders cn_aa linked cn3_aa typed rev_raw_d1 rev_nov_d1"
MB="ens10t analogy_d1_t3 analogy_s0_t3 holders cn_aa linked cn3_aa typed rev_raw_ens10t rev_nov_ens10t"
for seed in 1 2; do for set in "student+9:$S $M9" "ens10t+members+rev:$MB"; do
  name=${set%%:*}; members=${set#*:}
  for mode in standard rich; do flag=""; [ $mode = rich ] && flag="--rich"
    echo "=== $name / $mode / seed $seed" | tee -a results/stab1/selection.log
    $U blend_wiki.py search --members $members --min-rows 250 --seed $seed $flag 2>&1 | grep -E "HELD-OUT|head \(" | tee -a results/stab1/selection.log
  done; done; done
echo STAB1_DONE >> results/stab1/status.log
