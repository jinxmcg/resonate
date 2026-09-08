#!/bin/bash
set -x
cd "$(dirname "$0")/.."; export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH:-}"; export PATH="$HOME/.local/bin:$PATH"; unset WIKI_HOLDOUT
U="uv run python"; mkdir -p results/sel1
until grep -q ES1_DONE results/es1/status.log 2>/dev/null; do sleep 60; done
S=model_dist_s1.bf16; M9="analogy_d1_t3 holders cn_aa linked cn3_aa typed rev_raw_d1 rev_nov_d1"
best=$(grep -B1 "HELD-OUT" results/es1/selection.log | grep -A1 "=== ens.*rev" | grep HELD-OUT | awk '{print $NF}' | sort -n | tail -1)
for e in ens7s ens10t ens17; do grep -A3 "=== $e+members+rev" results/es1/selection.log | grep -q "HELD-OUT $best" && BEST=$e; done
BEST=${BEST:-ens10t}; MB="$BEST analogy_d1_t3 analogy_s0_t3 holders cn_aa linked cn3_aa typed rev_raw_$BEST rev_nov_$BEST"
for set in "student+9:$S $M9" "$BEST+members+rev:$MB"; do
  name=${set%%:*}; members=${set#*:}
  for mode in standard rich; do
    flag=""; [ $mode = rich ] && flag="--rich"
    echo "=== $name / $mode" | tee -a results/sel1/selection.log
    $U blend_wiki.py search --members $members --min-rows 250 --seed 0 $flag 2>&1 | grep -E "HELD-OUT|tail \(|head \(" | tee -a results/sel1/selection.log
  done
done
echo SEL1_DONE >> results/sel1/status.log
