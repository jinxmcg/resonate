#!/bin/bash
# RV1: reverse members for student s1 and teacher s0 on validation, then selection-blend (and, for
# information, learned-combiner) cross-fits with and without them. Validation only.
set -x
cd "$(dirname "$0")/.."
export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH:-}"
export PATH="$HOME/.local/bin:$PATH"
unset WIKI_HOLDOUT
U="uv run python"
mkdir -p results/rv1
[ -f ens_cache/rev_nov_d1.valid.npz ] || $U reverse_wiki.py --device cuda --model teachers/model_dist_s1.bf16.pt --tag d1 --split valid 2>&1 | grep -v "^dir" | tee results/rv1/reverse_d1.log
[ -f ens_cache/rev_nov_s0.valid.npz ] || $U reverse_wiki.py --device cuda --model teachers/model_wiki_s0.bf16.pt --tag s0 --split valid 2>&1 | grep -v "^dir" | tee results/rv1/reverse_s0.log
for who in "dist_s1:model_dist_s1.bf16 analogy_d1_t3:d1" "wiki_s0:model_wiki_s0.bf16 analogy_s0_t3:s0"; do
  label=${who%%:*}; rest=${who#*:}; base=${rest%:*}; tag=${rest##*:}
  M7="$base holders cn_aa linked cn3_aa typed"
  for set in "seven:$M7" "raw:$M7 rev_raw_$tag" "nov:$M7 rev_nov_$tag" "both:$M7 rev_raw_$tag rev_nov_$tag"; do
    sname=${set%%:*}; members=${set#*:}
    echo "=== $label / $sname" | tee -a results/rv1/selection_$label.log
    $U blend_wiki.py search --members $members --min-rows 250 --seed 0 2>&1 | grep -E "alone|HELD-OUT|held-out delta|tail \(|head \(" | tee -a results/rv1/selection_$label.log
    echo "=== $label / $sname" | tee -a results/rv1/learned_$label.log
    $U learned_blend.py --members $members --min-rows 250 --seed 0 --device cuda 2>&1 | grep -E "^ *selection|^ *learned|weights:" | tee -a results/rv1/learned_$label.log
  done
done
echo RV1_DONE | tee -a results/rv1/status.log
