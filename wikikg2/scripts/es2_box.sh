#!/bin/bash
set -x
cd "$(dirname "$0")/.."; export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH:-}"; export PATH="$HOME/.local/bin:$PATH"; unset WIKI_HOLDOUT
U="uv run python"; mkdir -p results/es2 compact; M="analogy_d1_t3 analogy_s0_t3 holders cn_aa linked cn3_aa typed"
for s in 0 1 2 3 4 5 6 7 8 9; do
  [ -f compact/teacher_c_s$s.pt ] || $U train_wiki.py --device cuda --tiered-from teachers/model_wiki_s$s.bf16.pt --widths 4,8,36,64 --subspaces 4096,4096,16,1 --steps 1 --lr 0 --table-lr 0 --batch 256 --neg 256 --save compact/teacher_c_s$s.pt --eval valid --probe-every 0 --ckpt-every 0 > results/es2/build_s$s.log 2>&1
  echo "s$s $(grep -a 'table params' results/es2/build_s$s.log | grep -o 'table params [0-9,]*') $(grep -a '\[valid\]' results/es2/build_s$s.log | cut -c1-30)" | tee -a results/es2/build_summary.log
done
[ -f ens_cache/rev_nov_ens10c.valid.npz ] || $U ens_cache.py --device cuda --tag ens10c --split valid --models $(ls compact/teacher_c_s*.pt) 2>&1 | grep -v "^dir" | tee results/es2/ens10c.log
for set in "ens10c:ens10c" "ens10c+members:ens10c $M" "ens10c+members+rev:ens10c $M rev_raw_ens10c rev_nov_ens10c"; do
  name=${set%%:*}; members=${set#*:}
  for mode in standard rich; do flag=""; [ $mode = rich ] && flag="--rich"
    echo "=== $name / $mode" | tee -a results/es2/selection.log
    $U blend_wiki.py search --members $members --min-rows 250 --seed 0 $flag 2>&1 | grep -E "HELD-OUT|head \(" | tee -a results/es2/selection.log
  done
done
echo ES2_DONE >> results/es2/status.log
