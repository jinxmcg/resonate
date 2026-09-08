#!/bin/bash
set -x
cd "$(dirname "$0")/.."; export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH:-}"; export PATH="$HOME/.local/bin:$PATH"; unset WIKI_HOLDOUT
U="uv run python"; mkdir -p results/cp3e; W=teachers/model_dist_s1.bf16.pt; M7="holders cn_aa linked cn3_aa typed"
until grep -q CP3C2_DONE results/cp3c/status.log 2>/dev/null; do sleep 60; done
for cfg in "4,8,36,64:w4" "8,16,36,64:w8"; do
  w=${cfg%%:*}; t=${cfg##*:}
  [ -f model_cp3_k4096_${t}_full.pt ] || $U train_wiki.py --device cuda --tiered-from $W --widths $w --subspaces 4096,4096,16,1 --steps 200000 --neg 4096 --batch 2048 \
      --opt rowadagrad --table-lr 0.6 --seed 0 --save model_cp3_k4096_${t}_full.pt --eval valid --probe-every 50000 --log-every 25000 --ckpt-every 0 \
      --distill $W --distill-w 1.0 --distill-T 2.0 > results/cp3e/refit_$t.log 2>&1
  echo "$(date +%H:%M:%S) CP3e $t: $(grep -a '\[valid\]' results/cp3e/refit_$t.log | tail -1)" >> results/night_status.log
  m=model_cp3_k4096_${t}_full.pt; tag=${t}f; name=$(basename $m .pt)
  $U cache_wiki.py --device cuda --split valid --models $m --out ens_cache 2>&1 | grep MRR
  $U reverse_wiki.py --device cuda --model $m --tag $tag --split valid 2>&1 | grep -v "^dir"
  $U retrieval_wiki.py --device cuda --model $m --tag $tag --split valid --out-dir ens_cache_$tag 2>&1 | grep -v "^dir . row"; cp -n ens_cache_$tag/analogy_${tag}_t3.valid.npz ens_cache/
  for mode in standard rich; do flag=""; [ $mode = rich ] && flag="--rich"
    echo "=== $tag+9 / $mode" | tee -a results/cp3e/selection.log
    $U blend_wiki.py search --members $name analogy_${tag}_t3 $M7 rev_raw_$tag rev_nov_$tag --min-rows 250 --seed 0 $flag 2>&1 | grep -E "HELD-OUT|head \(" | tee -a results/cp3e/selection.log
  done
done
echo CP3E_DONE >> results/cp3e/status.log
