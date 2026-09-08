#!/bin/bash
# CP3c, second pass: build the two K=4096 models with a zero-LR step (so they are saved), members, selection, diagnostic.
set -x
cd "$(dirname "$0")/.."; export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH:-}"; export PATH="$HOME/.local/bin:$PATH"; unset WIKI_HOLDOUT
U="uv run python"; mkdir -p results/cp3c; W=teachers/model_dist_s1.bf16.pt; M7="holders cn_aa linked cn3_aa typed"
until grep -q CP3C_DONE results/cp3c/status.log 2>/dev/null; do sleep 60; done
build() { [ -f $2 ] || $U train_wiki.py --device cuda --tiered-from $W --widths $1 --subspaces 4096,4096,16,1 --steps 1 --lr 0 --table-lr 0 --batch 256 --neg 256 --save $2 --eval valid --probe-every 0 --ckpt-every 0 > results/cp3c/build2_$3.log 2>&1; grep -a "\[valid\]" results/cp3c/build2_$3.log | tail -1; }
members() {  # $1 model, $2 tag
  name=$(basename $1 .pt)
  [ -f ens_cache/$name.valid.npz ] || $U cache_wiki.py --device cuda --split valid --models $1 --out ens_cache 2>&1 | grep MRR
  [ -f ens_cache/rev_nov_$2.valid.npz ] || $U reverse_wiki.py --device cuda --model $1 --tag $2 --split valid 2>&1 | grep -v "^dir"
  [ -f ens_cache/analogy_$2_t3.valid.npz ] || { $U retrieval_wiki.py --device cuda --model $1 --tag $2 --split valid --out-dir ens_cache_$2 2>&1 | grep -v "^dir . row"; cp -n ens_cache_$2/analogy_$2_t3.valid.npz ens_cache/; }
  for set in "$2:$name" "$2+9:$name analogy_$2_t3 $M7 rev_raw_$2 rev_nov_$2"; do
    nm=${set%%:*}; mem=${set#*:}
    for mode in standard rich; do flag=""; [ $mode = rich ] && flag="--rich"
      echo "=== $nm / $mode" | tee -a results/cp3c/selection2.log
      $U blend_wiki.py search --members $mem --min-rows 250 --seed 0 $flag 2>&1 | grep -E "HELD-OUT|head \(" | tee -a results/cp3c/selection2.log
    done
  done
}
build 8,16,36,64 model_cp3_k4096_w8.pt w8; build 4,8,36,64 model_cp3_k4096_w4.pt w4
members model_cp3_k4096_w8.pt c8; members model_cp3_k4096_w4.pt c4
[ -f model_cp3_k4096_w4_refit.pt ] && members model_cp3_k4096_w4_refit.pt c4r
$U tier_diag.py 2>&1 | grep -v "^WIKI\|loaded in" | tee results/cp3c/tier_diag2.log
echo CP3C2_DONE >> results/cp3c/status.log
