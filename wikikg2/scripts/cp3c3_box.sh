#!/bin/bash
# CP3c, third pass after the disk filled: the 50M model's rows (init and gentle refit) and the per-tier diagnostic.
set -x
cd "$(dirname "$0")/.."; export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH:-}"; export PATH="$HOME/.local/bin:$PATH"; unset WIKI_HOLDOUT
U="uv run python"; M7="holders cn_aa linked cn3_aa typed"
rows() {  # $1 model, $2 tag
  name=$(basename $1 .pt)
  [ -f ens_cache/$name.valid.npz ] || $U cache_wiki.py --device cuda --split valid --models $1 --out ens_cache 2>&1 | grep MRR
  [ -f ens_cache/rev_nov_$2.valid.npz ] || $U reverse_wiki.py --device cuda --model $1 --tag $2 --split valid 2>&1 | grep -v "^dir"
  [ -f ens_cache/analogy_$2_t3.valid.npz ] || { $U retrieval_wiki.py --device cuda --model $1 --tag $2 --split valid --out-dir ens_cache_$2 2>&1 | grep -v "^dir . row"; cp -n ens_cache_$2/analogy_$2_t3.valid.npz ens_cache/; rm -rf ens_cache_$2; }
  for mode in standard rich; do flag=""; [ $mode = rich ] && flag="--rich"
    echo "=== $2+9 / $mode" | tee -a results/cp3c/selection3.log
    $U blend_wiki.py search --members $name analogy_$2_t3 $M7 rev_raw_$2 rev_nov_$2 --min-rows 250 --seed 0 $flag 2>&1 | grep -E "HELD-OUT|head \(" | tee -a results/cp3c/selection3.log
  done
}
rows model_cp3_k4096_w4.pt c4; rows model_cp3_k4096_w4_refit.pt c4r
$U tier_diag.py 2>&1 | grep -v "^WIKI\|loaded in" | tee results/cp3c/tier_diag3.log
echo CP3C3_DONE >> results/cp3c/status.log
