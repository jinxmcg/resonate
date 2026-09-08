#!/bin/bash
set -x
cd "$(dirname "$0")/.."; export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH:-}"; export PATH="$HOME/.local/bin:$PATH"; unset WIKI_HOLDOUT
U="uv run python"; mkdir -p results/cp3c; W=teachers/model_dist_s1.bf16.pt; M7="holders cn_aa linked cn3_aa typed"
build() { [ -f $2 ] || $U train_wiki.py --device cuda --tiered-from $W --widths $1 --subspaces 4096,4096,16,1 --steps 0 --save $2 --eval valid --probe-every 0 --ckpt-every 0 > results/cp3c/build_$3.log 2>&1; }
members() {  # $1 model, $2 tag
  [ -f ens_cache/$(basename $1 .pt).valid.npz ] || $U cache_wiki.py --device cuda --split valid --models $1 --out ens_cache 2>&1 | grep MRR
  [ -f ens_cache/rev_nov_$2.valid.npz ] || $U reverse_wiki.py --device cuda --model $1 --tag $2 --split valid 2>&1 | grep -v "^dir"
  [ -f ens_cache/analogy_$2_t3.valid.npz ] || { $U retrieval_wiki.py --device cuda --model $1 --tag $2 --split valid --out-dir ens_cache_$2 2>&1 | grep -v "^dir . row"; cp -n ens_cache_$2/analogy_$2_t3.valid.npz ens_cache/; }
  name=$(basename $1 .pt)
  for set in "$2:$name" "$2+9:$name analogy_$2_t3 $M7 rev_raw_$2 rev_nov_$2"; do
    nm=${set%%:*}; mem=${set#*:}; echo "=== $nm" | tee -a results/cp3c/selection.log
    $U blend_wiki.py search --members $mem --min-rows 250 --seed 0 2>&1 | grep -E "HELD-OUT|head \(" | tee -a results/cp3c/selection.log
    echo "=== $nm / rich" | tee -a results/cp3c/selection.log
    $U blend_wiki.py search --members $mem --min-rows 250 --seed 0 --rich 2>&1 | grep -E "HELD-OUT|head \(" | tee -a results/cp3c/selection.log
  done
}
build 8,16,36,64 model_cp3_k4096_w8.pt w8; build 4,8,36,64 model_cp3_k4096_w4.pt w4
members model_cp3_k4096_w8.pt c8; members model_cp3_k4096_w4.pt c4
[ -f model_cp3_k4096_w4_refit.pt ] || $U train_wiki.py --device cuda --tiered-from $W --widths 4,8,36,64 --subspaces 4096,4096,16,1 --steps 50000 --neg 4096 --batch 2048 \
    --opt rowadagrad --table-lr 0.1 --seed 0 --save model_cp3_k4096_w4_refit.pt --eval valid --probe-every 10000 --log-every 10000 --ckpt-every 0 \
    --distill $W --distill-w 1.0 --distill-T 2.0 > results/cp3c/refit_w4.log 2>&1
grep -a "probe@\|\[valid\]" results/cp3c/refit_w4.log | tail -6
sed -i 's#("cp3_k256", "model_cp3_k256.pt")#("cp3_k256", "model_cp3_k256.pt"), ("k4096_w8", "model_cp3_k4096_w8.pt"), ("k4096_w4", "model_cp3_k4096_w4.pt"), ("k4096_w4_refit", "model_cp3_k4096_w4_refit.pt")#' tier_diag.py
$U tier_diag.py 2>&1 | grep -v "^WIKI\|loaded in" | tee results/cp3c/tier_diag.log
echo CP3C_DONE >> results/cp3c/status.log
