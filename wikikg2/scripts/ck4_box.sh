#!/bin/bash
set -x
cd "$(dirname "$0")/.."; export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH:-}"; export PATH="$HOME/.local/bin:$PATH"; unset WIKI_HOLDOUT
U="uv run python"; mkdir -p results/ck4
T=$(ls teachers/model_wiki_s*.bf16.pt | tr '\n' ' ')
[ -f model_dist_k6.pt ] || $U train_wiki.py --device cuda --steps 400000 --k 6 --block-size 36 --rev-frac 0.75 --opt rowadagrad --table-lr 0.6 --seed 0 \
    --save model_dist_k6.pt --eval valid --probe-every 100000 --log-every 50000 --ckpt-every 100000 --distill $T --distill-w 1.0 --distill-T 2.0 > results/ck4/dist_k6.log 2>&1
echo "$(date +%H:%M:%S) CK4 student k6 done: $(grep -a '\[valid\]' results/ck4/dist_k6.log | tail -1)" >> results/night_status.log
[ -f ens_cache/model_dist_k6.valid.npz ] || $U cache_wiki.py --device cuda --split valid --models model_dist_k6.pt --out ens_cache 2>&1 | grep MRR
[ -f ens_cache/rev_nov_k6d.valid.npz ] || $U reverse_wiki.py --device cuda --model model_dist_k6.pt --tag k6d --split valid 2>&1 | grep -v "^dir"
[ -f ens_cache/analogy_k6d_t3.valid.npz ] || { $U retrieval_wiki.py --device cuda --model model_dist_k6.pt --tag k6d --split valid --out-dir ens_cache_k6d 2>&1 | grep -v "^dir . row"; cp -n ens_cache_k6d/analogy_k6d_t3.valid.npz ens_cache/; }
M7="holders cn_aa linked cn3_aa typed"
for set in "k6d:model_dist_k6" "k6d+9:model_dist_k6 analogy_k6d_t3 $M7 rev_raw_k6d rev_nov_k6d"; do
  name=${set%%:*}; members=${set#*:}; echo "=== $name" | tee -a results/ck4/selection.log
  $U blend_wiki.py search --members $members --min-rows 250 --seed 0 2>&1 | grep -E "alone|HELD-OUT|head \(" | tee -a results/ck4/selection.log
done
echo CK4_DONE >> results/ck4/status.log
