#!/bin/bash
# CP3: tiered refit with K learned subspaces per tier (k-means init), widths 8,16,36,64, 200k steps, validation only; then the per-tier diagnostic.
set -x
cd "$(dirname "$0")/.."; export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH:-}"; export PATH="$HOME/.local/bin:$PATH"; unset WIKI_HOLDOUT
U="uv run python"; mkdir -p results/cp3
W=teachers/model_dist_s1.bf16.pt
[ -f model_cp3_k256.pt ] || $U train_wiki.py --device cuda --tiered-from $W --widths 8,16,36,64 --subspaces 256,256,16,1 --steps 200000 --neg 4096 --batch 2048 \
    --opt rowadagrad --table-lr 0.6 --seed 0 --save model_cp3_k256.pt --eval valid --probe-every 50000 --log-every 25000 --ckpt-every 0 \
    --distill $W --distill-w 1.0 --distill-T 2.0 > results/cp3/cp3_k256.log 2>&1
echo "$(date +%H:%M:%S) CP3 k256: $(grep -a '\[valid\]' results/cp3/cp3_k256.log | tail -1)" >> results/night_status.log
sed -i 's#("tiered108M", "model_tiered_16_32_64_64.pt")#("tiered108M", "model_tiered_16_32_64_64.pt"), ("cp3_k256", "model_cp3_k256.pt")#' tier_diag.py
$U tier_diag.py 2>&1 | grep -v "^WIKI\|loaded in" | tee results/cp3/tier_diag.log
echo CP3_DONE >> results/cp3/status.log
