#!/bin/bash
set -x
cd "$(dirname "$0")/.."; export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH:-}"; export PATH="$HOME/.local/bin:$PATH"; unset WIKI_HOLDOUT
U="uv run python"; mkdir -p results/cp3f; W=teachers/model_dist_s1.bf16.pt
until grep -q CP3E_DONE results/cp3e/status.log 2>/dev/null; do sleep 60; done
$U train_wiki.py --device cuda --tiered-from $W --widths 4,8,36,64 --subspaces 4096,4096,16,1 --tiered-random-init --steps 200000 --neg 4096 --batch 2048 \
    --opt rowadagrad --table-lr 0.6 --seed 0 --save model_cp3f_w4_scratch.pt --eval valid --probe-every 50000 --log-every 25000 --ckpt-every 0 \
    --distill $W --distill-w 1.0 --distill-T 2.0 > results/cp3f/scratch_w4.log 2>&1
echo "$(date +%H:%M:%S) CP3f: $(grep -a '\[valid\]' results/cp3f/scratch_w4.log | tail -1)" >> results/night_status.log
echo CP3F_DONE >> results/cp3f/status.log
