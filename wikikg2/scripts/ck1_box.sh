#!/bin/bash
# CK1: k=6 and k=4 dense-operator screens at 200k steps, validation only.
set -x
cd "$(dirname "$0")/.."
export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH:-}"; export PATH="$HOME/.local/bin:$PATH"; unset WIKI_HOLDOUT
U="uv run python"; mkdir -p logs results/ck1
for k in 6 4; do
  bs=$((k*k))
  $U train_wiki.py --device cuda --steps 200000 --neg 4096 --k $k --block-size $bs --rev-frac 0.75 \
     --opt rowadagrad --table-lr 0.6 --seed 0 --save model_k${k}_200k.pt --eval valid \
     --probe-every 100000 --log-every 50000 --ckpt-every 100000 > results/ck1/k${k}_200k.log 2>&1
  grep -a "\[valid\]\|params\|parameters" results/ck1/k${k}_200k.log | tail -2
done
echo CK1_DONE >> results/ck1/status.log
