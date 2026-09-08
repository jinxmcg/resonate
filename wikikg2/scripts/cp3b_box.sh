#!/bin/bash
set -x
cd "$(dirname "$0")/.."; export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH:-}"; export PATH="$HOME/.local/bin:$PATH"; unset WIKI_HOLDOUT
U="uv run python"; mkdir -p results/cp3b; W=teachers/model_dist_s1.bf16.pt
for K in 256 1024 4096; do for w in 8,16,36,64 4,8,36,64 4,4,16,64; do
  tag=K${K}_w$(echo $w | tr ',' '_')
  [ -f results/cp3b/$tag.log ] || $U train_wiki.py --device cuda --tiered-from $W --widths $w --subspaces $K,$K,16,1 --steps 0 --save /tmp/cp3b_$tag.pt --eval valid --probe-every 0 --ckpt-every 0 > results/cp3b/$tag.log 2>&1
  rm -f /tmp/cp3b_$tag.pt
  echo "$tag $(grep -a 'table params' results/cp3b/$tag.log | grep -o 'table params [0-9,]*') $(grep -a '\[valid\]' results/cp3b/$tag.log | cut -c1-40)" | tee -a results/cp3b/summary.log
done; done
echo CP3B_DONE >> results/cp3b/status.log
