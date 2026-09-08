#!/bin/bash
set -x
cd "$(dirname "$0")/.."; export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH:-}"; export PATH="$HOME/.local/bin:$PATH"; unset WIKI_HOLDOUT
U="uv run python"; mkdir -p results/cp3d; W=teachers/model_dist_s1.bf16.pt
for K in "512,4096,2048,512,16,1" "1024,4096,4096,1024,16,1"; do for w in 2,4,8,16,36,64 4,8,16,32,36,64 2,4,4,8,36,64; do
  tag=K$(echo $K | cut -d, -f1)_w$(echo $w | tr ',' '_')
  [ -f results/cp3d/$tag.log ] || $U train_wiki.py --device cuda --tiered-from $W --tiers 5,8,16,64,1024 --widths $w --subspaces $K --steps 0 --save /tmp/cp3d_$tag.pt --eval valid --probe-every 0 --ckpt-every 0 > results/cp3d/$tag.log 2>&1
  rm -f /tmp/cp3d_$tag.pt
  echo "$tag $(grep -a 'table params' results/cp3d/$tag.log | grep -o 'table params [0-9,]*') $(grep -a '\[valid\]' results/cp3d/$tag.log | cut -c1-40)" | tee -a results/cp3d/summary.log
done; done
echo CP3D_DONE >> results/cp3d/status.log
