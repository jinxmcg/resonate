#!/bin/bash
# CP2: tiered refit of the k=8 student's table, two width configurations, 200k steps, validation only.
set -x
cd "$(dirname "$0")/.."; export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH:-}"; export PATH="$HOME/.local/bin:$PATH"; unset WIKI_HOLDOUT
U="uv run python"; mkdir -p results/cp2
W=teachers/model_dist_s1.bf16.pt
for cfg in 8,16,36,64 16,32,64,64; do
  tag=$(echo $cfg | tr ',' '_')
  [ -f model_tiered_$tag.pt ] || $U train_wiki.py --device cuda --tiered-from $W --widths $cfg --steps 200000 --neg 4096 --batch 2048 \
      --opt rowadagrad --table-lr 0.6 --seed 0 --save model_tiered_$tag.pt --eval valid --probe-every 50000 --log-every 25000 --ckpt-every 0 \
      --distill $W --distill-w 1.0 --distill-T 2.0 > results/cp2/tiered_$tag.log 2>&1
  echo "$(date +%H:%M:%S) CP2 $cfg: $(grep -a '\[valid\]' results/cp2/tiered_$tag.log | tail -1)" >> results/night_status.log
done
echo CP2_DONE >> results/cp2/status.log
