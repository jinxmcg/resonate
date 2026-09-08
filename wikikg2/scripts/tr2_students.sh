#!/bin/bash
# TR2 part 1: retrain the lost T=2 students (seeds given), then the E procedure for each.  bash scripts/tr2_students.sh 0 2
set -x
cd "$(dirname "$0")/.."; export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH:-}"; export PATH="$HOME/.local/bin:$PATH"; unset WIKI_HOLDOUT
U="${PYBIN:-uv run python}"; mkdir -p results/tr1 teachers
for s in 0 1 2 3 4 5 6 7 8 9; do [ -f teachers/model_wiki_s$s.bf16.pt ] || curl -fsSL -o teachers/model_wiki_s$s.bf16.pt https://github.com/jinxmcg/resonate/releases/download/v2.0-two-boards/model_wiki_s$s.bf16.pt; done
T=$(ls teachers/model_wiki_s*.bf16.pt | tr '\n' ' ')
for s in "$@"; do
  [ -f teachers/model_dist_s$s.bf16.pt ] || $U train_wiki.py --device cuda --steps 400000 --k 8 --block-size 64 --rev-frac 0.75 --opt rowadagrad --table-lr 0.6 --seed $s \
      --save teachers/model_dist_s$s.bf16.pt --eval valid --probe-every 100000 --log-every 50000 --ckpt-every 0 --distill $T --distill-w 1.0 --distill-T 2.0 > results/tr1/student_s$s.log 2>&1
  echo "$(date +%H:%M:%S) TR2 student s$s: $(grep -a '\[valid\]' results/tr1/student_s$s.log | tail -1 | cut -c1-40)" >> results/tr1/status.log
  bash scripts/e_seed.sh $s
done
echo "TR2_E_DONE $*" >> results/tr1/status.log
