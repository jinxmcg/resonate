#!/bin/bash
# Row E (ensemble) on jinx: waits for data + all 10 checkpoints, then members, ensemble, one test read.
cd "$(dirname "$0")/.."                      # wikikg2/
export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH:-}"   # shared modules at the repo root
U=${UV:-uv}
PY="$U run python"
mkdir -p ens_cache results
for sp in valid test; do
  [ -f ens_cache/cn_aa.$sp.npz ] || $PY cn_wiki.py --device cuda --split $sp 2>&1 | grep -E "alone|cached|Trace"
  rm -f ens_cache/cn.$sp.npz
  [ -f ens_cache/cn3_aa.$sp.npz ] || $PY cn3_wiki.py --device cuda --split $sp --cap 16 2>&1 | grep -E "alone|cached|Trace"
  [ -f ens_cache/holders.$sp.npz ] || $PY retrieval_wiki.py --device cuda --model checkpoints/model_wiki_s0.pt --tag s0 --split $sp 2>&1 | grep -E "alone|cached|Trace"
  rm -f ens_cache/analogy_s0.$sp.npz
done
until [ "$(ls checkpoints/model_wiki_s*.pt 2>/dev/null | wc -l)" -ge 10 ]; do sleep 60; done
M=$(ls checkpoints/model_wiki_s*.pt | sort | tr '\n' ' ')
echo "ensemble of: $M"
$PY ensemble_wiki.py --device cuda --split valid --tag ens10 --models $M 2>&1 | grep -v -i warning
$PY ensemble_wiki.py --device cuda --split test --tag ens10 --models $M 2>&1 | grep -v -i warning
$PY blend_wiki.py freeze --members ens10 analogy_s0_t3 holders cn_aa linked cn3_aa --test --min-rows 500 \
    --out results/frozen_ens10.npz --result results/blend_ens10.json 2>&1 | grep -E "alone|frozen|fallback|official|direction:|hits|Trace"
echo ENS_DONE
