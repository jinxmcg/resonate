#!/bin/bash
# Row E+ on jinx: ensemble + base members + augmented members (seed-0 proposals) + typed. One test read.
cd "$(dirname "$0")/.."                      # wikikg2/
export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH:-}"   # shared modules at the repo root
U=${UV:-uv}
PY="$U run python"
M0=teachers/model_wiki_s0.bf16.pt
$PY augment_graph.py --model $M0 --relations all --p 0 --min-prec 0.7 --out extra_e.npz --device cuda 2>&1 | grep -E "^->|Trace"
for sp in valid test; do
  $PY typed_paths.py --device cuda --split $sp 2>&1 | grep -E "alone|Trace|^done"
  $PY cn_wiki.py --device cuda --split $sp --extra-edges extra_e.npz --suffix _aug 2>&1 | grep -E "alone|cached|Trace"
  rm -f ens_cache/cn_aug.$sp.npz ens_cache/linked_aug.$sp.npz
  $PY cn3_wiki.py --device cuda --split $sp --cap 16 --extra-edges extra_e.npz --suffix _aug 2>&1 | grep -E "alone|cached|Trace"
  $PY retrieval_wiki.py --device cuda --model $M0 --tag s0 --split $sp --extra-edges extra_e.npz --suffix _aug 2>&1 | grep -E "alone|cached|Trace"
  rm -f ens_cache/analogy_s0_aug.$sp.npz
done
df -h . | tail -1
$PY blend_wiki.py freeze --members ens10 analogy_s0_t3 holders cn_aa linked cn3_aa analogy_s0_t3_aug holders_aug cn_aa_aug cn3_aa_aug typed --test --min-rows 500 \
    --out results/frozen_ensplus.npz --result results/blend_ensplus.json 2>&1 | grep -E "alone|frozen|fallback|official|direction:|hits|Trace"
echo ENSPLUS_DONE
