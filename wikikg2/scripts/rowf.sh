#!/bin/bash
# Row F: per seed, single model + 10 members with the learned combiner; one test read per seed.
cd "$(dirname "$0")/.."                      # wikikg2/
export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH:-}"   # shared modules at the repo root
U=${UV:-uv}
for s in 1 2 3 4 5 6 7 8 9; do
  M=teachers/model_wiki_s$s.bf16.pt; tag=model_wiki_s$s; L=logs/rowf_s$s.log
  for sp in valid test; do
    [ -f ens_cache/$tag.$sp.npz ] || { $U run python cache_wiki.py --device cuda --split $sp --models $M 2>&1 | grep -E "MRR|cached|Trace" | tee -a $L; mv -f ens_cache/model_wiki_s$s.bf16.$sp.npz ens_cache/$tag.$sp.npz; }
    $U run python retrieval_wiki.py --device cuda --model $M --tag s$s --split $sp 2>&1 | grep -E "alone|cached|Trace" | tee -a $L
    rm -f ens_cache/analogy_s$s.$sp.npz
    $U run python retrieval_wiki.py --device cuda --model $M --tag s$s --split $sp --extra-edges extra_e.npz --suffix _aug 2>&1 | grep -E "alone|cached|Trace" | tee -a $L
    rm -f ens_cache/analogy_s${s}_aug.$sp.npz
  done
  $U run python learned_blend.py --freeze --test --members $tag analogy_s${s}_t3 holders cn_aa linked cn3_aa typed analogy_s${s}_t3_aug holders_aug cn_aa_aug cn3_aa_aug --out results/learned_s$s.npz --result results/learned_s$s.json 2>&1 | grep -v -i "warning\|setlocale" | tee -a $L
  rm -f ens_cache/$tag.test.npz ens_cache/analogy_s${s}_t3.*.npz ens_cache/analogy_s${s}_t3_aug.*.npz
done
echo ROWF_DONE
