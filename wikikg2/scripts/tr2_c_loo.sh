#!/bin/bash
# TR2 part 2: ten leave-one-out reads of entry C (nine-teacher ensembles), standard selection frozen on full validation.
set -x
cd "$(dirname "$0")/.."; export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH:-}"; export PATH="$HOME/.local/bin:$PATH"; unset WIKI_HOLDOUT
U="${PYBIN:-uv run python}"; mkdir -p results/tr1 ens_cache
st() { echo "$(date +%H:%M:%S) $*" >> results/tr1/status.log; }
for s in 0 1 2 3 4 5 6 7 8 9; do [ -f teachers/model_wiki_s$s.bf16.pt ] || curl -fsSL -o teachers/model_wiki_s$s.bf16.pt https://github.com/jinxmcg/resonate/releases/download/v2.0-two-boards/model_wiki_s$s.bf16.pt; done
[ -f teachers/model_dist_s1.bf16.pt ] || curl -fsSL -o teachers/model_dist_s1.bf16.pt https://github.com/jinxmcg/resonate/releases/download/v2.0-two-boards/model_dist_s1.bf16.pt
for sp in valid test; do
  [ -f ens_cache/cn_aa.$sp.npz ] || $U cn_wiki.py --device cuda --split $sp --out-dir ens_cache 2>&1 | grep -v "^dir . row"
  [ -f ens_cache/cn3_aa.$sp.npz ] || $U cn3_wiki.py --device cuda --split $sp --cap 16 --out-dir ens_cache 2>&1 | grep -v "^dir . row"
  [ -f ens_cache/typed.$sp.npz ] || $U typed_paths.py --device cuda --split $sp --out-dir ens_cache --table typed_lo.npz 2>&1 | grep -v "^dir . row"
  for m in "teachers/model_dist_s1.bf16.pt:d1" "teachers/model_wiki_s0.bf16.pt:s0"; do p=${m%%:*}; t=${m##*:}
    [ -f ens_cache/analogy_${t}_t3.$sp.npz ] || { $U retrieval_wiki.py --device cuda --model $p --tag $t --split $sp --out-dir ens_cache_$t 2>&1 | grep -v "^dir . row"; cp -n ens_cache_$t/analogy_${t}_t3.$sp.npz ens_cache/; [ -f ens_cache/holders.$sp.npz ] || cp ens_cache_$t/holders.$sp.npz ens_cache/; rm -rf ens_cache_$t; }
  done
done
st "C LOO: shared members built"
for i in 0 1 2 3 4 5 6 7 8 9; do
  TE=$(for s in 0 1 2 3 4 5 6 7 8 9; do [ $s != $i ] && echo -n "teachers/model_wiki_s$s.bf16.pt "; done)
  for sp in valid test; do [ -f ens_cache/rev_nov_loo$i.$sp.npz ] || $U ens_cache.py --device cuda --tag loo$i --split $sp --models $TE 2>&1 | grep -v "^dir"; done
  $U blend_wiki.py freeze --members loo$i analogy_d1_t3 analogy_s0_t3 holders cn_aa linked cn3_aa typed rev_raw_loo$i rev_nov_loo$i --test --min-rows 250 --out results/tr1/frozen_C_loo$i.npz --result results/tr1/C_loo$i.json 2>&1 | tee results/tr1/C_loo$i.log
  st "C LOO $i READ: $(grep -a 'official Evaluator test MRR' results/tr1/C_loo$i.log)"
  rm -f ens_cache/loo$i.*.npz ens_cache/rev_raw_loo$i.*.npz ens_cache/rev_nov_loo$i.*.npz
done
echo TR2_C_DONE >> results/tr1/status.log
