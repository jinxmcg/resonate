#!/bin/bash
# TR1 row C: the ten-teacher ensemble + members + reverse, standard selection frozen on full validation, ONE test read.
set -x
cd "$(dirname "$0")/.."; export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH:-}"; export PATH="$HOME/.local/bin:$PATH"; unset WIKI_HOLDOUT
U="uv run python"; mkdir -p results/tr1
st() { echo "$(date +%H:%M:%S) $*" >> results/tr1/status.log; }
[ -f ens_cache/cn_aa.test.npz ] || $U cn_wiki.py --device cuda --split test --out-dir ens_cache 2>&1 | grep -v "^dir . row"
[ -f ens_cache/cn3_aa.test.npz ] || $U cn3_wiki.py --device cuda --split test --cap 16 --out-dir ens_cache 2>&1 | grep -v "^dir . row"
[ -f ens_cache/typed.test.npz ] || $U typed_paths.py --device cuda --split test --out-dir ens_cache --table typed_lo.npz 2>&1 | grep -v "^dir . row"
st "C: shared test members built"
for m in "teachers/model_dist_s1.bf16.pt:d1" "teachers/model_wiki_s0.bf16.pt:s0"; do p=${m%%:*}; t=${m##*:}
  [ -f ens_cache/analogy_${t}_t3.test.npz ] || { $U retrieval_wiki.py --device cuda --model $p --tag $t --split test --out-dir ens_cache_$t 2>&1 | grep -v "^dir . row"; cp -n ens_cache_$t/analogy_${t}_t3.test.npz ens_cache/; [ -f ens_cache/holders.test.npz ] || cp ens_cache_$t/holders.test.npz ens_cache/; rm -rf ens_cache_$t; }
done
st "C: analogy test members built"
TE=$(for s in 0 1 2 3 4 5 6 7 8 9; do echo -n "teachers/model_wiki_s$s.bf16.pt "; done)
[ -f ens_cache/rev_nov_ens10t.test.npz ] || $U ens_cache.py --device cuda --tag ens10t --split test --models $TE 2>&1 | grep -v "^dir"
st "C: ensemble test caches built"
$U blend_wiki.py freeze --members ens10t analogy_d1_t3 analogy_s0_t3 holders cn_aa linked cn3_aa typed rev_raw_ens10t rev_nov_ens10t --test --min-rows 250 --out results/tr1/frozen_C.npz --result results/tr1/C.json 2>&1 | tee results/tr1/C.log
st "C READ: $(grep -a 'official Evaluator test MRR' results/tr1/C.log)"
echo C_DONE >> results/tr1/status.log
