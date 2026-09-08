#!/bin/bash
# TR1 (NOT TO BE RUN WITHOUT THE USER'S GO): test reads for the allowed rows. Usage: bash scripts/tr1_box.sh A|B|C|D [--rich]
# Builds the row's test caches, freezes the selection on full validation, applies once to test, records the official MRR.
set -x
cd "$(dirname "$0")/.."; export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH:-}"; export PATH="$HOME/.local/bin:$PATH"; unset WIKI_HOLDOUT
U="uv run python"; mkdir -p results/tr1; row=$1; RICH=${2:-}
shared() {  # model-free test members
  [ -f ens_cache/cn_aa.test.npz ] || $U cn_wiki.py --device cuda --split test --out-dir ens_cache 2>&1 | grep -v "^dir . row"
  [ -f ens_cache/cn3_aa.test.npz ] || $U cn3_wiki.py --device cuda --split test --cap 16 --out-dir ens_cache 2>&1 | grep -v "^dir . row"
  [ -f ens_cache/typed.test.npz ] || $U typed_paths.py --device cuda --split test --out-dir ens_cache --table typed_lo.npz 2>&1 | grep -v "^dir . row"
}
permodel() {  # $1 model path, $2 tag: scores, reverse, analogy (+holders) on test
  [ -f ens_cache/$(basename $1 .pt).test.npz ] || $U cache_wiki.py --device cuda --split test --models $1 --out ens_cache
  [ -f ens_cache/rev_nov_$2.test.npz ] || $U reverse_wiki.py --device cuda --model $1 --tag $2 --split test 2>&1 | grep -v "^dir"
  [ -f ens_cache/analogy_$2_t3.test.npz ] || { $U retrieval_wiki.py --device cuda --model $1 --tag $2 --split test --out-dir ens_cache_$2 2>&1 | grep -v "^dir . row"; cp -n ens_cache_$2/analogy_$2_t3.test.npz ens_cache/; [ -f ens_cache/holders.test.npz ] || cp ens_cache_$2/holders.test.npz ens_cache/; }
}
ST="teachers/model_dist_s1.bf16.pt teachers/model_dist_s4.bf16.pt teachers/model_dist_s5.bf16.pt teachers/model_dist_s6.bf16.pt teachers/model_dist_s7.bf16.pt teachers/model_dist_s8.bf16.pt teachers/model_dist_s9.bf16.pt"
TE=$(for s in 0 1 2 3 4 5 6 7 8 9; do echo -n "teachers/model_wiki_s$s.bf16.pt "; done)
shared
case $row in
  A) permodel teachers/model_dist_s1.bf16.pt d1
     $U blend_wiki.py freeze --members model_dist_s1.bf16 analogy_d1_t3 holders cn_aa linked cn3_aa typed rev_raw_d1 rev_nov_d1 --test --min-rows 250 $RICH --out results/tr1/frozen_A.npz --result results/tr1/A.json 2>&1 | tee results/tr1/A.log ;;
  B) permodel teachers/model_dist_s1.bf16.pt d1; permodel teachers/model_wiki_s0.bf16.pt s0
     [ -f ens_cache/rev_nov_ens7s.test.npz ] || $U ens_cache.py --device cuda --tag ens7s --split test --models $ST 2>&1 | grep -v "^dir"
     $U blend_wiki.py freeze --members ens7s analogy_d1_t3 analogy_s0_t3 holders cn_aa linked cn3_aa typed rev_raw_ens7s rev_nov_ens7s --test --min-rows 250 $RICH --out results/tr1/frozen_B.npz --result results/tr1/B.json 2>&1 | tee results/tr1/B.log ;;
  C) permodel teachers/model_dist_s1.bf16.pt d1; permodel teachers/model_wiki_s0.bf16.pt s0
     [ -f ens_cache/rev_nov_ens10t.test.npz ] || $U ens_cache.py --device cuda --tag ens10t --split test --models $TE 2>&1 | grep -v "^dir"
     $U blend_wiki.py freeze --members ens10t analogy_d1_t3 analogy_s0_t3 holders cn_aa linked cn3_aa typed rev_raw_ens10t rev_nov_ens10t --test --min-rows 250 $RICH --out results/tr1/frozen_C.npz --result results/tr1/C.json 2>&1 | tee results/tr1/C.log ;;
  D) permodel model_k6.pt k6
     $U blend_wiki.py freeze --members model_k6 analogy_k6_t3 holders cn_aa linked cn3_aa typed rev_raw_k6 rev_nov_k6 --test --min-rows 250 $RICH --out results/tr1/frozen_D.npz --result results/tr1/D.json 2>&1 | tee results/tr1/D.log ;;
esac
echo "TR1_${row}_DONE" >> results/tr1/status.log
