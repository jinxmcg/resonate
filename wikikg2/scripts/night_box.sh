#!/bin/bash
# CK3 + ES1 chain (waits for CK2 and the checkpoint downloads). Validation only.
set -x
cd "$(dirname "$0")/.."
export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH:-}"; export PATH="$HOME/.local/bin:$PATH"; unset WIKI_HOLDOUT
U="uv run python"; mkdir -p results/ck3 results/es1
st() { echo "$(date +%H:%M:%S) $*" >> results/night_status.log; }
until grep -q CK2_DONE results/ck2/status.log 2>/dev/null; do sleep 30; done
# ---- CK3: full k=6
[ -f model_k6.pt ] || $U train_wiki.py --device cuda --steps 800000 --neg 4096 --k 6 --block-size 36 --rev-frac 0.75 \
   --opt rowadagrad --table-lr 0.6 --seed 0 --save model_k6.pt --eval valid --probe-every 100000 --log-every 50000 --ckpt-every 100000 > results/ck3/k6_800k.log 2>&1
st "CK3 teacher k6 done: $(grep -a '\[valid\]' results/ck3/k6_800k.log | tail -1)"
[ -f ens_cache/model_k6.valid.npz ] || $U cache_wiki.py --device cuda --split valid --models model_k6.pt --out ens_cache 2>&1 | grep MRR
[ -f ens_cache/rev_nov_k6.valid.npz ] || $U reverse_wiki.py --device cuda --model model_k6.pt --tag k6 --split valid 2>&1 | grep -v "^dir"
[ -f ens_cache/analogy_k6_t3.valid.npz ] || $U retrieval_wiki.py --device cuda --model model_k6.pt --tag k6 --split valid --out-dir ens_cache_k6 2>&1 | grep -v "^dir . row"; cp -n ens_cache_k6/analogy_k6_t3.valid.npz ens_cache/ 2>/dev/null
S=model_dist_s1.bf16; M7="holders cn_aa linked cn3_aa typed"
for set in "k6:model_k6" "k6+9:model_k6 analogy_k6_t3 $M7 rev_raw_k6 rev_nov_k6" "student+k6:$S model_k6" \
           "student+k6+9:$S model_k6 analogy_d1_t3 $M7 rev_raw_d1 rev_nov_d1" "student+k6+11:$S model_k6 analogy_d1_t3 analogy_k6_t3 $M7 rev_raw_d1 rev_nov_d1 rev_raw_k6 rev_nov_k6"; do
  name=${set%%:*}; members=${set#*:}; echo "=== $name" | tee -a results/ck3/selection.log
  $U blend_wiki.py search --members $members --min-rows 250 --seed 0 2>&1 | grep -E "alone|model-only|HELD-OUT|tail \(|head \(" | tee -a results/ck3/selection.log
done
st "CK3 done"; echo CK3_DONE >> results/ck3/status.log
# ---- ES1: ensembles
until [ -f teachers/dl_status ]; do sleep 30; done
ST="teachers/model_dist_s1.bf16.pt teachers/model_dist_s4.bf16.pt teachers/model_dist_s5.bf16.pt teachers/model_dist_s6.bf16.pt teachers/model_dist_s7.bf16.pt teachers/model_dist_s8.bf16.pt teachers/model_dist_s9.bf16.pt"
TE=$(for s in 0 1 2 3 4 5 6 7 8 9; do echo -n "teachers/model_wiki_s$s.bf16.pt "; done)
[ -f ens_cache/rev_nov_ens7s.valid.npz ] || $U ens_cache.py --device cuda --tag ens7s --split valid --models $ST 2>&1 | grep -v "^dir" | tee results/es1/ens7s.log
st "ens7s cached"
[ -f ens_cache/rev_nov_ens10t.valid.npz ] || $U ens_cache.py --device cuda --tag ens10t --split valid --models $TE 2>&1 | grep -v "^dir" | tee results/es1/ens10t.log
st "ens10t cached"
[ -f ens_cache/rev_nov_ens17.valid.npz ] || $U ens_cache.py --device cuda --tag ens17 --split valid --models $ST $TE 2>&1 | grep -v "^dir" | tee results/es1/ens17.log
st "ens17 cached"
M="analogy_d1_t3 analogy_s0_t3 $M7"
for e in ens7s ens10t ens17; do
  for set in "$e:$e" "$e+members:$e $M" "$e+members+rev:$e $M rev_raw_$e rev_nov_$e"; do
    name=${set%%:*}; members=${set#*:}; echo "=== $name" | tee -a results/es1/selection.log
    $U blend_wiki.py search --members $members --min-rows 250 --seed 0 2>&1 | grep -E "alone|model-only|HELD-OUT|tail \(|head \(" | tee -a results/es1/selection.log
  done
done
st "ES1 done"; echo ES1_DONE >> results/es1/status.log
