#!/bin/bash
# CK2: ensembles of the k=8 T=2 student with the compact 200k screens (k=6, k=4), validation halves only.
set -x
cd "$(dirname "$0")/.."
export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH:-}"; export PATH="$HOME/.local/bin:$PATH"; unset WIKI_HOLDOUT
U="uv run python"; mkdir -p results/ck2
until grep -q CK1_DONE results/ck1/status.log 2>/dev/null; do sleep 30; done
$U cache_wiki.py --device cuda --split valid --models model_k6_200k.pt model_k4_200k.pt --out ens_cache 2>&1 | tee results/ck2/cache.log
S=model_dist_s1.bf16; M9="analogy_d1_t3 holders cn_aa linked cn3_aa typed rev_raw_d1 rev_nov_d1"
for set in "student:$S" "k6:model_k6_200k" "k4:model_k4_200k" "student+k6:$S model_k6_200k" "student+k4:$S model_k4_200k" "student+k6+k4:$S model_k6_200k model_k4_200k" \
           "student+9:$S $M9" "student+k6+9:$S model_k6_200k $M9" "student+k4+9:$S model_k4_200k $M9" "student+k6+k4+9:$S model_k6_200k model_k4_200k $M9"; do
  name=${set%%:*}; members=${set#*:}
  echo "=== $name" | tee -a results/ck2/selection.log
  $U blend_wiki.py search --members $members --min-rows 250 --seed 0 2>&1 | grep -E "alone|uniform z-mean|model-only|HELD-OUT|tail \(|head \(" | tee -a results/ck2/selection.log
done
echo CK2_DONE >> results/ck2/status.log
