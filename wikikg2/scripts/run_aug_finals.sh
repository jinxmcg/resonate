#!/bin/bash
# Row B+ : self-augmented graph members added to the blend, one test read per seed.
#   bash run_aug_finals.sh 0 1 2      (models model_wiki_s<seed>.pt must be present)
set -e
cd "$(dirname "$0")/.."                      # wikikg2/
export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH:-}"   # shared modules at the repo root
U=${UV:-uv}
MINROWS="${MINROWS:-500}"
for s in "$@"; do
  ck=model_wiki_s$s.pt; tag=model_wiki_s$s; L=logs/augfinal_s$s.log
  [ -f "$ck" ] || { echo "missing $ck"; continue; }
  $U run python augment_graph.py --model $ck --relations all --p 0 --min-prec 0.7 --out extra_s$s.npz 2>&1 | grep -E "^->|Trace" | tee -a $L
  [ -f ens_cache/$tag.valid.npz ] || $U run python cache_wiki.py --device cuda --split valid --models $ck 2>&1 | grep -E "MRR|Trace" | tee -a $L
  $U run python cache_wiki.py --device cuda --split test --models $ck 2>&1 | grep -E "cached|Trace" | tee -a $L
  for sp in valid test; do
    $U run python retrieval_wiki.py --device cuda --model $ck --tag s$s --split $sp 2>&1 | grep -E "alone|cached|Trace" | tee -a $L
    rm -f ens_cache/analogy_s$s.$sp.npz
    $U run python retrieval_wiki.py --device cuda --model $ck --tag s$s --split $sp --extra-edges extra_s$s.npz --suffix _aug 2>&1 | grep -E "alone|cached|Trace" | tee -a $L
    rm -f ens_cache/analogy_s${s}_aug.$sp.npz
    $U run python cn_wiki.py --device cuda --split $sp --extra-edges extra_s$s.npz --suffix _aug 2>&1 | grep -E "alone|cached|Trace" | tee -a $L
    rm -f ens_cache/cn_aug.$sp.npz ens_cache/linked_aug.$sp.npz
    $U run python cn3_wiki.py --device cuda --split $sp --cap 16 --extra-edges extra_s$s.npz --suffix _aug 2>&1 | grep -E "alone|cached|Trace" | tee -a $L
  done
  # 50 GB disks: keep 10 members (drop the max-analogy and plain-cn variants, ~2 GB each)
  rm -f ens_cache/analogy_s$s.*.npz ens_cache/analogy_s${s}_aug.*.npz ens_cache/cn.*.npz.tmp ens_cache/cn_aug.*.npz
  df -h / | tail -1 | tee -a $L
  # verify every member cache opens before the (expensive) freeze; full log of the freeze
  $U run python - <<PY 2>&1 | tee -a $L
import numpy as np
for m in "$tag analogy_s${s}_t3 holders cn_aa linked cn3_aa analogy_s${s}_t3_aug holders_aug cn_aa_aug cn3_aa_aug".split():
    for sp in ("valid", "test"):
        try:
            d = np.load(f"ens_cache/{m}.{sp}.npz"); _ = d["sn"].shape
        except Exception as e:
            print("BAD CACHE", m, sp, repr(e)[:80])
print("cache check done")
PY
  $U run python blend_wiki.py freeze --members $tag analogy_s${s}_t3 holders cn_aa linked cn3_aa \
      analogy_s${s}_t3_aug holders_aug cn_aa_aug cn3_aa_aug \
      --test --min-rows "$MINROWS" --out results/frozenaug_s$s.npz --result results/blendaug_s$s.json 2>&1 \
      | grep -v -i "warning\|setlocale" | tee -a $L
  rm -f extra_s$s.npz ens_cache/*_aug.*.npz ens_cache/analogy_s${s}_t3.*.npz ens_cache/$tag.test.npz $ck
done
echo AUGFINALS_DONE
