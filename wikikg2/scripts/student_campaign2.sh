#!/bin/bash
# Ten T=2 students, each: train (400k, 10 bf16 teachers) -> members -> learned freeze, one read.
#   bash student_campaign2.sh 0 2 3 4      (needs teachers/ and extra_e.npz; builds missing shared members)
# v2: full unfiltered stage logs (logs/student_s$s.full.log), per-seed test caches are
# checked and rebuilt before the freeze, the model is dropped only once the result exists.
cd "$(dirname "$0")/.."                      # wikikg2/
export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH:-}"   # shared modules at the repo root
U=${UV:-uv}
until [ "$(ls teachers/model_wiki_s*.bf16.pt 2>/dev/null | wc -l)" -ge 10 ]; do sleep 60; done
[ -f extra_e.npz ] || $U run python augment_graph.py --model teachers/model_wiki_s0.bf16.pt --relations all --p 0 --min-prec 0.7 --out extra_e.npz --device cuda   # seed-0 proposals (ensplus_chain.sh)
for sp in valid test; do
  [ -f ens_cache/typed.$sp.npz ] || $U run python typed_paths.py --device cuda --split $sp 2>&1 | grep -E "alone|Trace|^done"
  [ -f ens_cache/cn_aa_aug.$sp.npz ] || { $U run python cn_wiki.py --device cuda --split $sp --extra-edges extra_e.npz --suffix _aug 2>&1 | grep -E "alone|cached|Trace"; rm -f ens_cache/cn_aug.$sp.npz ens_cache/linked_aug.$sp.npz; }
  [ -f ens_cache/cn3_aa_aug.$sp.npz ] || $U run python cn3_wiki.py --device cuda --split $sp --cap 16 --extra-edges extra_e.npz --suffix _aug 2>&1 | grep -E "alone|cached|Trace"
done
for s in "$@"; do
  M=model_dist_s$s.pt; tag=model_dist_s$s; L=logs/student_s$s.log; F=logs/student_s$s.full.log
  [ -f results/learned_dist_s$s.json ] && continue
  [ -f $M ] || SEED=$s STEPS=400000 DT=2.0 CKPT="${CKPT:-100000}" bash run_distill.sh 2>&1 | tee -a $F | grep -E "distill:|valid\]|test\]|Trace|No space" | tee -a $L
  rm -f $M.ckpt
  if [ ! -f $M ]; then echo "FAILED seed $s: no model after training" | tee -a $L; continue; fi
  for attempt in 1 2; do
    for sp in valid test; do
      [ -f ens_cache/$tag.$sp.npz ] || $U run python cache_wiki.py --device cuda --split $sp --models $M 2>&1 | tee -a $F | grep -E "MRR|cached|Trace" | tee -a $L
      [ -f ens_cache/analogy_d${s}_t3.$sp.npz ] || { $U run python retrieval_wiki.py --device cuda --model $M --tag d$s --split $sp 2>&1 | tee -a $F | grep -E "alone|cached|Trace" | tee -a $L; rm -f ens_cache/analogy_d$s.$sp.npz; }
      [ -f ens_cache/analogy_d${s}_t3_aug.$sp.npz ] || { $U run python retrieval_wiki.py --device cuda --model $M --tag d$s --split $sp --extra-edges extra_e.npz --suffix _aug 2>&1 | tee -a $F | grep -E "alone|cached|Trace" | tee -a $L; rm -f ens_cache/analogy_d${s}_aug.$sp.npz; }
    done
    ok=1
    for f in $tag analogy_d${s}_t3 analogy_d${s}_t3_aug holders holders_aug cn_aa cn_aa_aug linked cn3_aa cn3_aa_aug typed; do
      for sp in valid test; do [ -f ens_cache/$f.$sp.npz ] || { echo "missing ens_cache/$f.$sp.npz (attempt $attempt)" | tee -a $L; ok=0; }; done
    done
    [ $ok = 1 ] && break
  done
  if [ $ok = 1 ]; then
    $U run python learned_blend.py --freeze --test --members $tag analogy_d${s}_t3 holders cn_aa linked cn3_aa typed analogy_d${s}_t3_aug holders_aug cn_aa_aug cn3_aa_aug --out results/learned_dist_s$s.npz --result results/learned_dist_s$s.json 2>&1 | tee -a $F | grep -E "learned freeze|official|direction|Trace" | tee -a $L
  fi
  if [ -f results/learned_dist_s$s.json ]; then
    [ "${KEEP_MODEL:-1}" = "1" ] || rm -f $M
    rm -f ens_cache/$tag.test.npz ens_cache/analogy_d${s}_t3*.npz
  else
    echo "FAILED seed $s: no result; model and caches kept" | tee -a $L
  fi
done
echo CAMPAIGN2_DONE
