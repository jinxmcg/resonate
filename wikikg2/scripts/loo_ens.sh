#!/bin/bash
# Leave-one-out ensembles (9 of 10 seeds each) + members + learned combiner: 10 test reads -> mean/std for the ensemble row.
# Needs per-seed model test/valid caches on box3 (row F leaves the valid ones; test ones are rebuilt here).
cd "$(dirname "$0")/.."                      # wikikg2/
export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH:-}"   # shared modules at the repo root
U=${UV:-uv}
for s in 0 1 2 3 4 5 6 7 8 9; do
  [ -f ens_cache/model_wiki_s$s.test.npz ] || { $U run python cache_wiki.py --device cuda --split test --models teachers/model_wiki_s$s.bf16.pt 2>&1 | grep -E "cached|Trace"; mv -f ens_cache/model_wiki_s$s.bf16.test.npz ens_cache/model_wiki_s$s.test.npz; }
  [ -f ens_cache/model_wiki_s$s.valid.npz ] || { $U run python cache_wiki.py --device cuda --split valid --models teachers/model_wiki_s$s.bf16.pt 2>&1 | grep -E "MRR|Trace"; mv -f ens_cache/model_wiki_s$s.bf16.valid.npz ens_cache/model_wiki_s$s.valid.npz; }
done
$U run python - <<'PY'
import numpy as np
for sp in ("valid", "test"):
    caches = [np.load(f"ens_cache/model_wiki_s{s}.{sp}.npz") for s in range(10)]
    sp_all = np.stack([c["sp"].astype(np.float32) for c in caches]); sn_all = np.stack([c["sn"].astype(np.float32) for c in caches])
    rel = caches[0]["rel"]
    for out in range(10):
        keep = [i for i in range(10) if i != out]
        np.savez(f"ens_cache/loo{out}.{sp}.npz", sp=sp_all[keep].mean(0).astype(np.float16), sn=sn_all[keep].mean(0).astype(np.float16), rel=rel)
    print(sp, "leave-one-out caches written", flush=True)
PY
for out in 0 1 2 3 4 5 6 7 8 9; do
  $U run python learned_blend.py --freeze --test --members loo$out analogy_s0_t3 holders cn_aa linked cn3_aa typed analogy_s0_t3_aug holders_aug cn_aa_aug cn3_aa_aug --out results/learned_loo$out.npz --result results/learned_loo$out.json 2>&1 | grep -E "learned freeze|official|direction" | tee -a logs/loo.log
  rm -f ens_cache/loo$out.*.npz
done
echo LOO_DONE
