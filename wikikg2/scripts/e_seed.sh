#!/bin/bash
# TR1 row E, per seed: compress the released T=2 student with its own clusters, refit 200k, nine members,
# rich selection frozen on full validation, ONE test read; test caches deleted afterwards.
#   bash scripts/e_seed.sh 4 5 6
set -x
cd "$(dirname "$0")/.."; export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH:-}"; export PATH="$HOME/.local/bin:$PATH"; unset WIKI_HOLDOUT
U="uv run python"; mkdir -p results/tr1 teachers
st() { echo "$(date +%H:%M:%S) $*" >> results/tr1/status.log; }
shared_valid() {
  [ -f ens_cache/cn_aa.valid.npz ] || $U cn_wiki.py --device cuda --split valid --out-dir ens_cache 2>&1 | grep -v "^dir . row"
  [ -f ens_cache/cn3_aa.valid.npz ] || $U cn3_wiki.py --device cuda --split valid --cap 16 --out-dir ens_cache 2>&1 | grep -v "^dir . row"
  [ -f ens_cache/typed.valid.npz ] || $U typed_paths.py --device cuda --split valid --out-dir ens_cache --table typed_lo.npz 2>&1 | grep -v "^dir . row"
}
shared_test() {
  [ -f ens_cache/cn_aa.test.npz ] || $U cn_wiki.py --device cuda --split test --out-dir ens_cache 2>&1 | grep -v "^dir . row"
  [ -f ens_cache/cn3_aa.test.npz ] || $U cn3_wiki.py --device cuda --split test --cap 16 --out-dir ens_cache 2>&1 | grep -v "^dir . row"
  [ -f ens_cache/typed.test.npz ] || $U typed_paths.py --device cuda --split test --out-dir ens_cache --table typed_lo.npz 2>&1 | grep -v "^dir . row"
}
shared_valid
for s in "$@"; do
  W=teachers/model_dist_s$s.bf16.pt
  [ -f $W ] || curl -fsSL -o $W https://github.com/jinxmcg/resonate/releases/download/v2.0-two-boards/model_dist_s$s.bf16.pt
  if [ "$s" = 1 ] && [ -f model_cp3_k4096_w4_full.pt ]; then M=model_cp3_k4096_w4_full.pt; tag=w4f; else M=model_cp3s${s}_w4_full.pt; tag=c$s; fi
  name=$(basename $M .pt)
  [ -f $M ] || $U train_wiki.py --device cuda --tiered-from $W --widths 4,8,36,64 --subspaces 4096,4096,16,1 --steps 200000 --neg 4096 --batch 2048 \
      --opt rowadagrad --table-lr 0.6 --seed 0 --save $M --eval valid --probe-every 50000 --log-every 25000 --ckpt-every 0 \
      --distill $W --distill-w 1.0 --distill-T 2.0 > results/tr1/refit_s$s.log 2>&1
  st "E s$s: refit done $(grep -a '\[valid\]' results/tr1/refit_s$s.log | tail -1 | cut -c1-40)"
  # validation members
  [ -f ens_cache/$name.valid.npz ] || $U cache_wiki.py --device cuda --split valid --models $M --out ens_cache 2>&1 | grep MRR
  [ -f ens_cache/rev_nov_$tag.valid.npz ] || $U reverse_wiki.py --device cuda --model $M --tag $tag --split valid 2>&1 | grep -v "^dir"
  [ -f ens_cache/analogy_${tag}_t3.valid.npz ] || { $U retrieval_wiki.py --device cuda --model $M --tag $tag --split valid --out-dir ens_cache_$tag 2>&1 | grep -v "^dir . row"; cp -n ens_cache_$tag/analogy_${tag}_t3.valid.npz ens_cache/; [ -f ens_cache/holders.valid.npz ] || cp ens_cache_$tag/holders.valid.npz ens_cache/; rm -rf ens_cache_$tag; }
  st "E s$s: validation members done"
  # test members
  shared_test
  [ -f ens_cache/$name.test.npz ] || $U cache_wiki.py --device cuda --split test --models $M --out ens_cache 2>&1 | grep -v "^dir"
  [ -f ens_cache/rev_nov_$tag.test.npz ] || $U reverse_wiki.py --device cuda --model $M --tag $tag --split test 2>&1 | grep -v "^dir"
  [ -f ens_cache/analogy_${tag}_t3.test.npz ] || { $U retrieval_wiki.py --device cuda --model $M --tag $tag --split test --out-dir ens_cache_$tag 2>&1 | grep -v "^dir . row"; cp -n ens_cache_$tag/analogy_${tag}_t3.test.npz ens_cache/; [ -f ens_cache/holders.test.npz ] || cp ens_cache_$tag/holders.test.npz ens_cache/; rm -rf ens_cache_$tag; }
  st "E s$s: test members built"
  $U blend_wiki.py freeze --members $name analogy_${tag}_t3 holders cn_aa linked cn3_aa typed rev_raw_$tag rev_nov_$tag --test --min-rows 250 --rich --out results/tr1/frozen_E_s$s.npz --result results/tr1/E_s$s.json 2>&1 | tee results/tr1/E_s$s.log
  st "E s$s READ: $(grep -a 'official Evaluator test MRR' results/tr1/E_s$s.log)"
  rm -f ens_cache/$name.test.npz ens_cache/rev_raw_$tag.test.npz ens_cache/rev_nov_$tag.test.npz ens_cache/analogy_${tag}_t3.test.npz
done
echo "E_DONE $*" >> results/tr1/status.log
