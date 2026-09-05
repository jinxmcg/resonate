# Validating the results without retraining

What a reviewer can run, what it costs, and what it printed when we ran it
ourselves on a machine that had nothing but this repository, its GitHub
release and the OGB download (a rented RTX 5090 on vast.ai, 128 CPUs,
`nvidia/cuda:12.8.0-devel-ubuntu22.04`, 2026-09-04/05; commit `e5fb2f3`).
Three tiers, in order of cost. Tiers 1 and 2 are what the leaderboard
claims rest on; tier 3 rebuilds one retrieval row from its checkpoint.

## Tier 1 — recompute every statistic from the shipped receipts (seconds, no GPU)

```
git clone https://github.com/jinxmcg/resonate && cd resonate
uv sync                                  # 129 s on one box, 1,975 s on another (network)
uv run python biokg/scripts/summarize.py         # submitted biokg ladder, from biokg/results/sparse/
uv run python biokg/scripts/summarize.py dense   # first ladder,          from biokg/results/dense/
uv run python wikikg2/summarize_wiki.py          # wikikg2,               from wikikg2/results/
```

Output of the first command (verbatim):

```
receipts: biokg/results/sparse

A. single 27M model (10 seeds)
  test  MRR  0.8158 +/- 0.0006
  valid MRR  0.8164 +/- 0.0006
  test hits@1/3/10  0.7457  0.8660  0.9412
  per seed: 0.8148 0.8158 0.8160 0.8167 0.8162 0.8150 0.8159 0.8158 0.8163 0.8156

B. 27M model + retrieval features
  test  MRR  0.8463 +/- 0.0004
  valid MRR  0.8472 +/- 0.0004
  held-out   0.8465 +/- 0.0004
  test hits@1/3/10  0.7875  0.8877  0.9523
  per seed: 0.8461 0.8461 0.8465 0.8461 0.8470 0.8460 0.8460 0.8466 0.8467 0.8462
  held-out - test: mean +0.0002, max |gap| 0.0007

C(T=1). distilled 27M model + retrieval features [ablation]
  test  MRR  0.8509 +/- 0.0003
  valid MRR  0.8520 +/- 0.0002
  held-out   0.8515 +/- 0.0003
  alone(val) 0.8285 +/- 0.0005
  test hits@1/3/10  0.7941  0.8915  0.9527
  per seed: 0.8508 0.8511 0.8506 0.8512 0.8506 0.8508 0.8514 0.8509 0.8510 0.8508
  held-out - test: mean +0.0005, max |gap| 0.0011

C. distilled 27M model (T=2) + retrieval features [submitted]
  test  MRR  0.8528 +/- 0.0002
  valid MRR  0.8537 +/- 0.0002
  held-out   0.8532 +/- 0.0003
  alone(val) 0.8321 +/- 0.0003
  test hits@1/3/10  0.7964  0.8933  0.9539
  per seed: 0.8531 0.8527 0.8526 0.8529 0.8527 0.8528 0.8529 0.8527 0.8531 0.8530
  held-out - test: mean +0.0004, max |gap| 0.0007
```

The `dense` variant prints the first ladder (row C: `test MRR 0.8505 +/- 0.0004`,
per seed `0.8509 0.8508 0.8506 0.8503 0.8508 0.8498 0.8502 0.8506 0.8505 0.8509`).
The wikikg2 script prints:

```
row A   single k=8 model (329M)
  test MRR   0.6676 +/- 0.0010 (n=10)   valid 0.7030 +/- 0.0009 (n=10)   test hits@1 0.6008 hits@10 0.7997
row B   A + 8 retrieval members, frozen blend
  test MRR   0.6820 +/- 0.0014 (n=10)   valid (in-sample) 0.7413 +/- 0.0007 (n=10)
row B+  + self-augmented members
  test MRR   0.6866 +/- 0.0015 (n=10)   valid (in-sample) 0.7467 +/- 0.0007 (n=10)
row F   A + 10 members + learned combiner
  test MRR   0.7222 +/- 0.0010 (n=10)   valid (in-sample) 0.7800 +/- 0.0004 (n=10)
row C-F T=2 student + members + learned combiner  [submitted]
  test MRR   0.7320 +/- 0.0010 (n=10)   valid (in-sample) 0.7880 +/- 0.0002 (n=10)
  student alone: test 0.6855 +/- 0.0008 (n=10)   valid 0.7190 +/- 0.0004 (n=10)
ensemble  ten-seed ensemble + members + learned combiner  [submitted]
  full ten-seed ensemble: test 0.7430   valid (in-sample) 0.7998  (one read)
  leave-one-out (9 of 10 seeds), ten reads: test 0.7426 +/- 0.0002 (n=10)   valid (in-sample) 0.7994 +/- 0.0001 (n=10)
  cross-fitted valid, dist_s1: 0.7874   ... dist_s9: 0.7870   ens: 0.7992   rowf_s0: 0.7791 ... rowf_s6: 0.7795
```

Every number in the paper's result tables is one of these lines.

## Tier 2 — split check and re-scoring the released checkpoints (minutes, GPU)

```
cd biokg
echo y | PYTHONPATH=.. uv run python verify.py --skip-models      # downloads ogbl-biokg (2.9 GB), asserts the splits are disjoint
scripts/fetch_checkpoints.sh                                       # sparse_s{0..9}.pt + dist_T2_s{0..9}.pt from release v2.0-two-boards, sha256-checked
PYTHONPATH=.. uv run python verify.py --device cuda --ladder sparse
```

What it printed on the fresh machine (timings from its log):

```
== 1. split disjointness ==
  train: 4,762,678 triples (4,762,678 unique)
  valid: 162,886 triples (162,886 unique)
  test: 162,870 triples (162,870 unique)
  train ∩ valid = 0  [OK]
  train ∩ test = 0  [OK]
  valid ∩ test = 0  [OK]                          (164 s including the download; 913 s on the slow-network box)

fetched 20 checkpoints, checksums: 20 OK           (117 s / 696 s)

== 2. checkpoint reproduction (test MRR vs logs) ==
  seed 0: recomputed 0.8148 vs logged 0.8148  (|d|=0.00000) [OK]
  seed 1: recomputed 0.8158 vs logged 0.8158  (|d|=0.00004) [OK]
  seed 2: recomputed 0.8160 vs logged 0.8160  (|d|=0.00002) [OK]
  seed 3: recomputed 0.8167 vs logged 0.8167  (|d|=0.00002) [OK]
  seed 4: recomputed 0.8162 vs logged 0.8162  (|d|=0.00002) [OK]
  seed 5: recomputed 0.8150 vs logged 0.8150  (|d|=0.00000) [OK]
  seed 6: recomputed 0.8159 vs logged 0.8159  (|d|=0.00001) [OK]
  seed 7: recomputed 0.8158 vs logged 0.8158  (|d|=0.00002) [OK]
  seed 8: recomputed 0.8163 vs logged 0.8163  (|d|=0.00004) [OK]
  seed 9: recomputed 0.8156 vs logged 0.8156  (|d|=0.00003) [OK]
  (these ten lines are from a GTX 1080 Ti on our side, torch 2.6; the 5090 printed the same verdicts)
  ALL 10 CHECKPOINTS REPRODUCE. test MRR 0.8158 +/- 0.0006     (40 s on the 5090)
```

`--ladder dense` does the same for the first ladder (`model_final_seed*.pt`
from release `v1.0-biokg`). The converted sparse checkpoints are re-scored
through the dense code path: seed 0's logged 0.8148 comes back as 0.8148.

## Tier 3 — rebuild one retrieval row from its checkpoint (hours, CPU)

```
ROW=sparse SEEDS="0" DEVICE=cuda scripts/run_retrieval.sh
```

Per seed this runs `cache_scores.py` (model scores on valid and test),
`analogy_member.py` (both splits), the shared `jaccard_member.py` (once,
both splits), `ensemble_weights.py` (held-out estimate on validation
halves) and `freeze_test.py` (weights frozen on full validation, one test
read). On the fresh machine, seed 0 took 8,508 s: Jaccard 44 min per split
(single-threaded Python; the caches are then reused by every seed and
row), analogy about 20 min per split with the GPU for the cosines, the
blend and the read under a minute. The end of `runs/committed_single_s0.log`:

```
weights frozen -> runs/frozen_single_s0.npz
frozen weights on full valid (in-sample): 0.8466

================= COMMITTED TEST RESULT =================
official Evaluator test MRR: 0.8461
hits@1: 0.7871
hits@3: 0.8875
hits@10: 0.9522
```

and `runs/committed_single_s0.json` held `"test_mrr": 0.8460542559623718`,
identical to `results/sparse/committed_single_s0.json` to the last digit;
`runs/pure_single_s0.log` held `HELD-OUT 0.8461`, identical to the shipped
`pure_single_s0.log`. The feature and blend stages are deterministic given
the checkpoint, so a match is exact, not approximate.

## wikikg2 — re-scoring the released models (minutes, GPU, 11 GB of downloads)

```
cd ../wikikg2
../biokg/scripts/fetch_checkpoints.sh wikikg2      # 10 teachers + 7 students, bf16 tables, into wikikg2/teachers/ (1,443 s)
echo y | PYTHONPATH=.. uv run python train_wiki.py --device cuda --eval-only --save teachers/model_wiki_s0.bf16.pt --eval test
echo y | PYTHONPATH=.. uv run python train_wiki.py --device cuda --eval-only --save teachers/model_dist_s5.bf16.pt --eval test
```

The first evaluation includes the dataset download and processing
(1,688 s in total on the fresh machine); the second is a minute. Printed:

```
[test] MRR 0.6678   ...      # teacher 0: equal to wikikg2/results/rowA/final_s0.log
[test] MRR 0.6873   ...      # student 5: equal to wikikg2/results/rowCF/distill_s5.log
```

The bf16 re-save of the tables changes nothing at four decimals. Students
0, 2 and 3 have no surviving checkpoint (deleted by the campaign script
after their reads); their receipts and logs are in `results/rowCF/`.

## The claim checker (H28), from a released model (minutes, GPU)

```
PYTHONPATH=.. uv run python cache_wiki.py --device cuda --split test --models teachers/model_dist_s5.bf16.pt
PYTHONPATH=.. uv run python claim_check.py ens_cache/model_dist_s5.bf16.test.npz
```

Printed (the full per-relation tables are in `results/h28/`):

```
cache ens_cache/model_dist_s5.bf16.test.npz: 598,543 test queries x 2 directions, 500 sampled alternatives each
all            n=1,197,086  AUROC true-vs-uniform 0.9824  true-vs-hard 0.7087
               rank<=1   TPR 0.620   FPR uniform 0.001   FPR hard 0.381
               rank<=3   TPR 0.718   FPR uniform 0.005   FPR hard 1.000
               rank<=10  TPR 0.823   FPR uniform 0.018   FPR hard 1.000
tail queries   n=598,543  AUROC true-vs-uniform 0.9974  true-vs-hard 0.9328
               rank<=1   TPR 0.929   FPR uniform 0.000   FPR hard 0.071
head queries   n=598,543  AUROC true-vs-uniform 0.9516  true-vs-hard 0.3470
               rank<=1   TPR 0.310   FPR uniform 0.001   FPR hard 0.690
```

## What this does not do

Nothing above retrains anything. `scripts/run_campaign_sparse.sh`,
`scripts/run_distill_sparse.sh` and the wikikg2 campaign scripts are in the
repository for that (about 45 GPU-minutes for the ten biokg row-A seeds,
hours per wikikg2 seed); training is seeded but CUDA kernels are not
bit-deterministic, so a retrained checkpoint reproduces its MRR to
seed-level noise, not bit-for-bit.

## Timings on the two fresh machines

| step | box 1 (fast network) | box 2 |
|---|---|---|
| clone + `uv sync` | 129 s | 1,975 s |
| statistics from receipts | 1 s | 1 s |
| biokg download + split check | 164 s | 913 s |
| 20 checkpoints fetched + checksummed | 117 s | 696 s |
| ten row-A models re-scored on test | 40 s | 59 s |
| row B seed 0 rebuilt from the checkpoint | — (box stopped) | 8,508 s |
| wikikg2 checkpoints (11 GB) | — | 1,443 s |
| wikikg2 download + teacher 0 re-scored | — | 1,688 s |
| student 5 re-scored | — | 67 s |

Rented time for both attempts: about 6 GPU-hours at $0.35–0.45 per hour.
