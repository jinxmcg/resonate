# CS1: candidate-side analogy

Status: registered and evaluated 2026-09-06. All six report folds completed;
independent endpoint audit passed. Mean MRR improved from **0.8555176435**
to **0.8582166512** (+0.0026990077). Both preregistered consistent-positive
and useful-gain flags passed. No automatic promotion or submission change.

[Protocol](../../CANDIDATE_RETRIEVAL.md), [runner](../../candidate_retrieval.py),
[tests](../../test_candidate_retrieval.py), [auditor](../../audit_candidate_retrieval.py).

One frozen H35F A, same source-side analogy/Jaccard and exact saved BC1
half-strength blends. Add only candidate-side max/top3 analogy: how similar
is each candidate to the query's known TRAIN answers, excluding a candidate's
own self-match? The two normalized scores are averaged into one new feature.

Within each of BC1's six fixed fit/report folds, select its additional weight
from {0,0.025,0.05,0.1,0.2}, with the existing family/direction and relation/
direction minimum of 2,000 fit queries. Old blend weights remain fixed for
that fold. Zero means no addition. No report labels select weights.

Matched baseline mean MRR is **0.8555176434837662**, already including
retrieval. One primary paired contrast, bootstrap over original triples
with both directions and all three partition repeats grouped together.
All validation is adaptively reused, not a fresh holdout or TEST result.

## Results

All numbers below are validation with retrieval, not model-alone or TEST
scores. Partition seeds change the fit/report split, not the model training
seed. Each partition reports every original triple once, with both directions
kept together; predictions from the three partitions are not ensembled.

| Two-way partition | Existing pipeline MRR | + Candidate-side MRR | Paired delta | Top-1 recovered / lost |
| --- | ---: | ---: | ---: | ---: |
| Seed 0 | 0.8555228741 | 0.8582454468 | +0.0027225727 | 2,462 / 1,372 |
| Seed 1 | 0.8554690595 | 0.8581787915 | +0.0027097319 | 2,515 / 1,436 |
| Seed 2 | 0.8555609968 | 0.8582257154 | +0.0026647186 | 2,500 / 1,439 |
| Mean | 0.8555176435 | 0.8582166512 | +0.0026990077 | — |

Primary descriptive paired 95% bootstrap interval: **[+0.0024912022,
+0.0029036106]**, resampling original triples after averaging directions and
partition repeats within each triple (2,000 replicates, seed 3641). Reusing
VALID adaptively limits this interval: it is not fresh-holdout evidence or
training-seed robustness. No official leaderboard placement follows from it.

| Report fold | Existing pipeline MRR | + Candidate-side MRR | Delta |
| --- | ---: | ---: | ---: |
| Seed 0 / fold 0 | 0.8550978918 | 0.8578497366 | +0.0027518448 |
| Seed 0 / fold 1 | 0.8559453176 | 0.8586387930 | +0.0026934754 |
| Seed 1 / fold 0 | 0.8550656544 | 0.8578087961 | +0.0027431417 |
| Seed 1 / fold 1 | 0.8558723755 | 0.8585487050 | +0.0026763295 |
| Seed 2 / fold 0 | 0.8549642800 | 0.8574618057 | +0.0024975257 |
| Seed 2 / fold 1 | 0.8561597833 | 0.8589922747 | +0.0028324914 |

| Direction | Existing pipeline MRR | + Candidate-side MRR | Delta |
| --- | ---: | ---: | ---: |
| Tail | 0.8498849188 | 0.8544681690 | +0.0045832502 |
| Head | 0.8611503682 | 0.8619651334 | +0.0008147652 |

Mean Hits@1 rises from 0.7998385374 to 0.8031435073; Hits@10 from
0.9551393817 to 0.9562035207. The tail-heavy gain supports complementarity
with the existing holder-side features, not a need for a second trained model.

Global beta selected zero in every fold. Only 7–17 of 102 directed relation
groups receive a nonzero beta, showing why the predeclared relation-aware
selection matters; this is not evidence for applying retrieval uniformly.

| Report fold | beta=0 | 0.025 | 0.05 | 0.1 | 0.2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Seed 0 / fold 0 | 90 | 8 | 4 | 0 | 0 |
| Seed 0 / fold 1 | 95 | 3 | 4 | 0 | 0 |
| Seed 1 / fold 0 | 85 | 13 | 3 | 1 | 0 |
| Seed 1 / fold 1 | 94 | 4 | 4 | 0 | 0 |
| Seed 2 / fold 0 | 89 | 9 | 4 | 0 | 0 |
| Seed 2 / fold 1 | 89 | 8 | 5 | 0 | 0 |

The new features alone score 0.6376353822 (max), 0.6459036240 (top-three)
and 0.6494387632 (fixed normalized pair). Those are deliberately incomplete
scorers, not the combined pipeline's 0.8582166512 MRR.

## Verification and boundaries

All 60 synthetic tests passed before execution. The runner checked sampled
features in all 102 directed relations against direct embedding dots, with
exact candidate-permutation/chunk invariance on the cosine cache. All six
baseline report rank vectors exactly match BC1; frozen model tensor hashes
and pinned source/input hashes are unchanged. Only TRAIN, VALID and raw node
counts were opened from the dataset. No optimizer, gradients or model updates;
LR=0. The independent endpoint replay passed: 105 representative queries
across all 102 directed relations rebuilt from TRAIN (maximum direct-feature
error 5.3644e-7), all normalization/pair scores exact, all six fit-only beta
recipes and baseline/treatment report ranks reproduced exactly, coverage,
metrics and clustered interval reproduced, and source/input/artifact hashes
verified. Raw features were independently rebuilt on those representative
queries, not exhaustively regenerated. [Endpoint audit](s0_retry1/endpoint_audit.json).

CPU-only because the local 1080 Ti is occupied by another workload. No
interference with it, new model training, external checkpoint, broader
blend search, path feature, TEST, full-VALID refit or submission update.
Stopped after the six folds and independent audit; no automatic next run.

Successful CPU run: 963.3 seconds through feature construction, 1,026.4
through summary, 1,054.9 seconds including final artifact hashing (~17m35s).
This is a cached research evaluation, not a benchmark of final submission
latency. [Summary](s0_retry1/summary.json), [folds](s0_retry1/folds.json),
[runner audit](s0_retry1/audit.json), [prerun receipt](s0_retry1/prerun.json).

## Technical restart

The initial synthetic run exposed a list-versus-array test-fixture mismatch
in the old holder helper; fixed before real-data execution. No old scorer
or pinned prior campaign source was modified.

The original `s0/` prereceipt, partial features and changed source snapshots
are retained with [interruption record](s0/interruption.json). No new MRR was
evaluated before the technical stop. Canonicalize type arrays once per build;
all scoring, beta selection, partitions and decisions remain unchanged.
All 60 tests passed again, including a canonicalized-type-array regression
check. Archived original source hashes match the interrupted prereceipt.
The retry reproduces both raw features exactly for all 7,164 queries across
the original five completed directed relations, all 501 candidates per
query, maximum error 0.0: [technical parity](s0_retry1/restart_parity.json).
Successful-run artifacts are saved under `s0_retry1/`.
