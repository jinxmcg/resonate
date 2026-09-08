# H35D: rank fusion does not improve these frozen teachers

Completed 2026-09-06 on the local GTX 1080 Ti. Full official VALID:
162,886 triples, both directions, 325,772 directed queries, 500 supplied
negatives each. This is a diagnostic, not a new single-model result or a
RelEns reproduction. No training, optimizer, gradients, learned weights,
TEST access, new model checkpoint or inference deployment change.

## Primary result

| Frozen scorer | MRR | Hits@1 | Hits@10 |
| --- | ---: | ---: | ---: |
| Original distilled student | 0.8321748668 | 0.7676565205 | 0.9471839200 |
| Ten teachers, equal mean raw logits | 0.8427075057 | 0.7817123632 | 0.9511959284 |
| Same teachers, equal mean candidate ranks | 0.8400257236 | 0.7752876245 | 0.9500233292 |

Rank-minus-logit fusion: **−0.0026817821 MRR**, paired-whole-triple bootstrap
95% **[−0.0029066557, −0.0024584540]**, 1,000 replicates, seed 3512.
The preregistered +0.001/positive-lower-bound gate failed. **Do not advance
rank-target distillation on the strength of this experiment.**

Rank fusion improved positive ranks on 13,764 queries, worsened 16,653, and
tied on 295,355 versus mean logits. It recovered 1,865 top-1 predictions but
lost 3,958. These are query counts, not inferred from `1 - MRR`.

## Teacher-to-student gap

Mean logits exceed the original student by **+0.0105326389 MRR**; descriptive
paired 95% interval **[+0.0101020083, +0.0109863656]**. Mean fusion improves
33,220 positive ranks, worsens 21,033, and leaves 271,519 equal. It recovers
10,304 student-missed top-1 queries and loses 5,725 student-correct ones.
Thus the teacher target has useful information the current student does not
fully retain, but it also makes errors the student avoids. This gap is not a
guaranteed gain from more KD, nor a bound on a newly trained architecture.

| Query direction | Student MRR | Mean-logit MRR | Rank-fusion MRR |
| --- | ---: | ---: | ---: |
| Tail | 0.83261683 | 0.84146619 | 0.83955566 |
| Head | 0.83173291 | 0.84394882 | 0.84049578 |

| Relation family | Queries | Student MRR | Mean-logit MRR | Rank-fusion MRR |
| --- | ---: | ---: | ---: | ---: |
| disease-protein | 7,554 | 0.60566954 | 0.65912989 | 0.65532706 |
| drug-disease | 534 | 0.34821893 | 0.47292097 | 0.43502928 |
| drug-drug | 62,270 | 0.68600231 | 0.69823314 | 0.69457401 |
| drug-protein | 13,078 | 0.84622420 | 0.86558888 | 0.86161442 |
| drug-sideeffect | 17,238 | 0.25578200 | 0.27182592 | 0.26944316 |
| function-function | 79,622 | 0.96461346 | 0.96646083 | 0.96514154 |
| protein-function | 86,362 | 0.85840217 | 0.87304382 | 0.86878873 |
| protein-protein | 59,114 | 0.96773791 | 0.97209906 | 0.97157023 |

These slices are descriptive, not used to fit relation/direction weights.
Protein-function and drug-drug have both substantial query counts and a
remaining student gap; drug-sideeffect remains difficult for the teachers
too. A large relative gain on the 534 drug-disease queries should not be
confused with a large global improvement.

Individual teacher MRRs, seeds 0 through 9:
`0.81545893, 0.81623692, 0.81681703, 0.81761248, 0.81659133,
0.81548609, 0.81631826, 0.81617908, 0.81684975, 0.81635386`.
Across all 45 teacher pairs, mean positive-rank agreement is 74.51498%;
mean disagreement on whether the positive is top-1 is 10.50927%. This is
not candidate-winner disagreement or a claim of distinct learned geometry.

## Interpretation and next decision

- Equal rank aggregation does not reveal an improvement over the current
  logit aggregation for these ten same-family teachers. Do not extrapolate
  this to heterogeneous models, learned relation weights, or all rank losses.
- Keep mean-logit KD as the control. Better compression has a measured
  +0.01053 teacher/student gap to investigate; that is much smaller than the
  leaderboard leader's published result. Changing aggregation alone has not
  closed the gap here.
- If proceeding with the user's internal-expert idea, separately preregister
  a B branch with genuinely different information or scoring geometry and
  initialization, retain KD, and compare with an equally trained single-bank
  control. H35C showed that a perturbed duplicate bank co-adapting on the same
  inputs/objective stayed nearly identical. H35D does not identify which new
  architecture will work; no context or mixed-geometry training was launched.

## Correctness, provenance and compute

- [Protocol](../../H35D.md) fixed before real-data scoring; no outcome-driven
  changes. [Runner](../../teacher_fusion_diagnostic.py) uses equal weights,
  average ties within each teacher, integer doubled-rank sums, and the
  official OGB average-tie evaluator. All 501 candidate positions use one
  identical scoring function. No positive-position advantage.
- **78 tests passed**, including 10 new synthetic diagnostic tests. Full-size
  synthetic GPU smoke used 11 models, batch 128 and 501 candidates; exact
  candidate-permutation parity, unchanged states and no gradients passed.
- Checkpoint/data/release/reference hashes verified before and after the run.
  All eleven model state digests remained identical; parameters were frozen,
  gradients absent. All archived H35C source hashes remained unchanged.
- Recomputed student ranks match all 325,772 archived H35 query ranks exactly.
  Independent post-run checks validated source and query hashes, array shape,
  triple/direction order, finite half-integer rank range and summary MRRs.
- Evaluation took **68.278 seconds**, excluding loading, hash audits and
  report generation. Peak allocated GPU memory **1,364,512,768 bytes**
  (about 1.27 GiB). fp32/complex64, no AMP/TF32. The rented GPU was untouched.
- `"lr": null` in machine-readable logs means **LR: N/A**: this is evaluation,
  so no learning rate exists. It is not an error or a stopped optimizer.
- Query archive SHA256:
  `df0a5deff600f0e4d55c49b8da9f740191b1c16dbdd227a1af99c085ef909832`.

Artifacts: [summary](campaign_s0/summary.json), [pre-run receipt](campaign_s0/prerun.json),
[state audit](campaign_s0/audit.json), [progress](campaign_s0/progress.jsonl),
[per-query ranks](campaign_s0/queries.npz). No full candidate-score cache.
VALID has been reused adaptively across experiments; intervals quantify
query variability, not seed uncertainty or untouched-holdout generalization.

Reproduce with the recorded environment from the repository root, selecting
a fresh output directory (existing runs are never overwritten):

```sh
/mnt/geocore/geocore/.venv/bin/python -m biokg.teacher_fusion_diagnostic --out biokg/results/h35d/reproduction
```
