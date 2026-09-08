# RF1: rarity-weighted Jaccard and support-aware retrieval

Status: evaluation completed 2026-09-06 on the local GTX 1080 Ti;
independent endpoint audit passed. Neither primary intervention is
promoted. The predeclared half-strength control is a promising separate
blend-selection result, not an automatically adopted submission.
41 synthetic tests passed: eleven RF1, ten FC1, seven H34 and thirteen
mixed-operator tests. Frozen H35F A only; no training, teacher, external
checkpoint, TEST access or submission change.

[Fixed protocol](../../RETRIEVAL_FOLLOWUPS.md), [runner](../../retrieval_followups.py),
[tests](../../test_retrieval_followups.py), [auditor](../../audit_retrieval_followups.py).

The baseline is FC1's improved model plus its ordinary analogy/Jaccard
pipeline, report-half MRR 0.8540538580. Two independent interventions:

1. Replace only Jaccard with TRAIN document-frequency-weighted overlap.
2. Keep ordinary metrics, but return retrieval weight to A for candidates
   lacking strong, repeated above-background TRAIN-holder support.

One descriptive fixed-half-strength control distinguishes candidate support
from the simpler effect of weakening retrieval. No combined rarity+support
arm. Same fit/report mask, finite weight selection and candidate-symmetric
official evaluation. Two primary comparisons use Bonferroni-adjusted
97.5% paired intervals; advance flag requires +0.001 and lower bound > 0.
No automatic promotion, extra seed, model training or TEST run.

All old model scores and ordinary features come from verified own FC1
VALID caches; new graph statistics use TRAIN only. Reused validation is
adaptive development data, not a pristine holdout or a leaderboard result.

Artifacts: [receipt](s0/prerun.json), [progress](s0/progress.jsonl).

## Matched report-half results

Same 81,199 VALID report triples / 162,398 directed queries as FC1. The
separate 81,687 fit triples select the finite mixtures; no full-VALID refit.
All model parameters and baseline scores are unchanged.

| Pipeline | Report MRR | Delta from baseline |
| --- | ---: | ---: |
| Unchanged improved-A pipeline | 0.8540538580 | — |
| 1. Rarity-weighted Jaccard | 0.8541798047 | +0.0001259467 |
| 2. Support-aware ordinary retrieval | 0.8510670133 | -0.0029868447 |
| Descriptive fixed-half-strength + fit-only mixture selection | **0.8550978918** | **+0.0010440338** |

The two primary contrasts use Bonferroni-adjusted 97.5% whole-triple
paired-bootstrap intervals (2,000 replicates each):

- Rarity: **[-0.0000812153, +0.0003217361]**, includes zero and misses +0.001.
  Top-one recoveries/losses against baseline: 385 / 349, net +36.
- Support: **[-0.0035614879, -0.0024054833]**, below zero. Top-one
  recoveries/losses: 1,890 / 2,597, net -707.

Both advance flags are false. These reject the exact registered variants
as upgrades, not every possible rarity or reliability mechanism. Do not
combine them or select relation-specific switches from the report results.

The half-strength control has a **descriptive 95%** interval
**[+0.0006450757, +0.0014571628]**, with 1,847 top-one recoveries and
1,641 losses (net +206). Support minus half-strength is -0.0040308784,
95% interval [-0.0045000231, -0.0035512751]. The control was predeclared,
but was not one of the two primary promotion tests. A new confirmation
protocol/seed check is needed before adoption; this is one adaptively reused
development split and not an untouched holdout or leaderboard measurement.

## The useful finding: mixture selection, not simply weaker retrieval

Holding the old baseline recipe fixed gives:

| Fixed-baseline-weights diagnostic | Report MRR |
| --- | ---: |
| Rarity-weighted Jaccard | 0.8542285655 |
| Support-aware retrieval | 0.8491923937 |
| Half-strength retrieval | **0.8497543394** |

Simply halving retrieval under old weights loses -0.0042995185. The winning
0.8550978918 result requires fit-only recipe reselection after the fixed
half-strength transformation. It is **not** evidence that lowering every
retrieval weight monotonically improves performance.

If A is normalized model score, R_j are the four normalized retrieval
members, and selected weights sum to one, the half-strength blend is

`w_A*A + sum_j w_j*(0.5*A + 0.5*R_j)`

which is an ordinary five-feature linear blend with effective weights
`w_A + 0.5*sum_j w_j` on A and `0.5*w_j` on each R_j. Algebraically no
new information, feature or model capacity is added. Finite-precision
rearrangement can introduce rounding differences; no export was substituted.

The global fallback illustrates why this is not just "trust A more":

| Effective global weight | Baseline | Half-strength + reselection |
| --- | ---: | ---: |
| A | 0.583175 | 0.533775 |
| Analogy max | 0.032533 | 0.095114 |
| Analogy top-three | 0.160707 | 0.093864 |
| Jaccard max | 0.044867 | 0.147398 |
| Jaccard top-three | 0.178717 | 0.129849 |

These are global fallback weights, not the weights of every relation.
The transformation changes individual-member fit MRRs and therefore the
mixtures proposed by the finite top-member/softmax search. The result is
evidence that its candidate set leaves useful blends unexplored. It does
not establish how much additional gain remains or promise second place.

Fit-half MRRs: baseline 0.8553192195, rarity 0.8553735825, support
0.8518164856, half-strength 0.8561104094. Report them separately from the
report-half scores; do not use them as held-out performance.

Support gate averages over all provided candidate columns were 0.20430
for analogy and 0.16532 for Jaccard; zero-gate fractions 60.87% / 70.06%.
This is substantial candidate-specific attenuation, not an exact match to
the 0.5 control. It combines support filtering with changed effective
weights; its failure does not isolate a single cause. The statistics do
not identify positive-answer support rates.

Descriptive slices vary: rarity improves disease-protein but loses on
drug-drug; support loses substantially on disease-protein/drug-protein tail
queries while gaining on function-function; the control has both gains and
losses. Full slice tables are retained in the summary, not used to retrofit
another rule on the same report labels.

## Verification and compute

- All 41 synthetic tests passed before real execution. They cover weighted
  Jaccard against brute force, uniform-weight equivalence, TRAIN population
  isolation, support formula/pooling, zero/one gates, candidate permutation,
  duplicate/self/missing holders, list metadata and fit/report isolation.
- All 102 directed relations passed real candidate-permutation checks.
  Recomputed unchanged features on their audit batches match FC1 **exactly**,
  maximum error 0.0. Baseline full rank vector also matches FC1 exactly.
- Model tensor hashes unchanged; gradients absent; 0 updates, LR 0.
  Source/input hashes unchanged. Dataset opens limited to TRAIN, VALID and
  raw node counts. No external trained artifacts, teacher or TEST use.
- Independent saved-record audit passed: source/artifact hashes verified;
  baseline ranks, normalization/support transforms, fit-only recipes, all
  saved ranks and summary metrics/bootstrap intervals reproduced exactly.
  This audit did not deserialize dataset splits.
- One repeated TRAIN edge was deduplicated in each direction. IDF weights
  on observed TRAIN targets ranged from 1.24661 to 11.01315. No VALID edge
  contributes to document frequencies or similarity backgrounds.
- Feature generation: 1,642.15 s (~27.37 min). Through all fixed comparisons
  and bootstrap: 1,802.68 s (~30.04 min), excluding initial loading/hash
  verification, final artifact hashing and independent audit. Peak PyTorch
  allocation 247,666,176 bytes (~0.23 GiB), not total CPU/cache memory.
  The local GPU has no remaining compute process.
- Existing checkpoints, baseline recipe and submissions were not modified.
  Recommended next step is confirmation of the exact predeclared control
  recipe, then a separately preregistered finite blend-candidate comparison,
  not a continuation of the unsuccessful support gate or an automatic sweep.

[Summary](s0/summary.json), [graph/data/state audit](s0/audit.json),
[independent endpoint audit](s0/endpoint_audit.json),
[rarity recipe](s0/rarity_recipe.json), [support recipe](s0/support_recipe.json),
[half-strength recipe](s0/half_strength_recipe.json), [ranks](s0/ranks.npz).
Generated features/gates and normalized/interpolated members are retained.

```sh
/mnt/geocore/geocore/.venv/bin/python -m biokg.audit_retrieval_followups biokg/results/retrieval_followups/s0
```
