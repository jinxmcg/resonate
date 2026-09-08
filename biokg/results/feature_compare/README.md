# FC1: original versus improved A with matched retrieval

Status: completed 2026-09-06 on local GTX 1080 Ti; independent saved-record
audit **passed**. Small positive gain, below the predefined
+0.001 useful-gain threshold. No submission change or next run launched.
29 synthetic tests passed before first execution. That launch stopped before
the first scored query because raw type metadata was a list, not an array.
The list-to-array fix and regression test leave the protocol unchanged;
the failed [receipt](s0/prerun.json) and [failure record](s0/failure.json) are
retained. All 30 tests passed after the fix; retry uses `s0_retry1`.
No model training, teachers, external checkpoints,
TEST, new feature definition or submission change.

[Fixed protocol](../../FEATURE_COMPARE.md),
[runner](../../compare_feature_pipeline.py),
[tests](../../test_feature_pipeline.py),
[saved-record auditor](../../audit_feature_pipeline.py).

Primary comparison: original k=12 student versus existing H35F improved k=12
A, each using its own direct score and two analogy features, plus identical
TRAIN-only Jaccard features. Same full VALID queries, paired fit/report mask,
finite hierarchical recipe selection and official average-tie ranks.
Selection uses fit queries only; report queries do not select weights.

Secondary component swaps freeze the original selected weights to separate
changes in direct scoring and analogy geometry. Crossed snapshots are
diagnostic only, not proposed single-model submissions. FP32 and official
ties differ from historical fp16/pessimistic-tie selection; this is a matched
new comparison, not an exact reproduction of the historical published score.

Artifacts: [retry receipt](s0_retry1/prerun.json), [progress](s0_retry1/progress.jsonl).

## Matched result

Same 81,199 report triples / 162,398 directed queries. The other 81,687
triples / 163,374 queries select finite recipes, with both directions of
each triple kept together. No full-VALID refit. These are repeatedly reused
development-validation data, not an untouched holdout or a TEST estimate.

| Report-half MRR | Original A | H35F improved A | Improved minus original |
| --- | ---: | ---: | ---: |
| Model alone | 0.8317775870 | 0.8337635885 | +0.0019860015 |
| Five features, fixed uniform blend | 0.8404697748 | 0.8410802952 | +0.0006105204 |
| Five features, fit-selected hierarchical blend | **0.8535134998** | **0.8540538580** | **+0.0005403582** |

Primary 95% whole-triple paired-bootstrap interval: **[+0.0002190631,
+0.0008650384]**. The improved pipeline recovers 1,351 top-one answers and
loses 1,260 relative to original, net +91. This is evidence of a small
positive effect in this paired development comparison, not seed robustness.
It does not meet the predeclared +0.001 useful-gain flag; there is no
automatic promotion or confirmation run.

Fit-half pipeline MRR: original 0.8542206204, improved 0.8553192195.
Do not report these fit scores as held-out performance. The raw model-only
full-VALID ranks match the archived checkpoints **exactly, every query**:
0.8321748668 original and 0.8340991235 improved. Those full-VALID scores
are separate from the report-half numbers above.

## What changed after training?

The feature pipeline adds +0.0217359128 to original A and +0.0202902695 to
improved A on the same report queries. Only about 27% of the +0.001986
model-only gain survives in the complete-pipeline comparison. This ratio
describes net metric changes; it is not the fraction of examples learned.

Fixed original fit-selected weights, no further selection:

| Attribution diagnostic | Report MRR | Delta from original pipeline | 95% paired interval |
| --- | ---: | ---: | --- |
| Swap improved direct score only | 0.8539664882 | +0.0004529884 | [+0.0001543693, +0.0007891718] |
| Swap improved analogy only | 0.8534452343 | -0.0000682655 | [-0.0001755940, +0.0000523809] |
| Swap both, retain original weights | 0.8539974194 | +0.0004839196 | [+0.0001637310, +0.0008139624] |

Most of the measured gain comes from direct scoring, not better analogy.
The analogy-only interval includes zero, so this is not a demonstrated
geometry regression. Re-selecting the improved recipe adds only +0.0000564
over reusing original weights. Swaps are non-additive rank diagnostics;
the first two require two snapshots and are **not submission candidates**.

Post-hoc descriptive overlap check, using saved report ranks only: the
improved plain model newly puts 2,110 queries at rank one; 1,181 (56%) were
already rank one in the original feature pipeline. Of the other 929, 479
become new pipeline top-one successes. This supports overlap between what
extra training learns and what existing retrieval already supplies; it
does not prove the complete cause of the smaller MRR gain. No report labels
were used to alter this experiment or construct a new training target.
[Recorded counts](s0_retry1/posthoc_overlap.json).

Descriptive family/direction differences include drug-drug +0.000989 tail /
+0.000998 head, protein-function +0.001871 head / -0.000076 tail, and
drug-sideeffect +0.001879 head / -0.000274 tail. These unadjusted slices are
not new selection criteria or a reason to retrofit switches on report data.

## Verification, compute and decision

- 30 synthetic tests passed: ten FC1, seven H34 and thirteen mixed-operator.
  The initial list-type failure and unchanged-design retry remain recorded.
- Both 27,124,129-parameter single models stayed byte-identical; gradients
  were disabled and absent. No optimizer, teacher, B, ensemble at inference,
  external trained artifact or TEST use. LR 0 denotes frozen evaluation.
- Actual dataset opens: raw node counts, TRAIN and VALID only. All source
  and input hashes matched after execution. A source-query candidate
  permutation check passed for every directed relation; synthetic checks
  also cover duplicate candidates, missing/self holders, set semantics,
  official ties and fit/report isolation.
- Both archived full-VALID model rank vectors reproduced exactly, with zero
  differing ranks and zero MRR discrepancy. The same TRAIN graph is shared
  by both feature pipelines; its single duplicate edge is removed once in
  each direction.
- The independent saved-record auditor verified all pinned source/artifact
  hashes, regenerated every normalization and feature rank, independently
  reselected the exact fit-only recipes, and reproduced all pipeline/swap
  ranks, summary metrics and bootstrap intervals. It loaded no dataset
  splits. [Endpoint audit](s0_retry1/endpoint_audit.json).
- Feature generation: 811.04 seconds. Through normalization, finite recipe
  selection, report scoring and bootstrap: 879.00 seconds (~14.65 minutes),
  excluding initial loading and final file hashing/independent audit.
  Peak PyTorch allocation 409,551,872 bytes (~0.38 GiB). The GPU is free.
- Keep H35F as the stronger existing A, while recording its small measured
  feature-pipeline gain. Do not replace the historical submission or claim
  second place from this development result. Future model changes should
  be judged with this full-pipeline comparison, not only plain-model MRR;
  the useful research question is additional information beyond what the
  current TRAIN features already recover. No next training run was started.

[Summary](s0_retry1/summary.json), [training/data/graph audit](s0_retry1/audit.json),
[original recipe](s0_retry1/original_recipe.json),
[improved recipe](s0_retry1/improved_recipe.json),
[query ranks](s0_retry1/ranks.npz), [metadata](s0_retry1/metadata.npz).
Raw and normalized fp32 features are retained for reproduction.

```sh
/mnt/geocore/geocore/.venv/bin/python -m biokg.audit_feature_pipeline biokg/results/feature_compare/s0_retry1
```
