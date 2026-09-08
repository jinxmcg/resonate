# KD confidence versus ordering: temperature-only change not prioritized

Completed 2026-09-06 on the local GTX 1080 Ti. All A/teacher weights stayed
unchanged. No validation, test, external trained artifact or rented GPU use.

[Protocol](../../KD_CALIBRATION_DIAGNOSTIC.md),
[runner](../../kd_calibration_diagnostic.py),
[tests](../../test_kd_calibration_diagnostic.py),
[summary](s0/summary.json), [receipt](s0/prerun.json),
[read-only auditor](../../audit_kd_calibration.py).

## Result

The previous gradient diagnostic showed strong CE/KD opposition on the
temperature parameter. This follow-up shows that **a scale mismatch explains
very little of the remaining KD discrepancy at these trained snapshots**.
A large scalar gradient is not itself a large reducible loss or a promise
of improved ranking.

Same 65,536 sampled TRAIN queries for both models, 4,096 shared typed
negative draws per batch. A fixed canonical-triple hash assigns 32,739
queries to scalar fitting and 32,797 to reporting; repeats and reverse
queries stay together. All queries are from TRAIN and already exposed to
the models. "Report" means excluded from scalar fitting, not unseen by A
or the teachers. The sample covers 36 of 102 directed relations.

| Report-TRAIN diagnostic | Original A | Improved A |
| --- | ---: | ---: |
| Global positive scale fitted on fit-TRAIN | 0.993877 | 0.993944 |
| Original T-squared KD | 0.06415925 | 0.06128514 |
| KD after applying the frozen fitted scale | 0.06390177 | 0.06101873 |
| Relative KD reduction, global scale | **0.4013%** | **0.4347%** |
| KD with best separate scale per report query | 0.06231337 | 0.05952910 |
| Relative reduction, optimistic per-query fit | **2.8770%** | **2.8654%** |
| Mean centered teacher/student score cosine | 0.95196 | 0.95296 |
| Teacher/student top-score-set overlap | 47.111% | 47.480% |

Global calibration gives similarly small reductions on the fit subset:
0.4253% for original A and 0.4363% for improved A. Both optima are interior
to the fixed [1/16,16] range. Final mean fitting derivatives have absolute
value below 1e-6 (before multiplying by T squared).

The per-query fit is an **optimistic same-query diagnostic**, not a
generalizing calibration policy. For improved A, its median scale is
0.99158, p10/p90 0.97260/1.00866; no fitted scale hits a bound. Even this
flexibility leaves about 97.13% of the original KD divergence. The
remaining divergence includes score-gap/shape differences as well as
ordering differences; it is not "97% ranking error".

Global and per-query positive scaling preserve the top-score sets of every
report query. The comparison used float64 multiplication to avoid creating
fp32 rounding ties; scoring, fitting and KD computations remained fp32.
No scalar was written into a model or used on validation/test inputs.

## What the ranking disagreement does and does not mean

Improved A and the teacher mean disagree at the top on 17,225/32,797 report
queries despite a high average score cosine. Thus distribution similarity
does not imply identical choices near the maximum.

These are **large, unfiltered TRAIN candidate pools**, not official OGB
evaluation pools. Sampled duplicates and other known-positive edges are
retained as in the current training recipe. A teacher's preferred candidate
is not ground truth, and top-score disagreement is not the fraction of
student mistakes or recoverable validation errors.

In particular, 15,041 of those 17,225 disagreements occur where neither
model puts the sampled TRAIN positive at the top. Some competing answers
may be other valid graph facts; this run did not classify them. On the
report subset the sampled positive is among the top-score set for 12.33%
of improved-A queries and 12.55% of teacher-mean queries. Teacher-only versus
student-only positive-top counts are 1,171 versus 1,100. Do not compare
these rates with the official validation MRR or infer a large teacher win.

## Decision and a separate next comparison

Follow-up: the proposal below was subsequently approved, preregistered and
completed as [CFKD1](../candidate_focus/README.md). It was not promoted:
focused full-VALID MRR 0.8339914 versus control 0.8338694 and starting A
0.8340991. The text below preserves the diagnostic's original proposal,
not the current launch status. The separately user-approved existing-feature
comparison is now complete as [FC1](../feature_compare/README.md).

- Keep A's current loss, temperature and trajectory unchanged. The results
  do not motivate prioritizing a new independent global KD temperature.
  They do not rule out different earlier-training dynamics or adaptive
  confidence mechanisms; neither was tested.
- Do not restart B, replace mean-logit teachers with rank fusion, or claim
  that deleting KD would help. H35D already found mean-rank fusion weaker
  than the current teacher mean; this result does not overturn that test.
- The next *training hypothesis* is a **candidate-focused KD addition**:
  keep CE + full-pool KD + trajectory, and test an extra conditional KD term
  on the union of the teacher's and student's leading candidates. It uses
  the same own-teacher mean logits, not a new ensemble or rank-fused target.
  This may emphasize decision-relevant differences, but the top-choice
  disagreement alone is not evidence that the extra term will generalize.

Suggested isolated screen, **proposed, not launched or preregistered**:
two copies of improved A; same TRAIN stream and 5,000 updates; unchanged
loss control versus the same loss plus a fixed-weight conditional KD term.
Candidate subset: top 32 from each scorer, include cutoff ties and the known
TRAIN positive; detached selection; T=2 in both KD terms. Keep architecture,
teacher mean, batch/negatives, trajectory, optimizer and schedule identical.
An initial explicit design would use added weight 0.25, batch 2,048,
4,096 negatives, fresh Adam A LR 1e-4 with a common 5,000-step cosine and
global clip 1.0. These are proposed fixed settings, not VALID-fitted values.

Before that screen: verify candidate-order/tie invariance, detached targets,
loss/gradient correctness and standalone checkpoint compatibility; record
the final protocol before execution. Compare fixed endpoints on VALID only,
both directions, with a predefined minimum gain and uncertainty reporting.
No TEST, automatic promotion, teacher change or hyperparameter sweep. This
would test the complete added-loss intervention, not isolate emphasis from
increased effective KD weight. Additional seeds/controls are needed for a
stronger mechanistic claim. Existing feature-pipeline comparison is still
separately pending.

## Audit and compute

- 25 synthetic tests passed, including known-scale recovery, fit/report
  isolation, canonical reciprocal splitting, exact retained-KD equivalence,
  positive-scaling invariance, ties and reversed-ranking examples.
- All two A and ten own-teacher tensor hashes matched before and after;
  gradients were disabled and absent, with zero model updates. All input
  and source hashes matched. Data-open guard recorded only official TRAIN
  and raw node counts; no dataset constructor or processed cache was used.
- Saved row arrays reproduce the summary exactly. No large teacher/student
  score matrices were persisted: they existed only in GPU memory. Saved
  output consists of diagnostic statistics, fitted scalars, hashes and logs.
- An initial independent check with absolute tolerance 1e-6 found small
  oracle-above-baseline differences in 53 original-A and 45 improved-A rows.
  The maximum is 3.8147e-6, consistent with fp32 reduction-layout rounding
  between contiguous baseline and indexed oracle batches. The raw records
  were retained unchanged. The auditor allows 1e-5 and explicitly reports
  these differences; their magnitude does not alter the conclusion.
- Score caching, scalar fits and row diagnostics: **5.33 seconds**, excluding
  model/data loading and endpoint hashing. Peak PyTorch allocation
  4,684,101,120 bytes (~4.36 GiB). The 1080 Ti has no remaining compute job.
- TRAIN-stream SHA256:
  `61de338a890f1934ab7d26d11f0dcff547538d1a00528e47b4054c84934760e2`.

Read-only saved-record audit (no split/checkpoint deserialization):

```sh
/mnt/geocore/geocore/.venv/bin/python -m biokg.audit_kd_calibration biokg/results/kd_calibration_diagnostic/s0
```
