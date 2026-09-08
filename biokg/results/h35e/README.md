# H35E: private-geometry pilot

Status: completed, 2026-09-06. **No promotion.** All three fixed arms finished
5,000 updates. No follow-up training was launched.

Three fixed arms, each trained for exactly 5,000 matched updates from the
original BioKG distilled student: single A, A + private dot-product B, and
A + identical-size private L1-distance B. This is a finite warm-start screen,
not a from-scratch comparison, leaderboard submission or inference ensemble.

[Fixed protocol](../../H35E.md), [model](../../mixed_operator.py),
[runner](../../train_mixed_operator.py), [tests](../../test_mixed_operator.py),
[endpoint auditor](../../audit_mixed_operator.py).

## Full-validation outcome

162,886 official VALID triples, both directions, 325,772 directed queries,
all 500 supplied negatives. These are the fixed endpoint scores, not the
smaller progress-probe scores or best-probe checkpoint selection.

| Endpoint | MRR | Hits@1 | Hits@10 |
| --- | ---: | ---: | ---: |
| Original frozen student | 0.8321748668 | 0.7676565205 | 0.9471839200 |
| Equally trained A-only control | 0.8327347614 | 0.7684392765 | 0.9473435409 |
| A + private dot-product B | 0.8326823323 | 0.7683533269 | 0.9472821483 |
| A + private L1-distance B | 0.8326803530 | 0.7683533269 | 0.9472852179 |

| Preregistered contrast | MRR delta | Adjusted 98.3333% interval |
| --- | ---: | --- |
| Distance minus dot | -0.0000019793 | [-0.0000906463, +0.0000861238] |
| Distance minus trained single | -0.0000544085 | [-0.0001637907, +0.0000415771] |
| Distance minus original frozen | +0.0005054861 | [+0.0001863826, +0.0008231588] |

All-three promotion requirements failed: distance did not beat either trained
control, and none of the deltas reached +0.001. Bootstrap resampled whole
triples with both directions paired, 1,000 replicates, seed 3514, with
Bonferroni adjustment for the three declared promotion comparisons.
Dot-minus-single was also negative: -0.0000524291; descriptive 95% interval
[-0.0001463034, +0.0000438081]. These intervals describe query variability,
not training-seed uncertainty or protection from adaptive VALID reuse.

The single arm exactly reproduced **all H35C single-endpoint ranks**. Both
new B arms were occasionally ahead on the 1,000-triple probe; that did not
translate to a full-VALID improvement. No probe-selected endpoint was used.

## Are A and B actually different?

Yes in their scores; no useful complementarity was demonstrated.

| Diagnostic at endpoint | Dot-B model | Distance-B model |
| --- | ---: | ---: |
| A-alone MRR within that model | 0.83274204 | 0.83276333 |
| B-alone MRR | 0.01290680 | 0.01353573 |
| Mean within-query A/B candidate-score correlation | -0.00072876 | +0.00150616 |
| Mean gate, VALID-query weighted | 0.04054663 | 0.04007015 |
| Private logit scale `exp(log_tau_b)` | 0.35592532 | 0.33527055 |
| Mean A candidate-score standard deviation | 4.41486168 | 4.41483688 |
| Mean gated B correction standard deviation | 0.01117757 | 0.00837830 |
| Mean per-query correction/A standard-deviation ratio | 0.27947% | 0.21892% |
| Combined-minus-A MRR, descriptive | -0.00005971 | -0.00008298 |
| Adding B recovers / loses A top-1 queries | 551 / 573 | 372 / 414 |

The gate began at 0.05 and private logit scale at 1.0. Both learned smaller
scales, leaving B's candidate-dependent correction tiny. A gate of 0.04 does
**not** mean B contributes 4% of useful evidence: score scale also matters.
Query-weighted gate means differ from unweighted means across 102 relations
(both about 0.04708); frequent queries see the more suppressed gates.

All private parameter groups received nonzero finite first-step gradients and
changed by the endpoint. B's mean entity-row cosine with its initial random
table remained 0.98351 (dot) / 0.98212 (distance); relative table changes were
18.39% / 19.08%. Thus this is not a disconnected-gradient bug or literally
unchanged B. Different/uncorrelated scores alone are not useful diversity.
A residual branch need not be strong alone, but here its combined contribution
also failed to improve A, supporting the cold-start/suppression interpretation.

The branch-alone metrics are **diagnostics**, not newly selected inference
models. Endpoint reads used only saved states/query records; extra explanatory
ratios and initial-table comparisons are explicitly post-hoc descriptive checks
in [endpoint diagnostics](campaign_s0/endpoint_diagnostics.json).

### Original-error cohort: does B recover what A originally missed?

Post-hoc descriptive check requested by the user: the original frozen student
missed top-1 on 75,691 directed queries and was correct on 250,081. This uses
actual ranks, not `1 - MRR` as an error fraction.

| Endpoint | Original top-1 errors fixed | Originally correct top-1 predictions broken |
| --- | ---: | ---: |
| Trained single | 3,501 | 3,246 |
| A + dot-B | 3,547 | 3,320 |
| A + distance-B | 3,527 | 3,300 |

Dot fixes 46 more original errors than continued single training but breaks
74 more originally correct predictions; distance fixes 26 more and breaks
54 more. Each has 28 fewer net correct top-1 predictions than the control.
Dot uniquely fixes 258 original misses that the trained single still misses,
but loses 212 original misses the trained single fixes; distance uniquely
fixes 170 and loses 144. There is some different recovery, not zero recovery,
but no useful net gain on this experiment.

MRR on the original-missed cohort: single 0.30235811, dot 0.30263824, distance
0.30249430. The slight cohort improvements are outweighed globally by damage
to previously correct predictions. These VALID error cohorts remain diagnostic
only; any future error-focused supervision must derive its targets from TRAIN
(preferably held-out TRAIN folds), never these validation failures.

## Implication for the user's from-scratch / GAN suggestion

B's entity representation already started independently at random. Only A
was warm-started; private relation scales started at identity and shift at zero.
The evidence is consistent with the strong A solving the combined objective
while the gate/temperature suppressed a still-weak B. It does not establish
that any particular longer-training or from-scratch recipe will succeed.

The targeted next hypothesis is a separately budgeted **TRAIN-only B learning
phase with its own CE/KD supervision**, followed by joint training, with
matched dot/distance and training-budget controls. This tests whether B needs
competence before competing for contribution to a strong A. Another random
seed alone does not address that issue. Joint training from scratch is also a
valid separate experiment, but requires a substantially longer matched budget;
this short warm-start result does not rule it out. No adversarial/GAN objective
is implied by random initialization, and none was used here.

No automatic next run: keep the original promoted model and archive these
three endpoints. Do not retrospectively change width, gates, loss or budget.

## Launch checks

- 91 synthetic tests passed, including 13 new H35E tests. New tests cover
  original scorer/loss/gradient parity; matched dot/distance initialization
  and parameter counts; direct versus shared-negative score/gradient parity;
  independent parameter storage; nonzero gate; candidate permutation,
  duplicate and candidate-set checks; retained A-only trajectory; all-group
  updates; teacher freezing; schedules; RNG-neutral probes; selective loading;
  and standalone restore without teacher weights.
- Full-size synthetic GPU smoke: ten teachers plus three students; 20 updates
  per student in 5.1364 seconds, 3,748,161,024 bytes peak allocated (~3.49 GiB).
  All student groups updated, teacher hashes unchanged, all teacher gradients
  absent, candidate permutation and duplicate scores exact. This gave a
  ~22-minute training estimate, excluding loading and full evaluation.
- Real-data launch verified hashes of student, all ten teachers, release
  checksums, TRAIN/VALID/mapping/processed inputs and original query reference.
  Prior H35C and H35D source hashes remained intact.
- Initial single probe exactly reproduced archived MRR 0.8411956954420665.
  Dot/distance have identical initial tensors but different scoring functions:
  their initial probe MRRs were 0.8415197562 and 0.8412894107. Those differences
  are not learned improvements. Probe is 1,000 triples / 2,000 directed queries,
  not full-validation MRR.
- Original A has 27,124,129 real parameters; each private-B model has
  30,134,760. B adds 3,010,631 (~11.1%). Count and compute are disclosed.
- Local GTX 1080 Ti only; rented GPU untouched. No TEST reads, validation
  gradients, weight fitting on VALID, candidate-position cues or new features.
  Validation probes are read-only and do not select stopping time/checkpoint.

Artifacts: [pre-run receipt](campaign_s0/prerun.json),
[progress](campaign_s0/progress.jsonl). LR A starts at 0.0001, LR B at 0.001;
both cosine-decay to zero. Progress logs explicitly identify the last probe
step so stale probe MRR is not mistaken for a fresh evaluation.

## Endpoint audit and measured compute

- Exactly 5,000 updates in all three arms; shared TRAIN stream SHA256
  `8821d02f21c2b0b50154e368ab9f82271224ab1d699fae9606e4f972ce9cbe44`,
  identical to H35C. Every student parameter group changed; all ten teacher
  state hashes stayed identical and teacher gradients remained absent.
- All recorded source/input hashes remained unchanged. Independent auditor
  passed: matched dot/distance initial tensors, loss accounting, both cosine
  schedules ending at zero, standalone checkpoint restore, candidate symmetry,
  endpoint/query checksums and exact summary/rank agreement. The auditor loads
  no split files. No TEST access, no validation gradients, no inference ensemble.
- Training, probes, state audits and checkpoint writes: **1,203.38 seconds**
  (~20.06 minutes). Teacher forwards 142.83 s; single updates 131.81 s; dot
  updates 172.48 s; distance updates 740.49 s. The native L1 arm was much more
  expensive per update despite equal parameter counts; budgets match updates
  and examples, not wall-clock time.
- Full VALID including per-arm state checks: single 7.13 s, dot 12.49 s,
  distance 12.57 s. Peak training allocation **3,782,256,128 bytes** (~3.52 GiB).
  No compute job remained on the local GPU after completion.
- Preserved endpoint checksums:
  - single: `c7dbf76124cc178cdb4603a7ebf727c0b2f6909ba5caa322db7a400b00daf90f`
  - dot: `f6e0a79b7ecc8f7342635baff6f93160d50fc36ecf83692915d5891b97958096`
  - distance: `712869eeec6e0d39b29ab5a50cd91d16ed85e3dc07fcc16c7c9810701916299d`

[Full summary including direction/family slices](campaign_s0/summary.json),
[training audit](campaign_s0/training_audit.json),
[single query ranks](campaign_s0/single_queries.npz),
[dot query records](campaign_s0/dot_queries.npz),
[distance query records](campaign_s0/distance_queries.npz).

Rerun the read-only endpoint audit from the repository root:

```sh
/mnt/geocore/geocore/.venv/bin/python -m biokg.audit_mixed_operator biokg/results/h35e/campaign_s0
```
