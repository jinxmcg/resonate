# H35F: direct B warm-up before joint training

Status: completed, 2026-09-06. **Warm-up strategy not promoted.** All arms
finished exactly 15,000 updates; no subsequent training or deployment change.

User-approved fixed three-arm experiment, 15,000 matched TRAIN batches each:
single A; original joint A+B; and 10,000 direct B-only updates with frozen
A/gate followed by 5,000 joint updates. All use the original released student
for A. Both B arms have the identical H35E dot architecture, independent random
initialization, width 32, and separate parameter storage. No L1 branch.

[Protocol](../../H35F.md), [warm-up loss/phase helpers](../../branch_warmup.py),
[runner](../../train_branch_warmup.py), [tests](../../test_branch_warmup.py),
[endpoint auditor](../../audit_branch_warmup.py).

## Full-validation result

162,886 official VALID triples / 325,772 directed queries, all 500 supplied
negatives. Fixed endpoints only, not best progress-probe checkpoints.

| Endpoint | MRR | Hits@1 | Hits@10 |
| --- | ---: | ---: | ---: |
| Original frozen student | 0.8321748668 | 0.7676565205 | 0.9471839200 |
| Single A, 15,000 updates | **0.8340991235** | 0.7703977015 | 0.9477794286 |
| Usual joint A+B, 15,000 updates | 0.8338250000 | 0.7699771619 | 0.9477579411 |
| B warm-up then joint A+B | 0.8321677853 | 0.7676043368 | 0.9472974964 |

| Preregistered contrast | MRR delta | Adjusted 98.3333% interval |
| --- | ---: | --- |
| Warmup minus usual joint | -0.0016572147 | [-0.0020000205, -0.0013231639] |
| Warmup minus single | -0.0019313382 | [-0.0022556648, -0.0016251523] |
| Warmup minus frozen | -0.0000070815 | [-0.0002250540, +0.0002231566] |

No promotion: warmup lost to both controls and was essentially unchanged from
the original frozen student. Whole-triple paired bootstrap, 1,000 replicates,
seed 3516, adjustment for three declared comparisons. Intervals quantify query
variability, not training-seed uncertainty or untouched-holdout generalization.

## Did B actually learn, and can it replace A?

Direct supervision changed B substantially. On the fixed 1,000-triple probe,
warmup B-alone MRR rose from 0.01596 initially to 0.19859 at step 500,
0.48791 at 2,000, 0.61213 at 5,000 and **0.67628 at the 10,000-step boundary**.
It ended at 0.67519 on that probe after 5,000 joint updates. The usual joint
arm's B stayed around 0.013–0.016. This is a same-initialization comparison of
the two training strategies, not evidence that the smaller architecture is
intrinsically superior or that its early growth can be extrapolated.

Full-VALID branch diagnostics at the final endpoint:

| Diagnostic | Usual joint | Warmup then joint |
| --- | ---: | ---: |
| A-alone MRR within the endpoint | 0.83408375 | 0.83184198 |
| B-alone MRR | 0.01216183 | **0.67510360** |
| Mean within-query A/B score correlation | -0.00499428 | 0.75101656 |
| Mean gate, query weighted | 0.04358406 | 0.04671798 |
| Final private logit scale | 0.69281179 | 1.93793845 |
| Mean per-query B correction/A score-spread ratio | 0.65269% | 4.61213% |
| Combined-minus-own-A MRR | -0.00025875 | **+0.00032581** |
| Adding B: A misses recovered / A correct predictions lost | 1,548 / 1,683 | **1,530 / 1,320** |

Warmed B can score independently using about **3.0 million real parameters**
(private entity table, maps, translation and scale; no A or gate), versus
27.1 million for A. But its **0.67510 MRR is not competitive with A's 0.83410**
here. No standalone B was exported or substituted for the current model.
The boundary checkpoint was preserved as preregistered; its B-alone number
above is the progress probe, not an additional full-VALID boundary evaluation.

There is now a small useful B signal: adding warmed B to its own endpoint A
gains 0.0003258054 MRR (descriptive 95% interval [+0.0001541709, +0.0005020688]).
B alone ranks the positive first on 10,007 queries where that A does not;
the fixed learned combination turns only some of those into net recoveries.
This does not imply that simply increasing its gate would help: wrong B
preferences can also overturn correct A predictions.

The full two-stage strategy still loses. Its A-alone endpoint is weaker than
the independently trained single control, and the small B benefit does not
bridge that gap. A-alone branch scores are diagnostic, not alternative selected
models. Warmup A/gate had 5,000 updates at the tail of the common cosine,
while the controls' A had 15,000 updates. That was disclosed in advance: this
is a strategy comparison, not an isolated causal test of warm-up alone.

## Original-error recovery

The frozen student had 75,691 missed top-1 queries and 250,081 correct ones.
These are actual query counts, not `1 - MRR` interpreted as an error rate.

| Endpoint | Original top-1 errors fixed | Originally correct predictions broken |
| --- | ---: | ---: |
| Single | 4,191 | 3,298 |
| Usual joint | 4,491 | 3,735 |
| Warmup then joint | 1,771 | 1,788 |

Warmup's useful correction relative to its own A therefore does not amount to
better net recovery from the original student. These VALID cohorts are
diagnostic only; no such failures were used as training examples, weights or
targets. Any later error-focused training must derive its labels from TRAIN.

## What to retain from this result

- The cold-start issue was real in this recipe: B learned a viable ranker when
  directly supervised and retained that competence after the phase switch.
  Nevertheless, competence is not sufficient for a superior combined model.
- Do not replace A with B or promote this two-stage endpoint. No automatic
  gate, schedule, width, objective or from-scratch follow-up was launched.
- The strongest arm here was continued A-only training. Its improvement over
  frozen is +0.0019242566 MRR; post-hoc descriptive paired 95% interval
  [+0.0016413441, +0.0022317435]. It also exceeds the previous 5,000-update
  H35E single by +0.0013643620, interval [+0.0010953596, +0.0016196685].
  These extra comparisons use saved ranks, seed 3517, and are **not** the
  preregistered warm-up promotion tests. The 15,000-step cosine differs from
  the earlier 5,000-step cosine; this is not an exact optimizer continuation.
  A-only is a candidate for separate confirmation, not a newly promoted
  submission or evidence of a top-three result.

## Endpoint correctness and measured compute

- 101 tests passed before launch. Independent endpoint auditor also passed:
  old/new source hashes intact; matched examples/optimizer calls; exact
  parameter update counts; frozen warm-up A/gate; unchanged teacher states;
  retained losses; one cosine without reset; standalone restore, candidate
  symmetry, checkpoint/query checksums and summary/rank consistency.
- At 10,000, A and gate state hashes exactly matched initialization. Every
  B group changed; A/gate had zero Adam steps, B groups had 10,000. The
  boundary checkpoint and optimizer counters were recorded before unfreezing.
- Endpoint counts: single A 15,000; joint A/B/gate 15,000; warmup B 15,000,
  warmup A/gate 5,000. All endpoint parameter groups changed. All ten teachers
  stayed frozen with absent gradients. Full input/source hashes were checked
  before and after execution. No TEST, validation gradients or rented GPU.
- Training, probes, boundary/endpoint checks and writes: **1,679.35 s**
  (~27.99 min). Teacher forwards 429.09 s; single updates 395.11 s; usual joint
  517.60 s; warmup-strategy updates 304.96 s. Equal examples/calls do not mean
  equal compute or per-parameter optimization exposure.
- Peak allocated GPU memory **2,771,969,536 bytes** (~2.58 GiB). No compute job
  remained on the local GTX 1080 Ti after completion.
- Shared TRAIN stream SHA256:
  - 5,000: `8821d02f21c2b0b50154e368ab9f82271224ab1d699fae9606e4f972ce9cbe44`
  - 10,000: `686a81ed7bb9f7e158fda0445fee47ada074249e2a448732d7dfa08a77897688`
  - 15,000: `e4ab0679d9a343e183e5e10c27e378438def3c2d69fbefa91ae19c674e23c99a`
- Endpoint checkpoint SHA256:
  - single: `df7c783e7b8a6110f126bca1c60d89209735accbaa5ffc0b362f2cab52d9d2f1`
  - joint: `69cdfdbfb804727d253d45573ae77ad82b73a6e2024e9d22a9132e5c1b56e707`
  - warmup: `0404ecd72d599b94cd88509b193ee6b189da15dec0dab5cb1b17f702e97e5ecb`

[Summary with family/direction slices](campaign_s0/summary.json),
[training audit](campaign_s0/training_audit.json),
[boundary audit](campaign_s0/boundary_audit.json),
[single ranks](campaign_s0/single_queries.npz),
[joint branch records](campaign_s0/joint_queries.npz),
[warmup branch records](campaign_s0/warmup_queries.npz).

Read-only endpoint audit (loads no dataset split):

```sh
/mnt/geocore/geocore/.venv/bin/python -m biokg.audit_branch_warmup biokg/results/h35f/campaign_s0
```

## Prerun verification

- 101 synthetic tests passed, including 10 new warm-up tests. Tests verify
  direct ungated B scorer/loss/gradient parity; no A/gate dependency in that
  loss; frozen groups with absent gradients and Adam state; B changes;
  state-preserving phase switch; restored original combined objective;
  effective and configured LR; RNG-neutral branch probes; selective loading;
  independent initial tensors/storage and teacher-free checkpoint restore.
- Full-size local GTX 1080 Ti synthetic smoke: ten teachers, three students,
  batch 2,048 / 4,096 shared negatives, 20 updates per phase. Warm-up phase
  2.5110 s, joint phase 2.3960 s; estimated total training 1,854.50 s (~31 min),
  excluding real loading, probes, audits and full evaluation. Peak allocation
  2,771,709,440 bytes (~2.58 GiB). Frozen A/gate, B updates, retained optimizer
  state, unchanged teachers and exact candidate-permutation scores passed.
- `null` gradient fields for A/gate during warm-up mean deliberately absent
  gradients, not numerical errors. Effective LR for those groups is zero;
  the configured common cosine schedule is logged separately.
- One common 15,000-step cosine schedule, A LR 0.0001 and B LR 0.001 initially;
  no reset at 10,000. Warmup A/gate receive only 5,000 parameter updates and
  join the schedule partway through. This is a two-stage strategy comparison,
  not perfectly isolated warm-up causality or equal wall-clock compute.
- TRAIN/VALID only, no TEST, no validation-error supervision, no graph changes.
  All teachers remain training-only; one unchanged-architecture inference
  checkpoint per arm. Historical exposure and adaptive VALID reuse remain
  disclosed. No automatic next run.

Artifacts: [prerun receipt](campaign_s0/prerun.json),
[progress](campaign_s0/progress.jsonl). Probes use the fixed H35 1,000-triple
subset (2,000 directed queries), not full validation. Logs identify both
combined and B-alone MRR and the last evaluated probe step.
