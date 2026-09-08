# CFKD1: candidate-focused A-only distillation

Status: completed 2026-09-06 on the local GTX 1080 Ti; **not promoted**.
Both arms completed the fixed 5,000 updates and full-VALID evaluation.
The starting H35F A checkpoint remains the best model-only checkpoint.

Both independent arms start at the same H35F improved A checkpoint:
k=12, 144 complex entity coordinates, 4x4 relation blocks, 27,124,129 real
parameters, full-VALID MRR 0.8340991235 before this experiment. Control keeps
the original loss. Focused adds weight-0.25 conditional KD on the union of
top-32 teacher/student candidates plus cutoff ties and the known TRAIN
positive; full KD, CE and trajectory remain intact. Same own frozen teachers,
TRAIN batches, dense Adam, LR 1e-4 and 5,000-step cosine for both.

[Fixed protocol](../../CFKD1.md), [loss](../../candidate_focus.py),
[runner](../../train_candidate_focus.py), [tests](../../test_candidate_focus.py),
[endpoint auditor](../../audit_candidate_focus.py).

## Result and decision

Full VALID: 162,886 triples / 325,772 directed queries, all 500 official
negatives, official average-tie ranks. These are model-only scores.

| Fixed endpoint | MRR | Hits@1 | Hits@10 |
| --- | ---: | ---: | ---: |
| Starting H35F A | **0.8340991235** | 0.7703977015 | 0.9477794286 |
| Unchanged-loss control, +5,000 updates | 0.8338693602 | 0.7699004212 | 0.9477886375 |
| Candidate-focused KD, +5,000 updates | 0.8339914437 | 0.7700907383 | 0.9478469605 |

Primary focused minus control MRR is **+0.0001220835**, with 95%
whole-triple paired-bootstrap interval **[+0.0000361302, +0.0002063263]**.
Focused recovers 401 top-1 answers and loses 339 relative to control.
This small positive matched-control effect misses the preregistered +0.001
minimum, and focused also fails to equal or exceed starting A. Therefore
`advance_for_seed_confirmation = false`; neither endpoint is promoted.

Descriptive differences from starting A:

- Control: -0.0002297633, 95% interval [-0.0004756039, +0.0000329336].
- Focused: -0.0001076798, 95% interval [-0.0003906362, +0.0001468858].

Both intervals include zero: the endpoints are numerically lower, not
demonstrated regressions. This one-seed adaptive development comparison
does not establish seed robustness or pristine-holdout generalization.
The extra term changes both candidate emphasis and effective KD weight;
the matched control does not separate those mechanisms.

The final small-probe scores were 0.8459186 / 0.8460035 (control / focused).
A transient focused probe lead of about +0.001 at step 4,000 did not persist
at the fixed endpoint or transfer to full VALID. No probe checkpoint was
selected. Drug-drug and drug-sideeffect, major baseline deficit families,
remain numerically below starting A in both directions; protein-function
head improves slightly. There is no broad recovery of those residuals.

This rejects this exact added-loss continuation as an upgrade, not k=12 or
candidate-focused learning generally. Both arms already used k=12; this
was not a k=8-to-k=12 capacity comparison. The next recommended measurement
was original A versus the existing H35F improved A with the same feature
protocol. The user subsequently approved that separate [FC1 comparison](../feature_compare/README.md),
now complete: report-half pipeline MRR 0.8535135 -> 0.8540539, a small positive
gain. CFKD1 endpoints were not used in FC1. No next run or submission was
launched automatically from CFKD1.

## Prerun checks

- 34 synthetic tests passed: conditional loss/gradient equivalence to explicit
  slices, exact control parity, candidate permutation with remapped positive
  labels, cutoff ties, detached targets/selection, finite masked gradients,
  all-support behavior, TRAIN/VALID-only loading, independent model state,
  teacher freezing and standalone restore; existing joint/mixed tests too.
- Full-size 1080 Ti smoke passed: 20 paired updates, ten synthetic teachers,
  2,048 positives and 4,096 negatives per batch, all A groups changed and all
  teachers unchanged. Candidate-order scores matched exactly.
- Smoke training time 2.25165 seconds; 5,000-pair extrapolation 562.91 seconds,
  excluding real loading, probes, final evaluation and audit. Peak allocated
  memory 2,342,623,232 bytes (~2.18 GiB).

## Endpoint audit and compute

- The read-only endpoint audit passed: source/input hashes, identical
  initialization, 5,000 optimizer steps for every A parameter group in both
  arms, loss components and complete LR schedule, immutable own teachers,
  standalone checkpoint restoration and candidate-order symmetry. All saved
  summary metrics and bootstrap intervals were reproduced exactly. The
  auditor did not deserialize dataset splits.
- Both independent endpoints contain 27,124,129 real parameters and no B.
  The legacy state key `a.H_b` denotes A's existing block operators, not a
  second branch. All A parameter groups changed; teachers had no gradients.
- Last actual LR used: 9.869604078449588e-12; next LR: 0. Neither is null.
- Recorded training, probes, checkpoint audits, full VALID and bootstrap:
  496.62 seconds (~8.28 minutes), excluding initial loading and final input
  hashing. Peak allocated memory 2,340,664,320 bytes (~2.18 GiB).
- Update component times: teachers 145.40 s, control 134.36 s, focused
  186.19 s. Focused updates cost about 38.6% more than control: the experiment
  matches update count, not compute. The local GPU is now free of compute jobs.

No external trained artifacts, TEST access, new inference branch, feature
changes or submission changes. VALID probes are read-only, not training
targets or configuration selection. Fixed endpoints only, no early stopping.
The historical submission + features score is not interchangeable with this
model-only experiment. No claim of second place follows from a probe gain.

Artifacts: [receipt](s0/prerun.json), [progress](s0/progress.jsonl),
[training audit](s0/training_audit.json), [summary](s0/summary.json),
[endpoint audit output](s0/endpoint_audit.json).

Reproduce the read-only endpoint audit:

```sh
/mnt/geocore/geocore/.venv/bin/python -m biokg.audit_candidate_focus biokg/results/candidate_focus/s0
```
