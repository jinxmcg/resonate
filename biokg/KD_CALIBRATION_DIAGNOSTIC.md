# TRAIN-only KD confidence versus ordering diagnostic

Fixed 2026-09-06 before execution. User approved the next investigation on
the local GTX 1080 Ti. No trained external artifacts, no VALID/TEST access,
no model updates, no submission changes. Reuse the previous diagnostic's
own-model provenance and TRAIN-only file-access guard.

## Fixed sample and quantities

- Original distilled A and H35F improved single A, the same ten own frozen
  teachers, exact previous input/source checksums. Verify all 12 model states
  and source/input files before and after. All model gradients disabled.
- Exactly 128 matched batches, 512 TRAIN positive draws and 4,096 shared
  typed random negatives each, `TrainStream` seed 3602. Natural relation
  frequencies, equal-probability prediction directions, unchanged duplicate
  and sampled-positive handling. The smaller positive batch is for bounded
  score-cache memory, not a new training recipe. No candidate-score caches
  are persisted; save only diagnostic row statistics and input hashes.
- Split by a fixed hash of the canonical original TRAIN triple `(h,r,t)`.
  Repeats and reciprocal queries stay together. Fit and report rows are both
  TRAIN, already seen by the models; report rows are held out only from this
  scalar calibration, not an untouched generalization set.
- Keep T=2 and the T-squared KD factor. Write `x=(s-mean(s))/T` and
  `p=softmax(teacher_logits/T)`. Candidate-independent centering preserves
  softmax and ordering. Fit a single positive scalar beta per A snapshot by
  minimizing `4*KL(p || softmax(beta*x))` on fit rows only. This is an
  auxiliary diagnostic scalar, never written into an A checkpoint.
- Convex one-dimensional fit by 24 bisection iterations, fixed bounds
  `[1/16,16]`; handle boundary/flat optima explicitly. Report fitted beta,
  boundary status and fit derivative before/after. Apply that same frozen
  beta to report rows. No relation-specific or VALID-fitted calibration.
- Primary measurements: baseline versus globally calibrated KD on fit and
  report TRAIN rows. Lower KL need not imply any better candidate ranking.
- Secondary optimistic scale-only bound: for each report TRAIN query, find
  its own best beta in the same interval against that query's teacher scores.
  This same-query fit is explicitly an oracle diagnostic, **not** a deployable
  policy, held-out calibration result or source of student supervision.
  Its residual KL still includes gap/shape differences even when rankings
  agree. It must not be called "pure ranking error".
- Direct ranking descriptors on report rows: teacher/student top-score-set
  overlap (tie-aware, candidate-symmetric), centered score cosine and whether
  the known TRAIN positive is among each model's maximum-scoring candidates.
  These are not official OGB MRR and do not use supplied VALID/TEST negatives.
  Assert global and per-query positive scaling leave top-score sets unchanged
  using float64 multiplication for this check to avoid introducing fp32 ties.
- No automatic optimizer ablation or scale promotion. Use these measurements
  to decide whether a separate training-only KD confidence scalar is a useful
  controlled A-only hypothesis, with the unchanged loss as matched control.
  Per-query oracle gains alone do not justify that global-scalar experiment.

## Checks and execution

Synthetic tests verify retained-KD equivalence, calibration derivatives,
known global/per-query scales, fit/report isolation, reciprocal-consistent
splitting, bounded optima, uniform distributions, candidate symmetry and
irreducible reversed-order examples. Existing gradient tests also run.

```sh
/mnt/geocore/geocore/.venv/bin/python -m unittest biokg.test_kd_calibration_diagnostic biokg.test_gradient_diagnostic biokg.test_joint_operator
/mnt/geocore/geocore/.venv/bin/python -m biokg.kd_calibration_diagnostic --out biokg/results/kd_calibration_diagnostic/s0
```

Progress logs report stage, fitted-scale iteration/query counts, LR=0 and
no MRR evaluation. All model parameters remain unchanged; numerical scalar
fitting is restricted to TRAIN-derived scores. No external GPU is used.
