# TRAIN loss-gradient diagnostic: completed on the GTX 1080 Ti

2026-09-06. **The distance-loss conflict hypothesis is weakened, not proven
false. No model or objective was changed.**

Follow-up: the [confidence-versus-ordering diagnostic](../kd_calibration_diagnostic/README.md)
is now complete. It found little reducible global or per-query scale mismatch,
so the temperature-gradient observation below did not justify prioritizing a
temperature-only training change.

[Fixed protocol](../../GRADIENT_DIAGNOSTIC.md),
[runner](../../gradient_diagnostic.py),
[tests](../../test_gradient_diagnostic.py),
[full summary and state audit](s0/summary.json),
[per-batch gradient Gram matrices](s0/batches.jsonl),
[receipt](s0/prerun.json), [progress](s0/progress.jsonl).

## What was measured

The original distilled A and improved H35F A-only endpoint were inspected on
the same 512 TRAIN batches: 2,048 positives and 4,096 shared typed negative
draws each, seed 3601. That is 1,048,576 positive draws per snapshot, not that
many distinct triples. There were 261 head-prediction and 251 tail-prediction
batches, covering 62 of 102 directed relations under the unchanged natural
relation-frequency sampler. Rare relations are incompletely represented.

Each loss was differentiated separately: CE, own-teacher T=2 KD including
its T-squared multiplier, and trajectory distance including its 0.1 weight.
The ten teachers are our existing H24 models, not external checkpoints.
All figures below describe real-coordinate Euclidean gradients before any
optimizer. They do not measure Adam updates, MRR or learning over time.

## Main result: trajectory is small and mildly aligned overall

Representation means the entity table plus relation operators, excluding
the scalar log temperature. Entries are means of per-batch measurements.

| Diagnostic | Original A | Improved A |
| --- | ---: | ---: |
| Trajectory / (CE+KD) gradient norm, representation | 3.097% | 2.989% |
| Cosine: trajectory vs CE+KD, representation | +0.09235 | +0.09265 |
| Batches with negative representation cosine | 23.63% | 22.85% |
| CE+KD descent projection ratio, representation | 1.00238 | 1.00231 |
| Trajectory / (CE+KD) gradient norm, operators only | 11.16% | 10.93% |
| Cosine: trajectory vs CE+KD, operators only | -0.03449 | -0.04139 |
| Batches with negative operator cosine | 60.94% | 61.52% |
| CE+KD descent projection ratio, operators only | 0.99401 | 0.99254 |

The projection ratio is
`1 + dot(g_CE+KD, g_trajectory_weighted) / ||g_CE+KD||^2`.
A value of 0.99254 means 0.75% less CE+KD descent projection for that
parameter group, on average, under an infinitesimal unpreconditioned step.
It does **not** mean a 0.75% MRR loss or a 0.75% slower Adam training run.

The improved model's operator projection p10/p90 is 0.97210/1.01490.
The worst sampled batch is 0.75616 (base relation 35, drug-drug, tail);
the next-lowest batches involve that relation and its reverse. Thus the
small aggregate effect does not exclude stronger local conflicts. There
are only three sampled batches for that forward relation and four reverse.
Do not infer a relation-specific loss rule from this tiny sample.

At the representation level, improved A has positive mean alignment in
both directions: head +0.10905, tail +0.07560. Its mean cosine is positive
for drug-drug, protein-function and drug-sideeffect in both directions.
The sideeffect slices contain only 6 head and 8 tail batches. Most negative
representation cosines occur in the function-function family, with a small
mean projection effect. Slices are descriptive TRAIN diagnostics, not
validation-failure weighting or evidence that a biological feature is absent.

**Interpretation:** this is not evidence of a large, pervasive trajectory
gradient fighting ranking at these two trained snapshots. Dropping trajectory
is not justified by this result alone. Its earlier-training influence,
regularization benefit, rare-relation behavior and Adam-preconditioned effect
remain unmeasured.

## More conspicuous interaction: distillation and score temperature

CE and KD push the scalar log temperature in opposite directions on
91.99% of batches for original A and 89.45% for improved A. On improved A,
the median fraction of KD's squared Euclidean gradient norm located in
that single scalar is **92.36%** (mean 70.63%, p10/p90 7.67%/98.18%).
Original A has median 87.12%, mean 64.16%.

On the representation itself, KD's norm is only 5.57% of CE's on average
for improved A (median 4.78%); original A is 5.84% (median 5.09%). Mean
representation CE/KD cosine is +0.01646 for improved A, despite the
negative all-parameter mean cosine (-0.10154). Aggregating the temperature
with the embeddings would obscure this distinction.

These squared-norm fractions depend on the parameterization and are not
fractions of actual learning or Adam movement. They do not show that KD
was ineffective: these students have already undergone distillation.
Opposition over confidence can also be a normal calibration tradeoff.
Positive score scaling alone cannot change a frozen per-query ranking,
but its training value changes the gradients into the representation.

**Next training hypothesis, not an authorized new run:** separate matching
the teachers' score confidence from transferring their candidate ordering.
A controlled investigation of KD calibration versus ranking information is
more motivated now than removing trajectory wholesale or adding another B.
Do not automatically increase KD weight, remove KD, detach its temperature,
add relation gains, or fit calibration on VALID. Any ablation needs its own
fixed TRAIN-only design and equally trained unchanged-loss control.

## Verification and resource use

- 17 synthetic tests passed before launch, covering original-score/loss
  parity, complex real-coordinate gradient products, summed-loss gradient
  parity, zero/unused gradients, TRAIN-only loading and deterministic typed
  sampling, plus existing joint-operator correctness tests.
- Separate gradients matched the original total on the first real batch
  for both snapshots. No optimizer was constructed; zero updates; LR 0;
  no parameter `.grad` populated. No MRR evaluation was performed.
- All two student and ten teacher tensor hashes matched exactly before and
  after. Input checkpoints, sources and TRAIN files also remained unchanged.
- The dataset-open audit records only `split/random/train.pt` and
  `raw/num-node-dict.csv.gz`; its guard rejects all other dataset files.
  No VALID, TEST, processed dataset cache, external weights or rented GPU.
- Independent saved-record verification recomputed all 1,024 records'
  metrics from their stored Gram matrices, reproduced the complete summary,
  and rechecked pinned file hashes. All passed.
- Gradient loop: **57.19 seconds**, excluding loading and endpoint hashing.
  Peak PyTorch allocation 2,252,383,232 bytes (~2.10 GiB); process memory
  observed by the GPU driver about 2.6 GiB including runtime overhead.
  No compute process remained on the 1080 Ti after completion.
- Shared TRAIN stream SHA256:
  `22b70d3c4180cbe8dc170abea330f4adf6ce2cb39ea00c8421f4570b178294c1`.

No model promotion, submission modification, new teacher or follow-up
training run was made. The separate old-A+features versus improved-A+same-
features comparison remains pending.
