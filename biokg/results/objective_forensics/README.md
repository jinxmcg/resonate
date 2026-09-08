# TF2: objective and existing-retrieval forensics

Status: completed and audited, 2026-09-07. Protocol frozen before execution
in `biokg/OBJECTIVE_FORENSICS.md`. User requested investigation, not training.

16 TRAIN near-misses (first eight per direction, unique unordered pairs) from
TF1's existing sample. Check the current fixed seed0/fold0 pipeline on a
pair-removed TRAIN graph and full drug catalog. Then inspect CE, own-teacher
KD and trajectory contributions to each positive-versus-competitor margin,
both per-example and in a same-relation minibatch. Compare ordinary random
negatives with the same pool containing the hard competitor.

Only our own frozen student and original own teachers; no external models.
TRAIN-only autograd, no optimizer/update, no VALID/TEST edges, no mixture
refitting or model promotion. CPU retrieval, local1080Ti gradients, LR=0.
Full-catalog TRAIN ranks and infinitesimal Euclidean gradient projections
are not official MRR, actual Adam steps or a forecast of generalization gain.

## Main finding: available local correction, sometimes conflicting context

For each case, measured the temperature-independent margin
`M = score_without_tau(positive) - score_without_tau(competitor)`.
All selected margins are negative. The table counts positive
`-dot_real(grad M, grad loss)`, meaning an infinitesimal ordinary gradient
descent step would increase that pair's margin. No such step was performed.

| Parameter group / loss context | Ordinary random pool | Competitor included |
| --- | ---: | ---: |
| Entities + operators, focal example total loss | 16/16 help | 16/16 help |
| Entities + operators, whole-batch total loss | 13/16 help | 16/16 help |
| Operators only, focal example total loss | 16/16 help | 16/16 help |
| Operators only, whole-batch total loss | 8/16 help | 9/16 help |

Thus the current 4x4 layout has a local corrective operator direction for
every inspected pair. That does not prove it can satisfy all graph constraints
simultaneously, or rule out a capacity limitation. The measured effect does
support investigating shared-parameter training context before increasing k.

In the three random-pool cases where the full representation gradient hurts,
the focal contribution is positive even after dividing by batch size2048;
the other rows outweigh it. These are arithmetic decompositions of the saved
projections, not additional gradient runs:

| Zero-based case | Focal contribution /2048 | Other-row contribution | Batch total |
| --- | ---: | ---: | ---: |
| 7 | +0.00896668 | -0.01689701 | -0.00793033 |
| 9 | +0.00652442 | -0.02404012 | -0.01751570 |
| 12 | +0.00775944 | -0.06854724 | -0.06078780 |

## Negative exposure: interesting, but not a training recommendation yet

The strongest unobserved competitor appears in6/16 ordinary pools. The10 other
pools were changed by replacing one shared negative with that competitor. All
three harmful whole-batch directions occurred among those10 and become helpful
under the changed pool.

Important limitation: this is a **shared** negative pool. Inserting a drug
changes all2,048 rows, not only the focal row. Across the10 changed pools, mean
representation margin-projection increase is+0.03058968; only+0.000286684 of
that is the change in the focal row's batch-averaged contribution. Most of the
effect therefore comes through other batch rows. This is not evidence that a
row-local hard-negative loss would produce the same gain, nor that blindly
forcing shared hard negatives would generalize. Known-positive collisions and
regressions on other queries remain important; no new loss was trained.

These pools are newly sampled conditional TRAIN batches, not historical batch
replay. Six of sixteen is not an estimate of the model's lifetime exposure.

## Distillation: shared mistakes, but no dominant local veto

The original mean of the ten own teachers prefers the correct candidate in
only5/16 selected near misses; it prefers the competitor in11/16. This is a
selected difficult in-sample set, not teacher accuracy over the dataset.

On focal losses with ordinary random negatives:

- CE helps16/16; mean representation projection+17.42124.
- Weighted trajectory helps16/16; mean+0.23760.
- KD helps9/16 and hurts7/16; mean+0.02335.
- Their total helps16/16; mean+17.68218.

Teacher ranking and KD's local parameter-gradient direction are different
measurements: the latter depends on the student and the full sampled pool.
These results do not support claiming KD or trajectory is overpowering the
correct-answer signal, or removing either wholesale. They also show that
simply copying these teachers' ordering cannot fix all selected mistakes.

## Original drug1381 --relation38--> drug786 case

The model ranks drug786 tenth. After the exact existing pipeline's frozen
weights and pair-removed TRAIN features, its rank becomes **6**, not1.
It now narrowly beats the old competitor1528 (pipeline margin+0.00741148),
but drug1048 and four other remaining candidates still outrank it. The
weakness is therefore not just the original one-versus-one comparison.

The ordinary4,096-draw pool contains zero occurrences of competitor1528 and
86 draws of other known TRAIN-positive answers for this query. No positive
mask was added: this reproduces the original conditional loss recipe.

Seven of ten own teachers prefer1528 to786; mean raw teacher pair margin
is-0.13563733. For this focal example the representation projections are
CE+15.87891, KD-0.03352, weighted trajectory+0.20476: total+16.05016.
Whole-batch total is+0.01181957, with operators alone mildly opposing
(-0.000079303). The example is not stuck because there is no local descent
direction in the existing architecture, nor because KD dominates its CE.

## Existing retrieval over the16 cases

Fixed seed0/fold0 pipeline improves7 ranks, worsens5, leaves4 unchanged,
and recovers zero rank-one answers. It overtakes the original model's strongest
competitor in2 cases. These are full-catalog TRAIN-filtered, deliberately
selected near misses, with focal graph edges removed—not official501-candidate
evaluation, and not a reason to discard the established retrieval pipeline.

Descriptive sample MRR: model0.24375, pipeline0.2338316198. Do not compare these
to the existing adaptive VALID pipeline aggregate0.8582166512. Candidate-pool
normalization, selection, graph and split differ. No new validation result.

## Verification and decision

117 tests passed, including complex-gradient finite differences, per-component
gradient-sum parity, direct holder features, candidate permutation, pair-copy
removal, exact nested blend arithmetic and deterministic batch construction.
Main run41.92 seconds; peak allocated GPU memory2,125,538,816 bytes (~1.98GiB).
Our student and all ten own teachers have identical before/after state hashes;
all parameter `.grad` fields remained empty. No optimizer or updates.

Saved-record endpoint audit passed: cases, TRAIN batches, all stored
normalization/blends/ranks, component projection sums and summary reproduced;
input/source/artifact hashes verified. It did not independently rerun GPU
gradients or every graph feature; those were checked directly in the runner
and synthetic tests. Only TRAIN/node counts dataset files opened. Existing
JSON hyperparameter recipes were read unchanged; no VALID/TEST dataset/caches.

Keep k12/4x4 and the current submission unchanged. This investigation motivates
further scrutiny of supervision, shared-batch interference and the actual
post-retrieval confusers; it does not establish a specific training fix or
authorize another experiment. No8x8 or larger-k run was started.

Artifacts in `s0/`: `prerun.json`, `cases.json`, `batches.npz`, `features.npz`,
`gradients.jsonl`, `summary.json`, `audit.json`, `endpoint_audit.json`.
