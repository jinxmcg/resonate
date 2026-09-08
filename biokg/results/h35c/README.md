# H35C — jointly trained single versus dual operator, retained distillation

Completed on the local GTX 1080 Ti, 2026-09-06. **No promotion:** the dual
does not improve on the equally trained single. All 5,000 updates completed
for each; no intermediate checkpoint was selected.

## Results

| Student | Full-validation MRR | Delta vs original |
| --- | ---: | ---: |
| Original frozen reference | 0.832174867 | — |
| Jointly trained single | 0.832734761 | +0.000559895 |
| Jointly trained dual | 0.832702455 | +0.000527588 |

Dual-minus-single: **−0.000032306**, paired-triple bootstrap adjusted 97.5%
interval **[−0.000151584, +0.000102898]**. Dual-minus-original interval:
**[+0.000200504, +0.000851564]**. Neither required +0.001 gain was reached.
These intervals do not address training-seed variation or project-wide
adaptive validation reuse. This is not a leaderboard result.

The banks remain very similar: mean query cosine **0.99878997**, median
0.99885464. B's mean positive-score responsibility is **0.47710**; its 5th/95th
percentiles are 0.42863 / 0.51165. A and B assign the same positive rank to
91.88% of queries. These are branch-score diagnostics, not learned neighbor
attention weights. Freeing all parameters did not create a strong, useful
complementary role under this initialization and objective.

| Direction | Single | Combined dual | A alone (diagnostic) | B alone (diagnostic) |
| --- | ---: | ---: | ---: | ---: |
| Predict tail | 0.832937384 | 0.832995413 | 0.833964730 | 0.831661629 |
| Predict head | 0.832532139 | 0.832409497 | 0.834760107 | 0.828927813 |

A alone reaches 0.834362418 overall, but **was not the preregistered treatment**;
do not select/promote that branch post hoc from these validation diagnostics.
No head/tail routing was imposed: the legacy model already has independent
forward/reverse operators, and this trial uses A and B in both directions.

After discussing these results, the user explicitly requested different
information and a different starting function for B. That is a separate
follow-up, not a modification or extension of H35C.

## What this tests

The control and dual student both start independently from the released
`dist_T2_s0.pt`. Embeddings, operators and temperature are **all trainable**.
The dual has one shared entity table and two parallel 4×4 complex operator
banks, combined with fixed smooth-OR logits. A starts at the original
operator; B has the predeclared 5% per-block perturbation.

Both retain CE plus T=2, weight=1, T²-scaled distillation from the same ten
frozen released teachers, plus the original 0.1 trajectory alignment. The
dual's alignment acts on bank A; bank B receives CE/KD gradients through the
combined predictions, without its own trajectory penalty. This intentional
asymmetry permits, but does not guarantee, a distinct role for B.

Both students get the same 5,000 TRAIN batches and typed negatives, and
independent fresh dense Adam optimizers: LR 0.0001, cosine to zero, clipping
1.0. The teacher target is computed once per batch and reused, with no
student-to-student teaching. This is a matched **warm-start screen**, not a
from-scratch comparison or continuation of the original optimizer state.

All choices and promotion criteria were fixed before real-data execution in
[H35C.md](../../H35C.md); [launch receipt](campaign_s0/prerun.json) hashes
the protocol, implementation, source checkpoint, all ten teachers, TRAIN,
VALID, indexing metadata and frozen reference ranks.

## Evaluation and safeguards

- Read-only progress probe: the same fixed 1,000 validation triples as H35,
  both directions and all 500 provided negatives, every 500 updates. Probe
  scores do not select checkpoints, change LR or extend the run.
- Endpoint: full validation, 325,772 directed queries, official OGB average
  tie handling, no candidate-position-dependent scoring. Compare with the
  equally trained single and unchanged original student.
- All new fitting uses TRAIN. No TEST split or test score cache is opened.
  Validation never supplies gradients or teacher targets. Historical test
  exposure is not erased; this experiment is not an eligibility certificate.
- One model at inference. Teachers exist only during training; their
  parameters and compute remain part of the disclosed training resources.
- 68 BioKG tests passed before launch. The [full-size synthetic GPU smoke](synthetic_smoke.json)
  verified updates to every student parameter group and unchanged teachers.

The [training audit](campaign_s0/training_audit.json) verifies that E, every
operator bank and temperature changed, all ten teachers stayed byte-identical
and gradient-free, and both students used 5,000 updates. Shared TRAIN stream
SHA256: `8821d02f21c2b0b50154e368ab9f82271224ab1d699fae9606e4f972ce9cbe44`.
Training plus probes/audit/checkpoint writes took **466.79 seconds**; teacher
forwards 136.69 s, single updates 124.47 s, dual updates 195.44 s. Peak allocated
GPU memory was **2,338,668,544 bytes** (about 2.18 GiB), excluding non-PyTorch
desktop/driver memory. Both jobs finished; the rented GPU was untouched.

Inference parameter counts: single 27,124,129; dual 27,241,633 reals. Ten
training-only teachers each contain 27,124,129 reals. The extra bank adds
117,504 reals; it does not require another entity table.

The read-only endpoint auditor passed source/protocol hashes, loss accounting,
LR schedule, matched updates, trained checkpoint hashes, standalone restore,
candidate permutation, and rank artifact consistency. No split is loaded by
that audit. Both H35/H35B source/protocol receipts also still match.
Raw [summary](campaign_s0/summary.json), [progress](campaign_s0/progress.jsonl)
and endpoint per-query NPZ files are archived alongside the checkpoints.

## Reproduction

From the repository root, with the recorded checkpoints and dataset installed:

```sh
/mnt/geocore/geocore/.venv/bin/python -m biokg.train_joint_operator \
  --model biokg/checkpoints/dist_T2_s0.pt \
  --teacher-dir /mnt/geocore/wiki_pull/h24/dense \
  --checksums /mnt/geocore/wiki_pull/release/SHA256SUMS \
  --data-root /mnt/geocore/geocore/data_ogb \
  --h35 biokg/results/h35/campaign_s0 \
  --out biokg/results/h35c/new_replication \
  --device cuda
```

The output directory must not exist. The historical `wiki_pull` path contains
the verified **BioKG** teachers; no WikiKG experiment or rented GPU is used.
Checkpoints restore without teachers through
`biokg.joint_operator.restore_joint(checkpoint, device)`.

Endpoint audit:

```sh
/mnt/geocore/geocore/.venv/bin/python -m biokg.audit_joint_operator \
  biokg/results/h35c/campaign_s0
```
