# H37N: H37 with underflow-safe normalization

2026-09-07. [Repair protocol](../../H37_NUMERICS.md), [original H37](../h37/README.md).
Status: completed, audited, and not promoted. The fixed overall gate failed.
Original failed attempt is preserved separately; no recipe sweep or
partial-checkpoint selection.

Clamp squared norm to fp32's smallest normal positive value before sqrt, then
use the unchanged +1e-8 denominator epsilon. Applies to source/hop normalization
in both arms. Same forward function at fp32 resolution on checked normal/tiny
rows, but finite backward where squared-norm underflow previously failed.
Same 4x4 storage, 27,124,129 parameters, epsilon1e-6 squared likelihood, trajectory,
learning rates, initial seed0, 50k steps and fixed primary gate as H37.

Mandatory steps: tests -> recorded failing-batch no-update reproducer -> same
synthetic cost pilot -> fresh matched training -> frozen evaluation -> audit.
The failure-state checkpoint is ONLY a reproducer input, never a warm start
or training input for the restarted pair. Every original and adapter source
is pinned into new receipts. No official VALID/TEST, external weights, teacher,
distillation or submission change. Internal holdout remains development data.

## Completed result

Both fresh seed-0 arms completed 50,000 updates. On the fixed internal TRAIN
holdout, with identical candidates and 472,570 queries:

| Metric / slice | Linear control | Squared likelihood |
| --- | ---: | ---: |
| All-query MRR | 0.6204363194 | 0.5741653145 |
| All-query Hits@1 | 0.4948579046 | 0.4386165013 |
| All-query Hits@10 | 0.8896967645 | 0.8320206530 |
| Drug–drug MRR, 110,520 queries | 0.2758951508 | 0.5011380910 |
| Other relations MRR, 362,050 queries | 0.7256115437 | 0.5964577292 |
| Final in-sample fit-probe MRR | 0.8825787593 | 0.6590160339 |

Primary squared-minus-linear delta: **−0.0462710049**, unordered-pair-cluster
95% bootstrap interval **[−0.0497774451, −0.0425128631]**. The preregistered
advance gate is false. The interval does not measure training-seed uncertainty;
this is an already-used internal development split, not official leaderboard
MRR. The non-drug–drug aggregate above is derived by subtracting the drug–drug
query-weighted contribution from the all-query aggregate. Fit-probe and
holdout candidate sets differ, so their difference is not a calibrated
generalization-gap estimate.

## What we learned and what remains uncertain

- The loss is not uniform across relations: drug–drug improves substantially,
  while the other relations lose more in aggregate. This provides a useful
  relation-specific learning signal, not permission to bypass the overall gate
  or claim a submission improvement. It does not identify which changed
  component caused that gain.
- The squared arm also ranks worse on the fit probe. This is not simply a
  case of improved training ranking and worse held-out ranking. The tested
  scorer, exact denominator and retained optimizer/trajectory combination
  merits investigation before interpreting this as a 4x4 capacity limit.
- The original failure exposed a concrete tiny-norm backward vulnerability.
  The no-update repair proof preserved the failing batch's query and loss
  exactly and made gradients finite, but the repaired entity-gradient norm
  was **6,351,189,180,416**. With global clip 1, that batch's raw gradients
  would be multiplied by roughly 1.57e-13. This can suppress other groups'
  current gradients; optimizer momentum can still cause updates. We have not
  measured how often this occurs or established that it caused the MRR loss.
- This changes scoring, objective and candidate coverage together. It rejects
  this complete recipe, not every squared scorer, the 4x4 representation, or
  the published ComplEx² model, which this is not a reproduction of.

Possible next diagnostic, **not launched**: inspect per-row entity norms,
per-parameter-group gradient norms and clipping factors on fixed FIT-only
batches, with relation slices. First establish whether tiny rows repeatedly
dominate clipping before proposing a separately fixed stabilization test.
No new architecture, loss mixture, hyperparameter sweep or training approved
by this result alone.

## Audit, backups and shutdown

Seven repair tests plus the previous 52 local tests passed, as did 29 separate
legacy regression tests (88 distinct local tests); 39 tests passed remotely.
Recorded-batch no-update proof passed. Synthetic cost pilot passed at
7.24466/7.02491 ms per paired step and 595,230,720 bytes peak allocation.
Training took 365.69 s, frozen evaluation 25.71 s and endpoint audit 25.45 s,
excluding setup and backup. Endpoint audit replayed scores, ranks, candidates
and pair-cluster statistics, regenerated the fresh initial state, and checked
saved optimizer counters/state and LR schedule; it does not claim full
optimizer-trajectory replay. All checks passed.

The final linear model and optimizer digests, training batch stream and
evaluation candidate stream exactly match the previous MN1 baseline. Local
copies of both endpoints, ranks, receipts, source/data hashes, pilot and
failure reproducer were verified against their recorded hashes. Original
failed-attempt artifacts and supervisor logs are also preserved locally.

Finite supervisor job biokg_h37_stable exited. At the user's request, Vast
instance **50109054**, RTX5090 at 79.19.114.194, was stopped on 2026-09-07.
CLI confirmed actual_status=exited, intended_status=stopped, cur_state=stopped.
The instance was not destroyed; disk is retained and storage charges may
continue. No other instance was touched and no follow-on job was launched.
Official VALID/TEST and the existing submission remain untouched.
