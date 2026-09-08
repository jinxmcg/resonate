# TF3: protected shared competitor

Protocol: [PROTECTED_COMPETITOR.md](../../PROTECTED_COMPETITOR.md).

Completed 2026-09-07 on the local GTX 1080 Ti. **Do not promote this shared-hard
variant to training.** It exposes a trade-off, not a clean corrective signal:
the chosen drug is often a correct answer elsewhere, and much of the apparent
focal improvement comes with worse margins on those known-correct uses.

Same 16 TF2 TRAIN cases, original 2,048-row batches, frozen own H35F student and
ten own teachers. Five arms keep pool size and hard/random substitution
multiplicity matched. Candidate-specific protection masks only false-negative
CE entries for the chosen drug; original distillation and trajectory retained.

## Results

All numbers below are temperature-independent margin directional derivatives
along negative total-loss gradients, **not actual updates, recovered ranks or
MRR gains**. Positive means locally corrective. Focal counts cover 16 queries;
known-correct counts use the mean over eight fixed probes in each case.

| Arm | Focal corrective / 16 | Mean focal derivative | Known-correct groups harmed / 16 | Mean known-correct derivative |
| --- | ---: | ---: | ---: | ---: |
| Original random baseline | 13 | +0.01739158 | 6 | −0.00661682 |
| Shared hard, unprotected | 16 | +0.03651013 | 15 | −0.02787698 |
| Shared hard, protected | 12 | +0.00832918 | 8 | +0.00468392 |
| Matched random, unprotected | 10 | −0.00203347 | 1 | +0.01474169 |
| Matched random, protected | 10 | −0.00203397 | 1 | +0.01474443 |

Protection improves the hard drug's known-correct group mean versus unprotected
hard in all 16 cases, while reducing focal correction in all 16. Compared with
protected random, protected hard improves focal correction in all 16, but
worsens that known-correct group mean in all 16. This is a trade-off, not a
claim that every individual protected answer is harmed.

Of the three baseline cases with harmful focal gradients, protection retains
the sign correction for only one. The other two remain harmful. Two additional
cases whose original random pools already contained the competitor become
harmful under protection, giving 12/16 overall versus baseline 13/16. These are
descriptive decompositions, not independently confirmed subgroups.

The model margin against the saved pipeline winner is corrective in 13/16
baseline, 16/16 unprotected-hard, 12/16 protected-hard and 10/16 each random-control
arm. This is **not** a derivative of the complete retrieval pipeline: its
features and normalization were not differentiated here.

Background group means are corrective in 13/16 cases for every arm. Individual
background margins are harmful in 77/256 baseline, 76/256 each hard arm, and
77/256 each random arm. This small within-relation probe does not establish
absence of wider collateral damage.

Across 128 known-correct hard-drug probe occurrences, harmful margins occur
in 72 baseline, 102 unprotected-hard, 66 protected-hard, and 45 each random arm.
These repeated occurrences are not independent observations.

## Mechanism and limitations

The hard drug is a known TRAIN answer for **18,150/32,768 batch-row occurrences
(55.39%)**, ranging from 261 to 1,759 rows per batch. These are edge-weighted
sampled rows, not 18,150 distinct queries. Broadcasting that drug as a negative
without protection imposes a false-negative CE penalty on all those rows.

Across the 16 cases, unprotected hard has mean focal positive-score derivative
+0.00551103 and competitor-score derivative −0.03099911; its competitor-norm
derivative is −0.03025497. The corresponding protected values are +0.00527212,
−0.00305707 and +0.00189288. This supports a substantial shared-entity suppression
effect under the unprotected gradient. It does not prove that norm is the sole
cause, or that an actual Adam step would reproduce these changes.

Even protected hard versus protected random gains +0.01036316 mean focal
margin almost entirely from a lower competitor-score derivative (−0.01031698),
not a higher correct-score derivative (+0.00004618). Its known-correct group
mean simultaneously falls by −0.01006051. Direct masked CE gradients are zero,
but shared parameters and unmasked KD/trajectory can still affect those answers.

Important control limitation: **15/16 uniform random control drugs have zero
TRAIN degree for the tested relation**; the remaining drug has degree 22.
Hard-drug relation degrees range 31–464. Only one case has random-known probes
(eight occurrences). The control is correctly typed and size/multiplicity
matched, but not relation-activity/degree matched. Do not interpret the hard
versus random difference as isolating hardness alone or claim random protection
is generally ineffective.

Original case, local drug 1381 / relation 38 / correct answer 786 / competitor
1528: competitor is a known TRAIN answer in 1,212/2,048 batch rows. Focal margin
derivative is +0.01181957 baseline, +0.04727439 unprotected hard, +0.02187069
protected hard. Its eight known-correct uses have mean margin derivative
+0.00883602, −0.02631939 and −0.00056398 respectively. The focal gain cannot be
read in isolation from those other queries.

## Decision and possible next test

Stop here; no training launched. This neither proves hard-negative training
cannot work nor supports a k/block-size change. It shows that the earlier
16/16 result is insufficient: shared suppression helps the selected query
while interfering with legitimate uses of the same drug.

A possible next experiment, requiring a separate approval/recipe, is per-query
hard candidates selected among relation-active drugs, excluding that query's
known TRAIN answers, against an activity-matched random control. Keep KD and
matched training effort. This is different from broadcasting one focal drug
across every row; prior TF2 already showed that inserting it only into the
focal row at the original weight did not repair the three whole-batch failures.
Whether query-specific selection across the other rows resolves interference
is untested. Do not infer an MRR gain or begin a pilot from this report alone.

## Verification and artifacts

- 128 synthetic/regression tests passed; CUDA synthetic JVP check passed.
- All 80 arms completed. Focal JVP/reverse-mode parity and original TF2
  baseline/forced projections reproduced; protected CE derivatives zero;
  KD/trajectory exactly unchanged for each matched protected/raw pool.
- Main process completed in 36.46 seconds including final hash checks;
  peak allocated GPU memory 1.663 GiB. GPU process released afterward.
- Endpoint audit reconstructed TRAIN masks, random pools, probe identities,
  all summaries and directly recomputed all 424 frozen-model probe score/rank
  occurrences on CPU. It did not independently replay every GPU gradient.
- Student and all ten own teacher state hashes unchanged; parameter `.grad`
  fields remained empty. Sources and inputs hash-verified before/after.
- Only TRAIN edges and node counts opened from the dataset. No optimizer,
  model changes, VALID/TEST access, external models or submission changes.

LR **0**. Prior pipeline VALID MRR **0.8582166512** unchanged, not newly measured.
No leaderboard position or generalization claim.

Artifacts in [s0](s0): [receipt](s0/prerun.json), [plans](s0/plans.json),
[all readouts/derivatives](s0/records.jsonl), [summary](s0/summary.json),
[runner audit](s0/audit.json), [endpoint audit](s0/endpoint_audit.json),
[progress](s0/progress.jsonl).
