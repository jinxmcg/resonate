# BC1: fixed half-strength blend-selection stability check

Registered 2026-09-06 before computing new split results. User approved
confirmation of RF1's predeclared half-strength control. This is a check of
the exact selection **method**, not evaluation of old weights on their own
fit labels, a larger weight search, or a new model-seed campaign.

## Fixed comparison and data boundary

Use our unchanged H35F single A and the audited FC1 fp32 normalized VALID
feature cache, channels 3,4,5,6,7: model score A, analogy max/top3 and
ordinary Jaccard max/top3. Its checkpoint SHA256 is
`df7c783e7b8a6110f126bca1c60d89209735accbaa5ffc0b362f2cab52d9d2f1`.
Do not deserialize any model, teacher, dataset split or external trained
artifact. The cache's graph features were built from TRAIN only. Verify
its recorded provenance, hashes, metadata and prior endpoint audit.
Block all opens beneath the dataset directory, including TRAIN/VALID.
Use only the named VALID caches, not legacy TEST predictions.

Baseline members: A and four ordinary retrieval members R_j. Treatment:
A and `0.5*R_j + 0.5*A`, using RF1's exact fp32 HalfFeature implementation.
No further normalization. Reuse FC1 fit_recipe without modifications:
identical finite candidates, eta {20,50,100}, min 2,000 fit queries for
local groups, hierarchy, fallbacks, tie-breaking and convexity constraints.
Do not search interpolation strengths, add candidates, alter group switches,
combine rarity/support, train, or use validation gradients.

## Repeated two-way partitions

Fix partition RNG seeds (0,1,2). For each, sample one mask
`default_rng(seed).random(162886) < 0.5`, tiled across head/tail directions.
Fold 0 fits that mask and reports its complement; fold 1 reverses them.
Each triple is reported once per partition, with both directions together.
Refit both arms on exactly the same fit rows in every fold. Save each
recipe before any report scoring. Never choose a best partition or fold.

Seed 0/fold 0 must exactly reproduce FC1/RF1 saved recipes and report ranks;
recompute it as a parity check, not new evidence. The reversed original
split and both new partitions supply the additional split-stability checks.
Save report ranks only, with one fully covered out-of-fit vector per arm
and partition. Do not fit or report a full-VALID selected recipe.

## Fixed summary and interpretation

One primary contrast: half-strength minus baseline. Compute reciprocal-rank
deltas, then average across the two directions and all three partitions
**within each original triple**. Report their mean and a 2,000-replicate
whole-triple paired bootstrap 95% interval, seed 3631. Do not count repeated
observations as independent samples or average candidate scores into an
inference ensemble. This is a summary of repeated selection evaluations.

Report each fold, all three complete partitions, and the pooled summary;
include mean per-arm MRR/Hits1/Hits10, direction summaries, and per-partition
top-one recoveries/losses. No post-hoc relation-specific switches. Preserve
the previous practical +0.001 threshold: useful-gain flag requires mean
delta >= 0.001, interval lower bound > 0, and all three partition deltas > 0.
Also report consistent-positive separately, so a smaller stable gain is not
misrepresented as zero. No automatic promotion whatever the result.

Every validation triple has prior experimental exposure. New partitions do
not create an untouched holdout or erase adaptive selection. Intervals are
descriptive, conditional on saved predictions; they do not capture model
training variability, adaptive research selection, or all dependencies
induced by shared fitting data. This is split stability on one fixed model,
not model-seed robustness, a final TEST estimate or proof of leaderboard rank.

## Verification and stopping

Before execution, test paired complementary masks and one-report-per-query
coverage, treatment/old-implementation parity, fit/report poisoning isolation
for both fold directions and arms, sparse fallback, official average ties,
candidate permutation/duplicates, correct repeated-triple aggregation and
deterministic bootstrap, invalid rank/coverage rejection, and dataset guard.
Pin protocol, runner, tests, auditor and used input hashes in a prereceipt.
Verify unchanged source/cache/checkpoint hashes again after execution.

CPU only; LR=0/not applicable, zero optimizer updates and no GPU operations.
Log completed-fold MRRs and progress without report-dependent decisions.
Independently refit all saved recipes and reproduce all report ranks,
coverage, summaries and bootstrap interval from the pinned records.
Stop after six folds and audit. No extra seed, new blend sweep, training,
full-VALID refit, TEST evaluation or submission modification is authorized
by this protocol. Any next experiment needs a separate decision.
