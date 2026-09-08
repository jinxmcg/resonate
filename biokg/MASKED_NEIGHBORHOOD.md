# MN1: fresh single model and matched masked-neighborhood continuation

2026-09-07. User approved a single-model comparison, no teachers/distillation,
on their Vast RTX 5090 at 79.19.114.194 (CLI-resolved instance 50109054,
direct SSH port 50034). Keep TH1's fixed 95/5 pair-grouped TRAIN reservation.
Only fit/holdout derived arrays, metadata and explicitly listed code go to
the isolated remote directory. No old checkpoint, VALID, TEST or external
trained artifact. uv Python 3.12, torch 2.11.0+cu128, numpy 2.5.2, scipy 1.18.1.
Read the remote /etc/vast-agents-guide.md; use supervisor for the finite job,
autorestart disabled, back up artifacts locally before handing off.

## Question and fixed implementation

Does training with a target-hidden neighborhood improve the **ordinary single
model's** ranking on unseen TRAIN-held-out pairs? This first version uses
same-relation **drug–drug** neighbors only; other relations keep ordinary loss.
No inference neighborhood, extra operator bank, attention, teacher or ensemble.
Failure rejects this version, not all masked graph models.

Fresh base: sparse-table ResonatE, k=12, free 4x4 complex blocks, fp32,
27,124,129 real parameters; seed 0. 50,000 steps on fit only, batch 2,048,
4,096 shared typed random negatives. Original CE + 0.1 trajectory; RowAdagrad
table LR 0.3, Adam other LR 0.005, cosine to zero, global gradient clip 1.
Natural edge-weighted relation selection, uniform prediction direction.

Fork two independent model **and optimizer state** copies from that base.
12,500 additional updates each; shared batches/negatives (NumPy seed 36850),
retained optimizer accumulators/moments, common restarted cosine LR from
0.06 table / 0.001 Adam to zero. Equal update/exposure budget, not equal GPU
time. Neither arm resumes from an old full-TRAIN checkpoint.

Control: original CE + 0.1 trajectory.
Masked: on drug–drug queries, form the unique directed fit-neighbor list for
source and relation. Remove the target before sampling; remove every duplicate
copy through deduplication. Both directions index only fit edges. Because only
this one-hop list is read, no target/reverse/cross-relation pair copy can enter
the context. Sample 8 remaining neighbors uniformly with replacement using a
separate torch generator (seed 36851), never the batch/negative RNG.

Let q be the original normalized query, and c the normalized mean of the
sampled neighbors' normalized entity embeddings. Auxiliary query is
`q_context = normalize(0.75*q + 0.25*c)`. Use the same target and negative
candidate columns as ordinary CE. Per-row loss is
`0.75*CE(q) + 0.25*CE(q_context) + 0.1*trajectory(q, E_target)`.
Rows with no remaining neighbors, and all non-drug–drug relations, use the
original loss exactly. All student embeddings/operators co-adapt; no detached
teacher. Total CE weight stays one. Original negative sampling remains
unfiltered in BOTH arms; this does not test or solve all false-negative
conflicts from TF3. Context may also be present among negative candidates.
Train-edge masking does not erase normal prior fit-label learning in weights.

At inference remove the context operation: both arms use exactly q and the
same ordinary candidate-scoring function. The intervention is training-only
neighborhood assistance; transfer to the unassisted scorer is the hypothesis.

## Frozen reporting / no held-out gradients

Training process may open only fit arrays and metadata, code and its own new
output checkpoints. Holdout artifact is denied even if present on disk.
Only a separate evaluation process, after both endpoints are saved, may open
holdout. No optimizer/model updates in evaluation. No official VALID/TEST.

Progress at least once per minute: stage, steps, both optimizer LRs, loss,
elapsed time. Every 5,000 steps and endpoint, plain-scorer TRAIN-probe MRR on
1,024 fit rows selected with RNG 36852 and 500 typed negatives; clearly label
in-sample TRAIN probe, no selection/stopping from that score. Probe operations
use private RNG and no_grad; never alter main training or context RNG states.
No holdout MRR exists until the frozen endpoint comparison.

Evaluate every reserved original row in both directions, with identical 500
typed negatives across base/control/masked. Candidates are uniform with
replacement, excluding source, designated target and known fit answers for
that directed query. Deterministic per-relation/direction RNG seed 36853;
fixed chunk 256. Do not consult other holdout answers to filter negatives.
Unknown candidates can be false negatives. Preserve cold endpoints and self
links. Same scorer for every column; official average-tie reciprocal ranks.
Save per-row ranks, original row/pair IDs and candidate-stream hash.

Primary: masked minus continued-control mean MRR over ALL held-out rows, both
directions averaged per original row. Descriptive pair-cluster bootstrap,
2,000 replicates seed 36854, resampling unordered pairs including duplicate
and cross-relation rows together. Report both versus frozen base, Hits@1/10,
drug–drug slice, per relation/direction and cold-endpoint slice. No subgroup
promotion or leaderboard score projection. Paired query uncertainty is not
model-seed uncertainty. This internal development result is not official MRR.

One fail-fast endpoint gate: delta >= 0.001 and lower paired 95% bound > 0,
and masked MRR >= frozen base. A pass permits proposing seed confirmation,
not automatic promotion/training. A fail stops this version; no tuning after
holdout outcomes. Before real training, stop on synthetic failure, nonfinite
tensors, masking leakage, device mismatch, inadequate memory or impractical
smoke throughput; never silently alter precision/neighborhood size/budget.

## Verification

Synthetic tests: exact target exclusion including duplicates/reverse rows,
sampling equals explicit leave-one-out list, zero-neighbor original-loss
parity, no extra parameters, candidate symmetry, unchanged empty-context
scoring, detached evaluation and RNG isolation, deterministic streams,
typed negative rejection, disjoint fit provenance, file guard, copied model
and optimizer states, finite sparse gradients and checkpoint roundtrip.
Full-size synthetic GPU smoke before real data updates.

Pin code/protocol/fit/manifest hashes before training, verify after. Record
base/two fork digests, optimizer digests and update counters, batch stream
hash, context coverage and target-exclusion checks, timings/memory and actual
LRs. Save all three standalone checkpoints. Endpoint auditor verifies frozen
input/source/checkpoint/rank hashes, standalone scoring reconstruction,
recorded schedule and initial-state parity, pair-cluster summary arithmetic,
split separation and lack of holdout access during training. Do not claim
independent replay of every optimizer update. No deletion of old experiments.
